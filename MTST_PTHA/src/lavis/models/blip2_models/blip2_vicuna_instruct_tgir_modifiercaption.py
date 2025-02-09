"""
Requires Transformer 4.28 and above, implementation may change according the Llama implementation
"""
import logging
import string
from packaging import version

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

import transformers
from torch.nn import functional as F

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
from lavis.models.blip_models.blip_outputs import BlipOutput, BlipOutputFeatures,BlipTGIROutput

import logging
from operator import itemgetter
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F

import contextlib

import os

import time
import datetime

from lavis.models.base_model import all_gather_with_grad, concat_all_gather
import numpy as np 
from typing import List, Tuple
from tqdm import tqdm
import lavis.common.dist_utils as dist_utils
from lavis.common.logger import MetricLogger
@registry.register_model("blip2_vicuna_instruct_tgir_modifiercaption")
class Blip2VicunaInstructTGIR4MC(Blip2Base):
    """
    BLIP2 Vicuna model.
    Supported model types:
        - vicuna7b
        - vicuna13b
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_vicuna_instruct_tgir_modifiercaption", "vicuna7b")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "vicuna7b": "configs/models/blip2/blip2_instruct_vicuna7b.yaml",
        "vicuna13b": "configs/models/blip2/blip2_instruct_vicuna13b.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        llm_model="",
        prompt="",
        max_txt_len=128,
        max_output_txt_len=256,
        apply_lemmatizer=False,
        qformer_text_input=True,
        embed_dim = 256
    ):
        super().__init__()
        transformers_version = version.parse(transformers.__version__)
        assert transformers_version >= version.parse("4.28"), "BLIP-2 Vicuna requires transformers>=4.28"        
        from transformers import LlamaTokenizer
        from lavis.models.blip2_models.modeling_llama import LlamaForCausalLM
        
        self.tokenizer = self.init_tokenizer(truncation_side="left")
        
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")
        
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )


        if not qformer_text_input:
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
        else:
            self.Qformer.resize_token_embeddings(len(self.tokenizer))
        self.Qformer.cls = None

        # self.itm_former, _ = self.init_Qformer(
        #     num_query_token, self.visual_encoder.num_features
        # )



        # self.itm_former.bert.embeddings.word_embeddings = None
        # self.itm_former.bert.embeddings.position_embeddings = None
        # for layer in self.itm_former.bert.encoder.layer:
        #     layer.output = None
        #     layer.intermediate = None
       
        # self.itm_former.cls = None

        self.llm_tokenizer = LlamaTokenizer.from_pretrained(llm_model, use_fast=False, truncation_side="left")
        self.llm_model = LlamaForCausalLM.from_pretrained(
            llm_model, torch_dtype=torch.float16
        )
        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llm_tokenizer.add_special_tokens({'bos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'eos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'unk_token': '</s>'})

        # self.llm_tokenizer.pad_token = self.llm_tokenizer.unk_token

        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))

        # self.eos_token_id = self.llm_tokenizer(
        #     self.llm_tokenizer.eos_token, add_special_tokens=False
        # ).input_ids[0]

        for name, param in self.llm_model.named_parameters():
            param.requires_grad = False

        self.llm_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llm_model.config.hidden_size
        )
        # self.vision_proj = nn.Linear(
        #     3*self.visual_encoder.num_features, self.visual_encoder.num_features
        # )

        self.embed_dim = embed_dim
        # self.temp = nn.Parameter(0.07 * torch.ones([]))
        self.max_txt_len = max_txt_len
        self.max_output_txt_len = max_output_txt_len
        self.prompt = prompt
        prompt_tokens = self.llm_tokenizer(self.prompt, return_tensors="pt")
        self.prompt_length = prompt_tokens.attention_mask.sum(1)

        self._lemmatizer = None
        self.qformer_text_input = qformer_text_input

    def concat_text_input_output(self, input_ids, input_atts, output_ids, output_atts):
        input_part_targets_len = []
        llm_tokens = {"input_ids": [], "attention_mask": []}
        for i in range(input_ids.size(0)):
            this_input_ones = input_atts[i].sum()
            input_part_targets_len.append(this_input_ones)
            llm_tokens['input_ids'].append(
                torch.cat([
                    input_ids[i][:this_input_ones],
                    output_ids[i][1:],
                    input_ids[i][this_input_ones:]
                ])
            )
            llm_tokens['attention_mask'].append(
                torch.cat([
                    input_atts[i][:this_input_ones],
                    output_atts[i][1:],
                    input_atts[i][this_input_ones:]
                ])
            )
        llm_tokens['input_ids'] = torch.stack(llm_tokens['input_ids'])
        llm_tokens['attention_mask'] = torch.stack(llm_tokens['attention_mask'])
        return llm_tokens, input_part_targets_len

    def forward(self, samples):
        # print('-----------------')
        # print(samples["text_input"])
        # print(samples["text_output"])
        # print('-----------------')

        reference_image = samples["reference_image"]
        target_image = samples["target_image"]
        # text = ["How to change from one image to another?" for i in samples["text_input"]]
        text = ["How to change from one image to the other image?" for i in samples["text_input"]]

        with self.maybe_autocast():
            reference_image_embeds = self.ln_vision(self.visual_encoder(reference_image))
            target_image_embeds = self.ln_vision(self.visual_encoder(target_image))


        reference_image_atts = torch.ones(reference_image_embeds.size()[:-1], dtype=torch.long).to(reference_image.device)
        target_image_atts = torch.ones(target_image_embeds.size()[:-1], dtype=torch.long).to(reference_image.device)
        tar_query_tokens = self.query_tokens.expand(
                target_image_embeds.shape[0], -1, -1
            )
        ref_query_tokens = self.query_tokens.expand(
                reference_image_embeds.shape[0], -1, -1
            )

        reference_query_output = self.Qformer.bert(
            query_embeds=ref_query_tokens,
            encoder_hidden_states=reference_image_embeds,
            encoder_attention_mask=reference_image_atts,
            use_cache=True,
            return_dict=True,
        )
        ref_query_embed = reference_query_output.last_hidden_state[:,:ref_query_tokens.size(1),:]
        inputs_llm_ref = self.llm_proj(ref_query_embed)
        ref_atts_llm = torch.ones(inputs_llm_ref.size()[:-1], dtype=torch.long).to(reference_image.device)
    

        target_query_output = self.Qformer.bert(
            query_embeds=tar_query_tokens,
            encoder_hidden_states=target_image_embeds,
            encoder_attention_mask=target_image_atts,
            use_cache=True,
            return_dict=True,
        )
        tar_query_embed = target_query_output.last_hidden_state[:,:tar_query_tokens.size(1),:]
        inputs_llm_tar = self.llm_proj(tar_query_embed)
        tar_atts_llm = torch.ones(inputs_llm_tar.size()[:-1], dtype=torch.long).to(reference_image.device)


        image_embeds = torch.cat((reference_image_embeds,target_image_embeds),dim = 1)
        # image_embeds = self.vision_proj(image_embeds)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(reference_image.device)


        bs = reference_image.size(0)

        query_tokens = self.query_tokens.expand(reference_image_embeds.shape[0], -1, -1)
        if self.qformer_text_input:
            text_Qformer = self.tokenizer(
                text,
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(reference_image.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(reference_image.device)
            Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask],dim=1)


            query_output = self.Qformer.bert(
                text_Qformer.input_ids,#only text
                attention_mask=Qformer_atts,#query+text
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,#768
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
        else:
            query_output = self.Qformer.bert(#only reference_image
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
        query_embed = query_output.last_hidden_state[:,:query_tokens.size(1),:]
        inputs_llm = self.llm_proj(query_embed)
        atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(reference_image.device)



        self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = 'left'#左侧截断
        text_input_tokens = self.llm_tokenizer(
            # samples['text_input'],
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(reference_image.device)

        self.llm_tokenizer.truncation_side = 'right'#右侧截断
        text_output_tokens = self.llm_tokenizer(
            [t + self.llm_tokenizer.eos_token for t in samples['text_input']],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_output_txt_len,
        ).to(reference_image.device)

        llm_tokens, input_part_targets_len = self.concat_text_input_output(
            text_input_tokens.input_ids,
            text_input_tokens.attention_mask,
            text_output_tokens.input_ids,
            text_output_tokens.attention_mask,
        )

        # do not apply loss to the padding
        targets = llm_tokens['input_ids'].masked_fill(
            llm_tokens['input_ids'] == self.llm_tokenizer.pad_token_id, -100
        )

        # do not apply loss to the text input (i.e., instruction)
        for i, l in enumerate(input_part_targets_len):
            targets[i][:l] = -100

        # do not apply loss to the query tokens
        empty_targets = (
            torch.ones(atts_llm.size(), dtype=torch.long).to(reference_image.device).fill_(-100)
        )
        empty_targets2 = (
            torch.ones(ref_atts_llm.size(), dtype=torch.long).to(reference_image.device).fill_(-100)
        )
        empty_targets3 = (
            torch.ones(tar_atts_llm.size(), dtype=torch.long).to(reference_image.device).fill_(-100)
        )
        targets = torch.cat([empty_targets2,empty_targets3,empty_targets, targets], dim=1)

        inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens['input_ids'])
        # pad_embeds = self.llm_model.get_input_embeddings()("[PAD]")
        # atts_pad = torch.ones(pad_embeds.size()[:-1], dtype=torch.long).to(reference_image.device).fill_(-100)

        # inputs_embeds = torch.cat([inputs_llm_ref,pad_embeds,inputs_llm_tar,pad_embeds,inputs_llm, inputs_embeds], dim=1)
        # attention_mask = torch.cat([ref_atts_llm,atts_pad,tar_atts_llm,atts_pad,atts_llm, llm_tokens['attention_mask']], dim=1)
        inputs_embeds = torch.cat([inputs_llm_ref,inputs_llm_tar,inputs_llm, inputs_embeds], dim=1)
        attention_mask = torch.cat([ref_atts_llm,tar_atts_llm,atts_llm, llm_tokens['attention_mask']], dim=1)



        with self.maybe_autocast():
            outputs = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )

        loss = outputs.loss 
        return {"loss": loss}



    # def generate_using_embedding(
    #     self,
    #     samples,
    #     use_nucleus_sampling=False,
    #     num_beams=5,
    #     max_length=256,
    #     min_length=1,
    #     top_p=0.9,
    #     repetition_penalty=1.5,
    #     length_penalty=1,
    #     num_captions=1,
    #     temperature=1,
    # ):
    #     self.llm_tokenizer.padding_side = "left"
    #     image_embeds = samples["image_embeds"]
    #     bs = image_embeds.size(0)
    
    #     query_tokens = self.query_tokens.expand(bs, -1, -1)
        


    #     inputs_llm = self.llm_proj(image_embeds[:,:query_tokens.size(1),:])
    #     atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(image_embeds.device)


    #     with self.maybe_autocast():

    #         inputs_embeds = inputs_llm
    #         attention_mask = atts_llm

    #         outputs = self.llm_model.generate(
    #             inputs_embeds=inputs_embeds,
    #             attention_mask=attention_mask,
    #             do_sample=use_nucleus_sampling,
    #             top_p=top_p,
    #             temperature=temperature,
    #             num_beams=num_beams,
    #             max_length=max_length,
    #             min_length=min_length,
    #             # eos_token_id=self.eos_token_id,
    #             repetition_penalty=repetition_penalty,
    #             length_penalty=length_penalty,
    #             num_return_sequences=num_captions,
    #         )

    #     outputs[outputs == 0] = 2 # convert output id 0 to 2 (eos_token_id)
    #     output_text = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
    #     output_text = [text.strip() for text in output_text]

    #     return output_text
    @torch.no_grad()
    def generate_modifier(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=256,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1,
        num_captions=1,
        temperature=1,
    ):
        self.llm_tokenizer.padding_side = "left"


        # prompt = samples["text_input"]
        prompt = "How to change from one image to the other image?"


        # image = samples["image"]
        reference_image = samples["reference_image"]
        target_image = samples["target_image"]

        bs = reference_image.size(0)
        if isinstance(prompt, str):
            prompt = [prompt] * bs
        else:
            assert len(prompt) == bs, "The number of prompts must be equal to the batch size."

        # For TextCaps
        if "ocr_tokens" in samples.keys() and "{}" in prompt[0]:
            prompt = [p.format(', '.join(samples['ocr_tokens'][i][:30])) for i, p in enumerate(prompt)]

        query_tokens = self.query_tokens.expand(bs, -1, -1)
        if self.qformer_text_input:
            # remove ocr tokens in q_former (for eval textvqa)
            # qformer_prompt = prompt
            # qformer_prompt = ['Question: ' + qp.split(' Question: ')[1] for qp in qformer_prompt]

            text_Qformer = self.tokenizer(
                prompt,
                padding='longest',
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(reference_image.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(reference_image.device)
            Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)




        with self.maybe_autocast():
            reference_image_embeds = self.ln_vision(self.visual_encoder(reference_image))
            target_image_embeds = self.ln_vision(self.visual_encoder(target_image))

        reference_image_atts = torch.ones(reference_image_embeds.size()[:-1], dtype=torch.long).to(reference_image.device)
        target_image_atts = torch.ones(target_image_embeds.size()[:-1], dtype=torch.long).to(reference_image.device)
        tar_query_tokens = self.query_tokens.expand(
                target_image_embeds.shape[0], -1, -1
            )
        ref_query_tokens = self.query_tokens.expand(
                reference_image_embeds.shape[0], -1, -1
            )
        reference_query_output = self.Qformer.bert(
            query_embeds=ref_query_tokens,
            encoder_hidden_states=reference_image_embeds,
            encoder_attention_mask=reference_image_atts,
            use_cache=True,
            return_dict=True,
        )
        ref_query_embed = reference_query_output.last_hidden_state[:,:ref_query_tokens.size(1),:]
        inputs_llm_ref = self.llm_proj(ref_query_embed)
        ref_atts_llm = torch.ones(inputs_llm_ref.size()[:-1], dtype=torch.long).to(reference_image.device)
    
        target_query_output = self.Qformer.bert(
            query_embeds=tar_query_tokens,
            encoder_hidden_states=target_image_embeds,
            encoder_attention_mask=target_image_atts,
            use_cache=True,
            return_dict=True,
        )
        tar_query_embed = target_query_output.last_hidden_state[:,:tar_query_tokens.size(1),:]
        inputs_llm_tar = self.llm_proj(tar_query_embed)
        tar_atts_llm = torch.ones(inputs_llm_tar.size()[:-1], dtype=torch.long).to(reference_image.device)


        image_embeds = torch.cat((reference_image_embeds,target_image_embeds),dim = 1)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(reference_image.device)


        if self.qformer_text_input:
            query_output = self.Qformer.bert(
                text_Qformer.input_ids,
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
        else:
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

        inputs_llm = self.llm_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])
        atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(reference_image.device)
        

        llm_tokens = self.llm_tokenizer(
            prompt,
            padding="longest",
            return_tensors="pt"
        ).to(reference_image.device)

        with self.maybe_autocast():

            inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens['input_ids'])
            # pad_embeds = self.llm_model.get_input_embeddings()("[PAD]")
            # atts_pad = torch.ones(pad_embeds.size()[:-1], dtype=torch.long).to(reference_image.device).fill_(-100)


            # inputs_embeds = torch.cat([inputs_llm_ref,pad_embeds,inputs_llm_tar,pad_embeds,inputs_llm, inputs_embeds], dim=1)
            # attention_mask = torch.cat([ref_atts_llm,atts_pad,tar_atts_llm,atts_pad,atts_llm, llm_tokens['attention_mask']], dim=1)
            inputs_embeds = torch.cat([inputs_llm_ref,inputs_llm_tar,inputs_llm, inputs_embeds], dim=1)
            attention_mask = torch.cat([ref_atts_llm,tar_atts_llm,atts_llm, llm_tokens['attention_mask']], dim=1)

            outputs = self.llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                # eos_token_id=self.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )

        outputs[outputs == 0] = 2 # convert output id 0 to 2 (eos_token_id)
        output_text = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output_text = [text.strip() for text in output_text]

        return output_text
        # return inputs_embeds,attention_mask,'output_text'

    # def predict_answers(
    #     self,
    #     samples,
    #     num_beams=5,
    #     inference_method="generate",
    #     max_len=10,
    #     min_len=1,
    #     num_ans_candidates=128,
    #     answer_list=None,
    #     prompt="",
    #     length_penalty=0,
    #     **kwargs
    # ):
    #     if isinstance(samples["text_input"], str):
    #         samples["text_input"] = [samples["text_input"]]

    #     if prompt:
    #         if prompt.count("{}") == 2:
    #             if 'ocr_tokens' in samples:
    #                 text_input = [
    #                     prompt.format(', '.join(samples['ocr_tokens'][i][:30]), samples["text_input"][i])
    #                 for i in range(len(samples["text_input"]))]
    #             elif 'choices' in samples:
    #                 text_input = []
    #                 for i in range(len(samples["text_input"])):
    #                     this_choices = [f"({string.ascii_lowercase[j]}) {ch}" for j, ch in enumerate(samples["choices"][i])]
    #                     this_choices = " ".join(this_choices)
    #                     text_input.append(prompt.format(samples["text_input"][i], this_choices))
    #         else:
    #             text_input = [prompt.format(question) for question in samples["text_input"]]
    #     else:
    #         text_input = samples["text_input"]

    #     samples["prompt"] = text_input

    #     output_text = self.generate(
    #         samples,
    #         num_beams=num_beams,
    #         max_length=max_len,
    #         min_length=min_len,
    #         length_penalty=length_penalty
    #     )

    #     if "apply_lemmatizer" in samples.keys() and samples["apply_lemmatizer"]:
    #         output_text = self._lemmatize(output_text)

    #     return output_text

    # def predict_class(
    #     self,
    #     samples,
    #     candidates,
    #     n_segments=1,
    # ):
    #     self.llm_tokenizer.padding_side = "left"

    #     # If candidates is a list of lists, each sample has its candidates, then we need to iterate one by one
    #     if type(candidates[0]) == list:
    #         results = []

    #         for i in range(samples["image"].size(0)):
    #             this_sample = {
    #                 "image": samples["image"][i].unsqueeze(0),
    #                 "prompt": samples["prompt"],
    #             }

    #             if "text_input" in samples.keys():
    #                 this_sample["text_input"] = [samples["text_input"][i]]

    #             if 'context' in samples.keys():
    #                 this_sample['context'] = [samples["context"][i]]

    #             if 'history' in samples.keys():
    #                 this_sample['history'] = [samples["history"][i]]

    #             if 'caption' in samples.keys():
    #                 this_sample['caption'] = [samples["caption"][i]]

    #             this_result = self._predict_class(this_sample, candidates[i], n_segments)
    #             results.append(this_result)

    #         try:
    #             results = torch.cat(results, dim=0)
    #         except:
    #             results = [res.tolist()[0] for res in results]

    #         return results

    #     return self._predict_class(samples, candidates, n_segments)

    # def _predict_class(
    #     self,
    #     samples,
    #     candidates,
    #     n_segments=1,
    # ):
    #     image = samples["image"]
    #     prompt = samples["prompt"]

    #     bs = image.size(0)

    #     if isinstance(prompt, str):
    #         prompt = [prompt] * bs
    #     else:
    #         assert len(prompt) == bs, "The number of prompts must be equal to the batch size."

    #     if "text_input" in samples.keys():
    #         if type(samples["text_input"][0]) == list:
    #             prompt = [prompt[i].format(*samples["text_input"][i]) for i in range(len(prompt))]
    #         else:
    #             prompt = [prompt[i].format(samples["text_input"][i]) for i in range(len(prompt))]

    #     # scienceqa
    #     if 'context' in samples.keys() and samples['context'] != '':
    #         prompt = [f'context: {samples["context"][i]}. {prompt[i]}' for i in range(len(prompt))]

    #     # visual dialog
    #     if 'history' in samples.keys() and samples['history'][0] != '':
    #         prompt = [f'dialog history: {samples["history"][i]}\n{prompt[i]}' for i in range(len(prompt))]

    #     if 'caption' in samples.keys() and samples['caption'][0] != '':
    #         prompt = [f'This image has the caption "{samples["caption"][i]}". {prompt[i]}' for i in range(len(prompt))]

    #     query_tokens = self.query_tokens.expand(bs, -1, -1)
    #     if self.qformer_text_input:
    #         text_Qformer = self.tokenizer(
    #             prompt,
    #             padding='longest',
    #             truncation=True,
    #             max_length=self.max_txt_len,
    #             return_tensors="pt"
    #         ).to(image.device)
    #         query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
    #         Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

    #     if image.dim() == 5:
    #         inputs_llm, atts_llm = [], []
    #         for j in range(image.size(2)):
    #             this_frame = image[:,:,j,:,:]
    #             with self.maybe_autocast():
    #                 frame_embeds = self.ln_vision(self.visual_encoder(this_frame))
    #                 frame_atts = torch.ones(frame_embeds.size()[:-1], dtype=torch.long).to(image.device)

    #             if self.qformer_text_input:
    #                 frame_query_output = self.Qformer.bert(
    #                     text_Qformer.input_ids,
    #                     attention_mask=Qformer_atts,
    #                     query_embeds=query_tokens,
    #                     encoder_hidden_states=frame_embeds,
    #                     encoder_attention_mask=frame_atts,
    #                     return_dict=True,
    #                 )
    #             else:
    #                 frame_query_output = self.Qformer.bert(
    #                     query_embeds=query_tokens,
    #                     encoder_hidden_states=frame_embeds,
    #                     encoder_attention_mask=frame_atts,
    #                     return_dict=True,
    #                 )

    #             frame_inputs_llm = self.llm_proj(frame_query_output.last_hidden_state[:,:query_tokens.size(1),:])
    #             frame_atts_llm = torch.ones(frame_inputs_llm.size()[:-1], dtype=torch.long).to(image.device)
    #             inputs_llm.append(frame_inputs_llm)
    #             atts_llm.append(frame_atts_llm)
    #         inputs_llm = torch.cat(inputs_llm, dim=1)
    #         atts_llm = torch.cat(atts_llm, dim=1)
    #     else:
    #         with self.maybe_autocast():
    #             image_embeds = self.ln_vision(self.visual_encoder(image))
    #         image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

    #         if self.qformer_text_input:
    #             query_output = self.Qformer.bert(
    #                 text_Qformer.input_ids,
    #                 attention_mask=Qformer_atts,
    #                 query_embeds=query_tokens,
    #                 encoder_hidden_states=image_embeds,
    #                 encoder_attention_mask=image_atts,
    #                 return_dict=True,
    #             )
    #         else:
    #             query_output = self.Qformer.bert(
    #                 query_embeds=query_tokens,
    #                 encoder_hidden_states=image_embeds,
    #                 encoder_attention_mask=image_atts,
    #                 return_dict=True,
    #             )

    #         inputs_llm = self.llm_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])
    #         atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(image.device)

    #     self.llm_tokenizer.padding_side = "right"
    #     self.llm_tokenizer.truncation_side = 'left'
    #     text_input_tokens = self.llm_tokenizer(
    #         prompt,
    #         return_tensors="pt",
    #         padding="longest",
    #         # truncation=True,
    #         # max_length=self.max_txt_len,
    #     ).to(image.device)

    #     empty_targets = torch.ones(atts_llm.size(), dtype=torch.long).to(image.device).fill_(-100)

    #     # self.llm_tokenizer.padding_side = "right"
    #     self.llm_tokenizer.truncation_side = 'right'
    #     n_cands = len(candidates)
    #     with self.maybe_autocast(dtype=torch.bfloat16):
    #         all_losses = []
    #         for n in range(n_segments):
    #             seg_len = n_cands // n_segments
    #             if n == (n_segments - 1):
    #                 seg_len = n_cands - seg_len * (n_segments - 1)

    #             start_i = n * (n_cands // n_segments)
    #             end_i = start_i + seg_len

    #             this_output_tokens = self.llm_tokenizer(
    #                 candidates[start_i:end_i],
    #                 return_tensors="pt",
    #                 padding="longest",
    #                 # truncation=True,
    #                 # max_length=self.max_output_txt_len,
    #             ).to(image.device)

    #             this_input_tokens_ids = text_input_tokens.input_ids.repeat_interleave(seg_len, dim=0)
    #             this_input_tokens_atts = text_input_tokens.attention_mask.repeat_interleave(seg_len, dim=0)

    #             this_output_tokens_ids = this_output_tokens.input_ids.repeat(bs, 1)
    #             this_output_tokens_atts = this_output_tokens.attention_mask.repeat(bs, 1)

    #             this_llm_tokens, this_input_targets_len = self.concat_text_input_output(
    #                 this_input_tokens_ids,
    #                 this_input_tokens_atts,
    #                 this_output_tokens_ids,
    #                 this_output_tokens_atts
    #             )

    #             this_llm_input_ids = this_llm_tokens['input_ids']
    #             this_llm_atts = this_llm_tokens['attention_mask']
    #             # this_llm_input_ids = torch.cat([this_input_tokens_ids, this_output_tokens_ids], dim=1)
    #             # this_llm_atts = torch.cat([this_input_tokens_atts, this_output_tokens_atts], dim=1)

    #             inputs_embeds = self.llm_model.get_input_embeddings()(this_llm_input_ids)
    #             inputs_embeds = torch.cat([inputs_llm.repeat_interleave(seg_len, dim=0), inputs_embeds], dim=1)
    #             attention_mask = torch.cat([atts_llm.repeat_interleave(seg_len, dim=0), this_llm_atts], dim=1)

    #             this_targets = this_llm_input_ids.masked_fill(this_llm_input_ids == self.llm_tokenizer.pad_token_id, -100)
    #             # this_targets[:, :this_input_tokens_ids.size(1)] = -100
    #             for i, l in enumerate(this_input_targets_len):
    #                 this_targets[i][:l] = -100

    #             this_targets = torch.cat([empty_targets.repeat_interleave(seg_len, dim=0), this_targets], dim=1)

    #             outputs = self.llm_model(
    #                 inputs_embeds=inputs_embeds,
    #                 attention_mask=attention_mask,
    #                 return_dict=True,
    #                 labels=this_targets,
    #                 reduction="none",
    #             )

    #             loss = outputs.loss

    #             loss = loss.reshape(bs, seg_len)
    #             # output_class_ranks = torch.argsort(loss, dim=-1)
    #             all_losses.append(loss)

    #         all_losses = torch.cat(all_losses, dim=-1)
    #         output_class_ranks = torch.argsort(all_losses, dim=-1)

    #     return output_class_ranks

    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        llm_model = cfg.get("llm_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 128)
        max_output_txt_len = cfg.get("max_output_txt_len", 256)

        apply_lemmatizer = cfg.get("apply_lemmatizer", False)

        qformer_text_input = cfg.get("qformer_text_input", True)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            llm_model=llm_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            max_output_txt_len=max_output_txt_len,
            apply_lemmatizer=apply_lemmatizer,
            qformer_text_input=qformer_text_input,
        )

        # if qformer_text_input:
        #     # Hard-coded to load from BLIP-2 stage-1 pre-trained model (not ideal)
        #     model.load_from_pretrained(
        #         url_or_filename="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained.pth"
        #     )

        model.load_checkpoint_from_config(cfg)

        return model
    @torch.no_grad()
    def extract_features(self, samples, mode="multimodal"):
        """
        Extract features for multimodal or unimodal samples.
        Args:
            samples (dict): A dictionary of samples, containing the following keys:
                - image (torch.Tensor): A tensor of shape (B, C, H, W) containing the image.
                    Raw images should be preprocessed before being passed to feature extractor.
                - text_input (list): A list of strings containing the text, length B.
            mode (str): The mode of feature extraction. Can be either "multimodal", "text" or "image".
                If "multimodal", return image features and multimodal features;
                if "text", return text features;
                if "image", return image features.
                Default: "multimodal".
        Returns:
            BlipOutputFeatures: A BlipOutputFeatures object containing the features.
                See lavis/models/blip_models/blip_outputs.py for more details.
        """
        image = samples.get("image")
        caption = samples.get("text_input")


        # assert mode is one of "image", "text", "multimodal"
        assert mode in [
            "image",
            "text",
            "multimodal",
        ], "mode must be one of 'image', 'text', 'multimodal'"

        # initalize output
        image_embeds, text_embeds, multimodal_embeds = None, None, None
        image_features, text_features = None, None

        if mode == "image":
            assert (
                image is not None
            ), "Image is not provided for mode 'image' or 'multimodal'"
            # return query features
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(
                image_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.device)
            query_tokens = self.query_tokens.expand(
                image_embeds_frozen.shape[0], -1, -1
            )

            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            image_embeds = image_embeds_frozen
            # image_features = F.normalize(self.vision_proj(image_embeds), dim=-1)
            # image_features = F.normalize(self.vision_proj(query_output.last_hidden_state), dim=-1)
            image_features = F.normalize(query_output.last_hidden_state, dim=-1)


        elif mode == "text":
            assert (
                caption is not None
            ), "text input is None for mode 'text' or 'multimodal'"

            # return text features
            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )

            text_output = self.Qformer.bert(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
            )
            text_embeds = text_output.last_hidden_state
            # text_features = self.text_proj(text_embeds)
            text_features = text_embeds

            text_features = F.normalize(text_features, dim=-1)

        elif mode == "multimodal":
            # return multimodel query features
            # with self.maybe_autocast():
            #     image_embeds_frozen = image
            image_embeds_frozen = image
            image_atts = torch.ones(
                image_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.device)
            query_tokens = self.query_tokens.expand(
                image_embeds_frozen.shape[0], -1, -1
            )
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                self.device
            )

            # text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
            #     self.device
            # )

            text = self.tokenizer(caption,
                        padding="max_length",
                        truncation=True,
                        max_length=35,
                        return_tensors="pt",
                    ).to(self.device)
            attention_mask = torch.cat([query_atts, text.attention_mask], dim=1)

            # print(text.input_ids.shape)
            # print(query_tokens.shape)
            # print(attention_mask.shape)
            # print(image_embeds_frozen.shape)
            # print(image_atts.shape)

            query_output = self.Qformer.bert(
                text.input_ids,
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeds_frozen,#768
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            query_embed = query_output.last_hidden_state[:,:query_tokens.size(1),:]
            inputs_llm = self.llm_proj(query_embed)
            atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(image_embeds_frozen.device)

            self.llm_tokenizer.padding_side = "right"
            self.llm_tokenizer.truncation_side = 'left'#左侧截断
            text_input_tokens = self.llm_tokenizer(
                samples['text_input'],
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
            ).to(image_embeds_frozen.device)

            llm_tokens = {"input_ids": [], "attention_mask": []}
            llm_tokens["input_ids"] = text_input_tokens.input_ids
            llm_tokens["attention_mask"] = text_input_tokens.attention_mask
            inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens['input_ids'])
            inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
            attention_mask = torch.cat([atts_llm, llm_tokens['attention_mask']], dim=1)

            with self.maybe_autocast():
                outputs = self.llm_model.model(
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    return_dict=True,
                )
            multimodal_embeds = outputs[0][:, :query_tokens.size(1), :]
            composed_features = F.normalize(
            self.text_proj(multimodal_embeds.to(torch.float32)), dim=-1
            )
            return BlipOutputFeatures(
            image_embeds=image_embeds,
            image_embeds_proj=image_features,
            text_embeds=text_embeds,
            text_embeds_proj=text_features,
            multimodal_embeds=multimodal_embeds,
            # multimodal_embeds_proj=composed_features,

            ),None



        return BlipOutputFeatures(
            image_embeds=image_embeds,
            image_embeds_proj=image_features,
            text_embeds=text_embeds,
            text_embeds_proj=text_features,
            multimodal_embeds=multimodal_embeds,
            # multimodal_embeds_proj=composed_features,

        )




    def compute_sim_matrix_tgir(self, sims_matrix,index_embeds,input_ids,task_cfg):

        k_test = task_cfg.k_test
        logging.info("Computing features for evaluation...")
        start_time = time.time()


        metric_logger = MetricLogger(delimiter="  ")
        header = "Evaluation:"
       
        score_matrix = torch.full(
           sims_matrix.shape, -100.0
        ).to(self.device)

        num_tasks = dist_utils.get_world_size()
        rank = dist_utils.get_rank()
        step = sims_matrix.size(0) // num_tasks + 1
        start = rank * step
        end = min(sims_matrix.size(0), start + step)

        for i, sims in enumerate(
            metric_logger.log_every(sims_matrix[start:end], 50, header)
        ):
            topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
            text_input =input_ids[start + i].repeat(k_test, 1).to(self.device)
            text_atts = torch.ones(text_input.shape, dtype=torch.long).to(self.device)
            image_inputs = index_embeds[topk_idx.cpu()].to(self.device)
            score = self.compute_itm(
                image_inputs= image_inputs,
                text_ids= text_input,
                text_atts=text_atts
            ).float()
            score_matrix[start + i, topk_idx] = score + topk_sim

        sims_matrix = sims_matrix.t()

        if dist_utils.is_dist_avail_and_initialized():
            dist.barrier()
            torch.distributed.all_reduce(
                score_matrix, op=torch.distributed.ReduceOp.SUM
            )


        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info("Evaluation time {}".format(total_time_str))

        return score_matrix.cpu()

    def extract_index_features_blip(self,dataloader) :
        """
        Extract FashionIQ or CIRR index features
        :param dataset: FashionIQ or CIRR dataset in 'classic' mode
        :param model: CLIP model
        :return: a tensor of features and a list of images
        """
        # feature_dim = model.visual.output_dim
        # index_embeds = torch.empty((0,730, )).to(self.device, non_blocking=True)
        index_features_for_sesarch= torch.empty((0,32, 256)).to(self.device, non_blocking=True)

        index_names = []
        index_embeds = torch.empty((0,730, 1408)).to(self.device, non_blocking=True).cpu()
        name_to_feat = dict()
        for samples in tqdm(dataloader):
            names = samples["image_name"]
            images = samples["image"]
            images = images.to(self.device, non_blocking=True)
            with torch.no_grad():
                output = self.extract_features({"image":images}, mode="image")
                batch_features = output.image_embeds.cpu()
                index_embeds = torch.vstack((index_embeds, batch_features))
                index_features_for_sesarch = torch.vstack((index_features_for_sesarch, output.image_embeds_proj))
                index_names.extend(names)
                batch_name_to_feat = dict(zip(names, batch_features))
                name_to_feat.update(batch_name_to_feat)
        # name_to_feat = dict(zip(index_names, index_embeds))
        return index_embeds,name_to_feat, index_features_for_sesarch,index_names
    def generate_cirr_val_predictions_blip(self, relative_val_loader, name_to_feat) :
        """
        Compute CIRR predictions on the validation set
        :param self: CLIP model
        :param relative_val_dataset: CIRR validation dataset in relative mode
        :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                                features
        :param index_features: validation index features
        :param index_names: validation index names
        :return: predicted features, reference names, target names and group members
        """

        # Get a mapping from index names to index features
        # name_to_feat = dict(zip(index_names, index_features))
        predicted_features = torch.empty((0, 32,self.embed_dim)).to(self.device, non_blocking=True)
        # predicted_embeds = torch.empty((0, 32,768)).to(self.device, non_blocking=True)

        target_names = []
        group_members = []
        reference_names = []
        input_ids = []

        for samples in tqdm(relative_val_loader):  # Load data
            batch_reference_names = samples["reference_image_name"]
            batch_target_names = samples["target_hard_name"]
            captions = samples["text_input"]
            batch_group_members = samples["group_members"]

            # Concatenate the captions in a deterministic way
            # text = ["Forward:"+i+" and remain other part similar" for i in captions]
            text = [i+" and remain other part similar, what will the image be?" for i in captions]
            batch_group_members = np.array(batch_group_members).T.tolist()

            # Compute the predicted features
            with torch.no_grad():
                reference_image_Embeds = torch.stack(itemgetter(*batch_reference_names)(name_to_feat)).to(self.device, non_blocking=True) # To avoid unnecessary computation retrieve the reference image features directly from the index features
                multimodal_embeds,batch_input_ids = self.extract_features({"image":reference_image_Embeds,"text_input":text}, mode="multimodal")
                multimodal_embeds = multimodal_embeds.multimodal_embeds
                multimodal_features =  F.normalize(self.text_proj(multimodal_embeds.to(torch.float32)), dim=-1)
                # input_ids.append(batch_input_ids)
            predicted_features = torch.vstack((predicted_features, multimodal_features))
            # predicted_embeds = torch.vstack((predicted_embeds, multimodal_embeds))

            target_names.extend(batch_target_names)
            group_members.extend(batch_group_members)
            reference_names.extend(batch_reference_names)
        # input_ids = torch.cat(input_ids, dim=0)
        return predicted_features,reference_names, target_names,group_members,input_ids

    def generate_fiq_val_predictions_blip(self, relative_val_loader, name_to_feat) :
        """
        Compute FashionIQ predictions on the validation set
        :param model: CLIP model
        :param relative_val_dataset: FashionIQ validation dataset in relative mode
        :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                                features
        :param index_features: validation index features
        :param index_names: validation index names
        :return: predicted features and target names
        """

        # Get a mapping from index names to index features
        # name_to_feat = dict(zip(index_names, index_features))
        predicted_features = torch.empty((0, 32,self.embed_dim)).to(self.device, non_blocking=True)
        # predicted_embeds = torch.empty((0, 32,768)).to(self.device, non_blocking=True)

        target_names = []
        reference_names = []
        input_ids = []

        for samples in tqdm(relative_val_loader):  # Load data
            batch_reference_names = samples["reference_image_name"]
            batch_target_names = samples["target_hard_name"]
            captions = samples["text_input"]
            # Concatenate the captions in a deterministic way
            text = [i+" and remain other part similar, what will the image be?" for i in captions]

            # Compute the predicted features
            with torch.no_grad():
                reference_image_Embeds = torch.stack(itemgetter(*batch_reference_names)(
                        name_to_feat)).to(self.device, non_blocking=True)  # To avoid unnecessary computation retrieve the reference image features directly from the index features
                multimodal_embeds,batch_input_ids = self.extract_features({"image":reference_image_Embeds,"text_input":text}, mode="multimodal")
                multimodal_embeds = multimodal_embeds.multimodal_embeds
                multimodal_features =  F.normalize(self.text_proj(multimodal_embeds.to(torch.float32)), dim=-1)
                # input_ids.append(batch_input_ids)
            predicted_features = torch.vstack((predicted_features, multimodal_features))
            # predicted_embeds = torch.vstack((predicted_embeds, multimodal_embeds))

            target_names.extend(batch_target_names)
         
            reference_names.extend(batch_reference_names)
        # input_ids = torch.cat(input_ids, dim=0)
        return predicted_features,reference_names, target_names,input_ids
    

    @torch.no_grad()
    def generate_target(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=256,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1,
        num_captions=1,
        temperature=1,
    ):
        self.llm_tokenizer.padding_side = "left"
        if "prompt" in samples.keys():
            prompt = samples["text_input"]
        else:
            prompt = self.prompt

        image_embeds = samples["image"]

        bs = image_embeds.size(0)

        if isinstance(prompt, str):
            prompt = [prompt] * bs
        else:
            assert len(prompt) == bs, "The number of prompts must be equal to the batch size."

        # For TextCaps
        if "ocr_tokens" in samples.keys() and "{}" in prompt[0]:
            prompt = [p.format(', '.join(samples['ocr_tokens'][i][:30])) for i, p in enumerate(prompt)]

        query_tokens = self.query_tokens.expand(bs, -1, -1)
        if self.qformer_text_input:
            # remove ocr tokens in q_former (for eval textvqa)
            # qformer_prompt = prompt
            # qformer_prompt = ['Question: ' + qp.split(' Question: ')[1] for qp in qformer_prompt]

            text_Qformer = self.tokenizer(
                prompt,
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image_embeds.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image_embeds.device)
            Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

       
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)

        if self.qformer_text_input:
            query_output = self.Qformer.bert(
                text_Qformer.input_ids,
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
        else:
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

        inputs_llm = self.llm_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])
        atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(image_embeds.device)

        llm_tokens = self.llm_tokenizer(
            prompt,
            padding="longest",
            return_tensors="pt"
        ).to(image_embeds.device)

        with self.maybe_autocast():
            inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
            attention_mask = torch.cat([atts_llm, llm_tokens.attention_mask], dim=1)

            outputs = self.llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                # eos_token_id=self.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )

        outputs[outputs == 0] = 2 # convert output id 0 to 2 (eos_token_id)
        output_text = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output_text = [text.strip() for text in output_text]

        return output_text
        # return inputs_embeds,attention_mask,'output_text'

    


    def compute_sim_matrix_tgir(self, dataloader_classic,data_loader,task_cfg):
        index_embeds,name_to_feat, index_features_for_sesarch,index_names = self.extract_index_features_blip(dataloader_classic)
        
        predicted_features,reference_names, target_names,group_members,input_ids,query_embeds = self.generate_cirr_val_predictions_blip(
            data_loader,name_to_feat)
        sims_matrix = []
        # score_matrix = torch.full((len(data_loader.dataset.image), len(dataloader_classic)), -100.0).to(self.device)
        for predicted_feature in predicted_features:

            sim_q2i =F.normalize(index_features_for_sesarch, dim=-1) @ predicted_feature[0,:].t()
            sim_q2t, _ = sim_q2i.max(1)
            sims_matrix.append(sim_q2t.t())
        sims_matrix = torch.stack(sims_matrix, dim=0)


        k_test = task_cfg.k_test
        rerankitm = task_cfg.rerankitm
        # rerankiic = task_cfg.rerankiic

        
        logging.info("Computing features for evaluation...")
        start_time = time.time()


        metric_logger = MetricLogger(delimiter="  ")
        header = "Evaluation:"
       
        score_matrix = torch.full(
           sims_matrix.shape, -100.0
        ).to(self.device)

        num_tasks = dist_utils.get_world_size()
        rank = dist_utils.get_rank()
        step = sims_matrix.size(0) // num_tasks + 1
        start = rank * step
        end = min(sims_matrix.size(0), start + step)

        for i, sims in enumerate(
            metric_logger.log_every(sims_matrix[start:end], 50, header)
        ):
            topk_sim, topk_idx = sims.topk(k=k_test, dim=0)

            if rerankitm:
                text_input =input_ids[start + i].repeat(k_test, 1).to(self.device)
                # print(query_embeds[start + i].shape)
                query_embed =query_embeds[start + i].repeat(k_test, 1,1).to(self.device)
                image_inputs = index_embeds[topk_idx.cpu()].to(self.device)
                score = self.compute_itm(
                    image_inputs= image_inputs,
                    text_inputs= text_input,
                    query_embed = query_embed,
                ).float()
                score_matrix[start + i, topk_idx] = score + topk_sim
            # else if rerankiic:

            else:
                score_matrix[start + i, topk_idx] = topk_sim


        sims_matrix = sims_matrix.t()

        if dist_utils.is_dist_avail_and_initialized():
            dist.barrier()
            torch.distributed.all_reduce(
                score_matrix, op=torch.distributed.ReduceOp.SUM
            )


        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info("Evaluation time {}".format(total_time_str))

        return 1-score_matrix.cpu(),index_names,target_names,reference_names,group_members, name_to_feat
    
    def extract_index_features_blip(self,dataloader) :
        """
        Extract FashionIQ or CIRR index features
        :param dataset: FashionIQ or CIRR dataset in 'classic' mode
        :param model: CLIP model
        :return: a tensor of features and a list of images
        """
        # feature_dim = model.visual.output_dim
        # index_embeds = torch.empty((0,730, 1408)).to(self.device, non_blocking=True)
        index_features_for_sesarch= torch.empty((0,32, self.embed_dim)).to(self.device, non_blocking=True)

        index_names = []
        name_to_feat = dict()
        index_embeds = torch.empty((0,730, 1408)).to(self.device, non_blocking=True).cpu()
        for samples in tqdm(dataloader):
            names = samples["image_name"]
            images = samples["image"]
            images = images.to(self.device, non_blocking=True)
            with torch.no_grad():
                output = self.extract_features({"image":images}, mode="image")
                batch_features = output.image_embeds.cpu()
                index_embeds = torch.vstack((index_embeds, batch_features))
                index_features_for_sesarch = torch.vstack((index_features_for_sesarch, output.image_embeds_proj))
                index_names.extend(names)
                batch_name_to_feat = dict(zip(names, batch_features))
                name_to_feat.update(batch_name_to_feat)
        # name_to_feat = dict(zip(index_names, index_embeds))
        return index_embeds,name_to_feat, index_features_for_sesarch,index_names
    
    def generate_cirr_val_predictions_blip(self, relative_val_loader, name_to_feat) :
        """
        Compute CIRR predictions on the validation set
        :param self: CLIP model
        :param relative_val_dataset: CIRR validation dataset in relative mode
        :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                                features
        :param index_features: validation index features
        :param index_names: validation index names
        :return: predicted features, reference names, target names and group members
        """

        # Get a mapping from index names to index features
        # name_to_feat = dict(zip(index_names, index_features))
        predicted_features = torch.empty((0, 32,self.embed_dim)).to(self.device, non_blocking=True)
        
        # predicted_embeds = torch.empty((0, 32,768)).to(self.device, non_blocking=True)

        target_names = []
        group_members = []
        reference_names = []


        input_ids = []
        query_embeds = torch.empty((0, 32,768)).cpu()


        for samples in tqdm(relative_val_loader):  # Load data
            batch_reference_names = samples["reference_image_name"]
            batch_target_names = samples["target_hard_name"]
            captions = samples["text_input"]
            batch_group_members = samples["group_members"]

            # Concatenate the captions in a deterministic way
            text = [i for i in captions]
            # text = [i+" and remain other part similar, what will the image be?" for i in captions]
            batch_group_members = np.array(batch_group_members).T.tolist()

            # Compute the predicted features
            with torch.no_grad():
                reference_image_Embeds = torch.stack(itemgetter(*batch_reference_names)(name_to_feat)).to(self.device, non_blocking=True) # To avoid unnecessary computation retrieve the reference image features directly from the index features
                query_embed,batch_input_ids,multimodal_embeds = self.extract_features({"image":reference_image_Embeds,"text_input":text}, mode="multimodal")
                query_embed = query_embed.multimodal_embeds
                query_embeds = torch.vstack((query_embeds,query_embed.cpu()))
                multimodal_features =  F.normalize(self.text_proj(multimodal_embeds[:,:32,:]), dim=-1)
                input_ids.append(batch_input_ids)
            predicted_features = torch.vstack((predicted_features, multimodal_features))
            # predicted_embeds = torch.vstack((predicted_embeds, multimodal_embeds))
            target_names.extend(batch_target_names)
            group_members.extend(batch_group_members)
            reference_names.extend(batch_reference_names)
        input_ids = torch.cat(input_ids, dim=0)
        return predicted_features,reference_names, target_names,group_members,input_ids,query_embeds

    def generate_fiq_val_predictions_blip(self, relative_val_loader, name_to_feat) :
        """
        Compute FashionIQ predictions on the validation set
        :param model: CLIP model
        :param relative_val_dataset: FashionIQ validation dataset in relative mode
        :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                                features
        :param index_features: validation index features
        :param index_names: validation index names
        :return: predicted features and target names
        """

        # Get a mapping from index names to index features
        # name_to_feat = dict(zip(index_names, index_features))
        predicted_features = torch.empty((0, 32,self.embed_dim)).to(self.device, non_blocking=True)
        # predicted_embeds = torch.empty((0, 32,768)).to(self.device, non_blocking=True)

        target_names = []
        reference_names = []
        input_ids = []
        query_embeds = torch.empty((0, 32,768)).cpu()
        # query_embeds = []

        for samples in tqdm(relative_val_loader):  # Load data
            batch_reference_names = samples["reference_image_name"]
            batch_target_names = samples["target_hard_name"]
            captions = samples["text_input"]
            # Concatenate the captions in a deterministic way
            text = [i for i in captions]

            # Compute the predicted features
            with torch.no_grad():
                reference_image_Embeds = torch.stack(itemgetter(*batch_reference_names)(
                        name_to_feat)).to(self.device, non_blocking=True)  # To avoid unnecessary computation retrieve the reference image features directly from the index features
                query_embed,batch_input_ids,multimodal_embeds = self.extract_features({"image":reference_image_Embeds,"text_input":text}, mode="multimodal")
                query_embed = query_embed.multimodal_embeds
                query_embeds = torch.vstack((query_embeds,query_embed.cpu()))
                multimodal_features =  F.normalize(self.text_proj(multimodal_embeds[:,:32,:]), dim=-1)
                input_ids.append(batch_input_ids)
            predicted_features = torch.vstack((predicted_features, multimodal_features))
            # predicted_embeds = torch.vstack((predicted_embeds, multimodal_embeds))

            target_names.extend(batch_target_names)
         
            reference_names.extend(batch_reference_names)
        
        input_ids = torch.cat(input_ids, dim=0)
        return predicted_features,reference_names, target_names,input_ids,query_embeds
    
    
    def generate_cirr_test_predictions_blip(self, relative_val_loader, name_to_feat) :
        """
        Compute CIRR predictions on the validation set
        :param self: CLIP model
        :param relative_val_dataset: CIRR validation dataset in relative mode
        :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                                features
        :param index_features: validation index features
        :param index_names: validation index names
        :return: predicted features, reference names, target names and group members
        """

        # Get a mapping from index names to index features
        # name_to_feat = dict(zip(index_names, index_features))
        predicted_features = torch.empty((0, 32,self.embed_dim)).to(self.device, non_blocking=True)
        
        # predicted_embeds = torch.empty((0, 32,768)).to(self.device, non_blocking=True)
        pairs_id = []
        group_members = []
        reference_names = []
        input_ids = []
        query_embeds = torch.empty((0, 32,768)).cpu()


        for samples in tqdm(relative_val_loader):  # Load data
            batch_reference_names = samples["reference_image_name"]
            batch_pairs_id =  samples["pair_id"]
            captions = samples["text_input"]
            batch_group_members = samples["group_members"]
            # Concatenate the captions in a deterministic way
            text = [i for i in captions]
            # text = [i+" and remain other part similar, what will the image be?" for i in captions]
            batch_group_members = np.array(batch_group_members).T.tolist()

            # Compute the predicted features
            with torch.no_grad():
                reference_image_Embeds = torch.stack(itemgetter(*batch_reference_names)(name_to_feat)).to(self.device, non_blocking=True) # To avoid unnecessary computation retrieve the reference image features directly from the index features
                query_embed,batch_input_ids,multimodal_embeds = self.extract_features({"image":reference_image_Embeds,"text_input":text}, mode="multimodal")
                query_embed = query_embed.multimodal_embeds
                query_embeds = torch.vstack((query_embeds,query_embed.cpu()))
                multimodal_features =  F.normalize(self.text_proj(multimodal_embeds[:,:32,:]), dim=-1)
                input_ids.append(batch_input_ids)
            predicted_features = torch.vstack((predicted_features, multimodal_features))
            # predicted_embeds = torch.vstack((predicted_embeds, multimodal_embeds))

            group_members.extend(batch_group_members)
            pairs_id.extend(batch_pairs_id)
            reference_names.extend(batch_reference_names)
        input_ids = torch.cat(input_ids, dim=0)
        return predicted_features,reference_names,group_members,input_ids,query_embeds,pairs_id
    def evaluation(self,dataloader_classic,data_loader,task_cfg):
        score_matrix,index_names,target_names,reference_names,group_members,name_to_feat = self.compute_sim_matrix_tgir(dataloader_classic,data_loader, task_cfg)
        return score_matrix,index_names,target_names,reference_names,group_members,name_to_feat
    def evaluation_fiq(self,dataloader_classic,data_loader,task_cfg):
        score_matrix,index_names,target_names,reference_names = self.compute_sim_matrix_tgir_fiq(dataloader_classic,data_loader, task_cfg)
        return score_matrix,index_names,target_names,reference_names
    def compute_sim_matrix_tgir_test(self, dataloader_classic,data_loader,task_cfg):
        index_embeds,name_to_feat, index_features_for_sesarch,index_names = self.extract_index_features_blip(dataloader_classic)
        
        predicted_features,reference_names,group_members,input_ids,query_embeds,pairs_id = self.generate_cirr_test_predictions_blip(
            data_loader,name_to_feat)
        
        sims_matrix = []
        # score_matrix = torch.full((len(data_loader.dataset.image), len(dataloader_classic)), -100.0).to(self.device)
        for predicted_feature in predicted_features:

            sim_q2i =F.normalize(index_features_for_sesarch, dim=-1) @ predicted_feature[0,:].t()
            sim_q2t, _ = sim_q2i.max(1)
            sims_matrix.append(sim_q2t.t())
        sims_matrix = torch.stack(sims_matrix, dim=0)


        k_test = task_cfg.k_test
        rerankitm = task_cfg.rerankitm
        # rerankiic = task_cfg.rerankiic

        
        logging.info("Computing features for evaluation...")
        start_time = time.time()


        metric_logger = MetricLogger(delimiter="  ")
        header = "Evaluation:"
       
        score_matrix = torch.full(
           sims_matrix.shape, -100.0
        ).to(self.device)

        num_tasks = dist_utils.get_world_size()
        rank = dist_utils.get_rank()
        step = sims_matrix.size(0) // num_tasks + 1
        start = rank * step
        end = min(sims_matrix.size(0), start + step)

        for i, sims in enumerate(
            metric_logger.log_every(sims_matrix[start:end], 50, header)
        ):
            topk_sim, topk_idx = sims.topk(k=k_test, dim=0)

            if rerankitm:
                text_input =input_ids[start + i].repeat(k_test, 1).to(self.device)
                # print(query_embeds[start + i].shape)
                query_embed =query_embeds[start + i].repeat(k_test, 1,1).to(self.device)
                image_inputs = index_embeds[topk_idx.cpu()].to(self.device)
                score = self.compute_itm(
                    image_inputs= image_inputs,
                    text_inputs= text_input,
                    query_embed = query_embed,
                ).float()
                score_matrix[start + i, topk_idx] = score + topk_sim
            # else if rerankiic:

            else:
                score_matrix[start + i, topk_idx] = topk_sim


        sims_matrix = sims_matrix.t()

        if dist_utils.is_dist_avail_and_initialized():
            dist.barrier()
            torch.distributed.all_reduce(
                score_matrix, op=torch.distributed.ReduceOp.SUM
            )


        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info("Evaluation time {}".format(total_time_str))

        return 1-score_matrix.cpu(),index_names,reference_names,group_members, name_to_feat,pairs_id
