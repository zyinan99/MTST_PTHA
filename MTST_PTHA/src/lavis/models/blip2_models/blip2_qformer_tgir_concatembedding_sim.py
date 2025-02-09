"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
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

from lavis.common.registry import registry
from lavis.models.base_model import all_gather_with_grad, concat_all_gather
from lavis.models.blip2_models.blip2 import (
    Blip2Base,
    compute_sim_matrix,
    disabled_train,
)
import numpy as np 
from typing import List, Tuple
from lavis.models.blip_models.blip_outputs import BlipOutput, BlipOutputFeatures
from tqdm import tqdm
import lavis.common.dist_utils as dist_utils
from lavis.common.logger import MetricLogger
import pickle
from einops import rearrange


@registry.register_model("blip2_tgir_ce_sim")

class Blip2Qformer4TGIRCESIM(Blip2Base):
    """
    BLIP2 first-stage model with Q-former and ViT.
    Supported model types:
        - pretrained: pretrained model with vit-g
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_tgir_ce", "pretrain")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/blip2/blip2_pretrain.yaml",
        "pretrain_mtst":"configs/models/blip2/blip2_pretrain_mtst.yaml"

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
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=40,
        llm_model=''
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        print(vit_model)
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                # print(name)
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")

        self.num_query_token = num_query_token
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features, cross_attention_freq
        )
        self.Qformer.resize_token_embeddings(len(self.tokenizer))

        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")

        for name, param in self.Qformer.cls.named_parameters():   
            param.requires_grad = False
        


        self.embed_dim = embed_dim
        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)


        self.temp = nn.Parameter(0.07 * torch.ones([]))

        self.max_txt_len = max_txt_len

        self.prompt_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, self.Qformer.config.hidden_size)
        )
        self.prompt_tokens.data.normal_(mean=0.0, std=self.Qformer.config.hidden_size)

    def forward(self, samples):
        
        target_image = samples["target"]
        reference_image = samples["image"]


        text = [i for i in samples["text_input"]]
        text_rev = [i for i in samples["text_rev"]]

        reference_image_embeds = self.ln_vision(self.visual_encoder(reference_image))
        reference_image_atts = torch.ones(reference_image_embeds.size()[:-1], dtype=torch.long).to(
            reference_image.device
        )

        query_tokens = self.query_tokens.expand(reference_image_embeds.shape[0], -1, -1)

        query_tokens_reverse = self.query_tokens.expand(reference_image_embeds.shape[0], -1, -1)

        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(reference_image.device)
        query_atts_reverse = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(reference_image.device)

        target_image_embeds = self.ln_vision(self.visual_encoder(target_image))
        target_image_atts = torch.ones(target_image_embeds.size()[:-1], dtype=torch.long).to(
            reference_image.device
        )
        target_query_tokens = self.query_tokens.expand(target_image_embeds.shape[0], -1, -1)
        target_query_output = self.Qformer.bert(
            query_embeds=target_query_tokens,
            encoder_hidden_states=target_image_embeds,
            encoder_attention_mask=target_image_atts,
            use_cache=True,
            return_dict=True,
        )
        sequence_output_target = target_query_output.last_hidden_state
        target_image_feats = F.normalize(
            self.vision_proj(target_query_output.last_hidden_state), dim=-1
        )

        #text encode
        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(reference_image.device)

        text_tokens_reverse = self.tokenizer(
            text_rev,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(reference_image.device)
    

        attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
        attention_mask_reverse = torch.cat([query_atts_reverse, text_tokens_reverse.attention_mask], dim=1)

        composed_query_output = self.Qformer.bert(
                text_tokens.input_ids,
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=reference_image_embeds,#768
                encoder_attention_mask=reference_image_atts,
                return_dict=True,
            )
    
        sequence_output = composed_query_output[0][:, :query_tokens.shape[1], :]#(b,32,768)
        

        attention_mask2 = torch.ones(attention_mask.size(), dtype=torch.long).to(reference_image.device)
        composed_output = self.Qformer.bert(
            text_tokens.input_ids,
            query_embeds=sequence_output,
            attention_mask=attention_mask2,
            return_dict=True,
        )     


        multimodal_embeds =  composed_output[0][:,:33,:].mean(dim=1)


        composed_feat = F.normalize(
            self.text_proj(multimodal_embeds), dim=-1
        )



        composed_query_output_reverse = self.Qformer.bert(
                text_tokens_reverse.input_ids,
                query_embeds=query_tokens_reverse,
                attention_mask=attention_mask_reverse,
                encoder_hidden_states=target_image_embeds,#768
                encoder_attention_mask=target_image_atts,
                return_dict=True,
            )
    
        sequence_output_reverse = composed_query_output_reverse[0][:, :query_tokens.shape[1], :]#(b,32,768)
        
    
        ##============== Image-text Contrastive ===================###

        sim_t2q = torch.matmul(
            composed_feat.unsqueeze(1).unsqueeze(1), target_image_feats.permute(0, 2, 1)
        ).squeeze()
        sim_t2i, t2iindex = sim_t2q.max(-1)

        sim_t2i = sim_t2i / self.temp  # [batch_size, batch_size*num_gpu]
        
        bs = target_image.size(0)
        targets = torch.linspace(0,  bs - 1, bs, dtype=int).to(
            target_image.device
        )

                        
        loss_itc = F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
        ###============== Loss t2t  ===================###
        
        prompt_tokens = self.prompt_tokens.expand(bs, -1, -1)

        text_only_output = self.Qformer.bert(
            text_tokens.input_ids,
            query_embeds=prompt_tokens,
            attention_mask=attention_mask,
            return_dict=True,
            no_img=True
        )

        text_only_feat = F.normalize(
            self.text_proj(text_only_output.last_hidden_state[:, 0, :]), dim=-1
        )

        sim_r2t = torch.matmul(
            text_only_feat.unsqueeze(1).unsqueeze(1), target_image_feats.permute(0, 2, 1)
        ).squeeze()

        sim_r2t, _ = sim_r2t.max(-1)
        sim_r2t = sim_r2t / self.temp
        loss_t2t = F.cross_entropy(sim_r2t, targets)

        ###============== Loss p2p ===================###
        
        p1 = sequence_output[:,:32,:].mean(1)
        p2 = sequence_output_reverse[:,:32,:].mean(1)
        loss_p2p =  mseloss(p1,p2.detach())
       

        return {
            'loss_itc': loss_itc, 
            'loss_rtc': loss_t2t,
            'loss_align': loss_p2p,
            }
       


    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=3,
        max_length=30,
        min_length=10,
        top_p=0.9,
        repetition_penalty=1.0,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        image = samples["image"]
        image_embeds = self.ln_vision(self.visual_encoder(image))

        if not use_nucleus_sampling:
            image_embeds = image_embeds.repeat_interleave(num_beams, dim=0)
        else:
            num_beams = 1
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        model_kwargs = {
            "encoder_hidden_states": image_embeds,
            "encoder_attention_mask": image_atts,
        }

        input_ids = (
            torch.LongTensor(image.size(0), 1)
            .fill_(self.tokenizer.bos_token_id)
            .to(image.device)
        )
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        outputs = self.Qformer.generate(
            input_ids=input_ids,
            query_embeds=query_tokens,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **model_kwargs
        )
        captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return captions

    def forward_image(self, image):
        image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        return query_output.last_hidden_state, image_embeds

    def forward_text(self, text_tokens):
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        return text_output.last_hidden_state[:, 0, :]

    def compute_itm(self, image_inputs, text_ids, text_atts):
        image_atts = torch.ones(image_inputs.size()[:-1], dtype=torch.long).to(
            image_inputs.device
        )
        query_tokens = self.query_tokens.expand(image_inputs.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            image_inputs.device
        )
        attention_mask = torch.cat([query_atts, text_atts], dim=1)
        output_itm = self.Qformer.bert(
            text_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=image_inputs,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens.size(1), :]
        itm_logit = self.itm_head(vl_embeddings)
        itm_logit = itm_logit[:, :, 1].mean(dim=1)
        return itm_logit
    

    @torch.no_grad()
    def inference(self, reference_embeds, target_embeds, text,stage2 = False):
        reference_embeds = reference_embeds.cuda()
        target_embeds = target_embeds.cuda()
        

        image_atts = torch.ones(reference_embeds.size()[:-1], dtype=torch.long).to(
            reference_embeds.device
        )
        # query tokens
        query_tokens = self.query_tokens.expand(reference_embeds.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            self.device
        )
        # text tokens
        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(reference_embeds.device)

        attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
        fusion_output = self.Qformer.bert(
            text_tokens.input_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=reference_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )


        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            query_embeds=fusion_output.last_hidden_state[:, : query_tokens.size(1), :],
            attention_mask=attention_mask,
            return_dict=True,
        )

        fusion_feats = F.normalize(
            self.text_proj(text_output.last_hidden_state[:, 32, :]), dim=-1
        )
        if stage2:
            return text_output
        target_feats = F.normalize(self.vision_proj(target_embeds), dim=-1)
       
        extend = False
        if fusion_feats.shape[0]==1:
            extend =True
        sim_t2q = torch.matmul(
            fusion_feats.unsqueeze(1).unsqueeze(1), target_feats.permute(0, 2, 1)
        ).squeeze()
        if  extend:
            sim_t2q = sim_t2q.unsqueeze(0)


        sim_i2t, _ = sim_t2q.max(-1)

        return sim_i2t




    @torch.no_grad()
    def inference_reference(self, reference_embeds, text,stage2 = False):
        reference_embeds = reference_embeds.cuda()

        image_atts = torch.ones(reference_embeds.size()[:-1], dtype=torch.long).to(
            reference_embeds.device
        )
        # query tokens
        query_tokens = self.query_tokens.expand(reference_embeds.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            self.device
        )
        # text tokens
        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(reference_embeds.device)

        attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
        fusion_output = self.Qformer.bert(
            text_tokens.input_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=reference_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            query_embeds=fusion_output.last_hidden_state[:, : query_tokens.size(1), :],
            attention_mask=attention_mask,
            return_dict=True,
        )

        fusion_feats = F.normalize(
            self.text_proj(text_output.last_hidden_state[:, 32, :]), dim=-1
    
        )
        return fusion_feats
    @torch.no_grad()
    def inference_target(self, target_embeds):
        target_embeds = target_embeds.cuda()
        target_feats = F.normalize(self.vision_proj(target_embeds), dim=-1)
        return target_feats
        



    @torch.no_grad()
    def extract_target_features(self, image, mode='mean', target_only=False, ref_only=False):
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
        image_embeds = query_output.last_hidden_state
        image_features = image_embeds
        # return image_embeds
        # image_features = F.normalize(self.vision_proj(image_embeds), dim=-1)
        if target_only:
            return image_features.detach().cpu()
        if ref_only:
            return image_embeds_frozen.detach()
        return image_features.detach().cpu(), image_embeds_frozen.detach().cpu()


    @torch.no_grad()
    def inference_gis(self, image, target, text):
        bs, n, _, h, w = target.shape
        target = target.view(-1, 3, h, w)
        with self.maybe_autocast():
            image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
        image_embeds_frozen = image_embeds_frozen.float()
        image_atts = torch.ones(
            image_embeds_frozen.size()[:-1], dtype=torch.long
        ).to(self.device)
        query_tokens = self.query_tokens.expand(
            image_embeds_frozen.shape[0], -1, -1
        )
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            self.device
        )
        # text tokens
        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image.device)

        attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
        fusion_output = self.Qformer.bert(
            text_tokens.input_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds_frozen,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            query_embeds=fusion_output.last_hidden_state[:, : query_tokens.size(1), :],
            attention_mask=attention_mask,
            return_dict=True,
        )

        fusion_feats = F.normalize(
            self.text_proj(text_output.last_hidden_state[:, 32, :]), dim=-1
        )

        with self.maybe_autocast():
            taregt_embeds = self.ln_vision(self.visual_encoder(target))
        taregt_embeds = taregt_embeds.float()
        target_atts = torch.ones(taregt_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )
        query_tokens = self.query_tokens.expand(
            taregt_embeds.shape[0], -1, -1
        )
        target_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=taregt_embeds,
            encoder_attention_mask=target_atts,
            use_cache=True,
            return_dict=True,
        )
        target_feats = F.normalize(
            self.vision_proj(target_output.last_hidden_state), dim=-1
        )
        target_feats = target_feats.view(bs, n, 32, -1).view(bs, n * 32, -1)
        sim_t2q = torch.matmul(fusion_feats.unsqueeze(1), target_feats.permute(0, 2, 1)).squeeze(1)
        sim_t2q = sim_t2q.view(bs, n, 32)
        sim_i2t, _ = sim_t2q.max(-1)
        return sim_i2t


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
            image_embeds = query_output.last_hidden_state
            image_features = F.normalize(self.vision_proj(image_embeds), dim=-1)

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
            text_features = self.text_proj(text_embeds)
            text_features = F.normalize(text_features, dim=-1)

        elif mode == "multimodal":
            # return multimodel query features
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(
                image_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.device)
            query_tokens = self.query_tokens.expand(
                image_embeds_frozen.shape[0], -1, -1
            )
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                self.device
            )

            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )
            attention_mask = torch.cat([query_atts, text.attention_mask], dim=1)

            output = self.Qformer.bert(
                text.input_ids,
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            multimodal_embeds = output.last_hidden_state[:, : query_tokens.size(1), :]

        return BlipOutputFeatures(
            image_embeds=image_embeds,
            image_embeds_proj=image_features,
            text_embeds=text_embeds,
            text_embeds_proj=text_features,
            multimodal_embeds=multimodal_embeds,
        )

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        cross_attention_freq = cfg.get("cross_attention_freq", 2)

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        max_txt_len = cfg.get("max_txt_len", 32)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            cross_attention_freq=cross_attention_freq,
            max_txt_len=max_txt_len,
        )
        model.load_checkpoint_from_config(cfg)

        return model

    def compute_sim_matrix(self, data_loader, task_cfg):
        """
        Compute similarity i2t, t2i matrix for the given data loader.
        """
        k_test = task_cfg.k_test

        return compute_sim_matrix(model=self, data_loader=data_loader, k_test=k_test)

def compute_gradcam(model, visual_input,visual_input2, text_input, tokenized_text, text_input2, tokenized_text2, block_num=10):
    model.Qformer.bert.encoder.layer[
        block_num
    ].crossattention.self.save_attention = True
    output = model({"image": visual_input,"target":visual_input2, "text_input": text_input,"text_rev": text_input2})
    loss = output
    model.zero_grad()
    loss.backward()
    with torch.no_grad():
        mask = tokenized_text.attention_mask.view(
            tokenized_text.attention_mask.size(0), 1, -1, 1, 1
        )  # (bsz,1,token_len, 1,1)
        print('mask.shape',mask.shape)
        token_length = tokenized_text.attention_mask.sum(dim=-1) - 2
        token_length = token_length.cpu()
        # grads and cams [bsz, num_head, seq_len, image_patch]
        grads = model.Qformer.bert.encoder.layer[
        block_num].crossattention.self.get_attn_gradients()
        cams = model.Qformer.bert.encoder.layer[
        block_num].crossattention.self.get_attention_map()
        print(cams.shape)
        # assume using vit with 576 num image patch

        cams = cams[:, :, :, 1:].reshape(visual_input.size(0), 12, -1, 16, 16) * mask
        grads = (
            grads[:, :, :, 1:].clamp(0).reshape(visual_input.size(0), 12, -1, 16, 16)
            * mask)
        gradcams = cams * grads
        gradcam_list = []

        for ind in range(visual_input.size(0)):
            token_length_ = token_length[ind]
            gradcam = gradcams[ind].mean(0).cpu().detach()
            # [enc token gradcam, average gradcam across token, gradcam for individual token]
            gradcam = torch.cat(
                (
                    gradcam[0:1, :],
                    gradcam[1 : token_length_ + 1, :].sum(dim=0, keepdim=True)
                    / token_length_,
                    gradcam[1:, :],
                )
            )
            gradcam_list.append(gradcam)
            
    return gradcam_list, output
class FFN(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.mlp = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        y = self.mlp(x)
        # x = x + self.drop(y)
        x = self.drop(y)

        x = self.norm(x)
        return x
