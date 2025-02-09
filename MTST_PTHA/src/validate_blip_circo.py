import multiprocessing
from argparse import ArgumentParser
from operator import itemgetter
from pathlib import Path
from statistics import mean
from typing import List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from lavis.models import load_model_and_preprocess
from data_utils import squarepad_transform, FashionIQDataset, targetpad_transform, CIRRDataset
from data_utils import CIRCODataset
from utils import extract_index_features, collate_fn, extract_index_blip_features, device
import os
from pathlib import Path
import shutil
import cv2
import json
from statistics import mean, geometric_mean, harmonic_mean


CIRCO_dataset_path = "{CIRCO_dataset_path}"
out_put_path = "{json_path}"

def compute_circo_val_metrics(relative_val_dataset: CIRCODataset, blip_model, index_features,
                             index_names: List[str], txt_processors) -> Tuple[
    float, float, float, float, float, float, float]:
    """
    Compute validation metrics on CIRR dataset
    :param relative_val_dataset: CIRR validation dataset in relative mode
    :param clip_model: CLIP model
    :param index_features: validation index features
    :param index_names: validation index names
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                            features
    :return: the computed validation metrics
    """
    # Generate predictions
    pred_sim, reference_names, target_names, captions_all,ids = \
        generate_circo_val_predictions(blip_model, relative_val_dataset, index_names, index_features, txt_processors)

    print("Compute CIRR validation metrics")

    # Compute the distances and sort the results
    distances = 1 - pred_sim
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Delete the reference image from the results
    reference_mask = torch.tensor(
        sorted_index_names != np.repeat(np.array(reference_names), len(index_names)).reshape(len(reference_names), -1))
    sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0],
                                                                    sorted_index_names.shape[1] - 1)
    rst_dict = {}
    for i,l in enumerate(sorted_index_names):
        rn = reference_names[i]
        rst_list = l[:50].tolist()
        rst_dict.update({str(ids[i]):rst_list})
    with open(out_put_path , "w") as f:
        json.dump(rst_dict, f)
    print("")




def generate_circo_val_predictions(blip_model, relative_val_dataset: CIRCODataset, 
                                  index_names: List[str], index_features, txt_processors) -> \
        Tuple[torch.tensor, List[str], List[str], List[List[str]]]:
    """
    Compute CIRCO predictions on the validation set
    :param clip_model: CLIP model
    :param relative_val_dataset: CIRR validation dataset in relative mode
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                            features
    :param index_features: validation index features
    :param index_names: validation index names
    :return: predicted features, reference names, target names and group members
    """
    print("Compute CIRCO validation predictions")
    relative_val_loader = DataLoader(dataset=relative_val_dataset, batch_size=2, num_workers=2,
                                     pin_memory=True, collate_fn=collate_fn)


    # Initialize predicted features, target_names, group_members and reference_names
    distance = []
    target_names = []
    group_members = []
    reference_names = []
    captions_all = []
    ids = []
    for item in tqdm(relative_val_loader):
        batch_reference_images = item['reference_img']
        batch_reference_names = item['reference_imd_id']
        captions = item['relative_caption']
        captions = [txt_processors["eval"](caption) for caption in captions]
        qid = item['query_id']
        with torch.no_grad():
    
            batch_reference_images = batch_reference_images.to(device, non_blocking=True)
            _, reference_image_features = blip_model.extract_target_features(batch_reference_images,  mode="mean")
            batch_distance = blip_model.inference(reference_image_features, index_features[0], captions)
            distance.append(batch_distance)
            captions_all += captions
        ids.extend(qid)
        reference_names.extend(batch_reference_names)
    
    distance = torch.vstack(distance)

    return distance, reference_names, target_names, captions_all,ids




def main():
    parser = ArgumentParser()
    parser.add_argument("--blip-model-name", default="RN50x4", type=str, help="CLIP model to use, e.g 'RN50', 'RN50x4'")
    parser.add_argument("--blip-model-path", type=Path, help="Path to the fine-tuned CLIP model")
    parser.add_argument("--target-ratio", default=1.25, type=float, help="TargetPad target ratio")
    parser.add_argument("--transform", default="targetpad", type=str,
                        help="Preprocess pipeline, should be in ['clip', 'squarepad', 'targetpad'] ")
    parser.add_argument("--ntype", default="val", type=str,help="ntype should be in ['val', 'test'] ")


    args = parser.parse_args()

    blip_validate_circo(args.blip_model_name, args.blip_model_path, args.transform, args.target_ratio,args.ntype)

def blip_validate_circo(blip_model_name, blip_model_path, transform, target_ratio,ntype):


    blip_model, _, txt_processors = load_model_and_preprocess(name=blip_model_name, model_type="pretrain", is_eval=False, device=device)

    checkpoint = torch.load(blip_model_path, map_location=device)
    msg = blip_model.load_state_dict(checkpoint[blip_model.__class__.__name__], strict=False)
    print("Missing keys {}".format(msg.missing_keys))

    input_dim = 224

    preprocess = targetpad_transform(1.25, input_dim)


    classic_val_dataset = CIRCODataset(f"{CIRCO_dataset_path}",ntype,'classic',preprocess)
    relative_val_dataset = CIRCODataset(f"{CIRCO_dataset_path}",ntype,'relative',preprocess)


    val_index_features, val_index_names = extract_index_blip_features(classic_val_dataset, blip_model)
    # 
    results = compute_circo_val_metrics(relative_val_dataset, blip_model, val_index_features,
                                        val_index_names, txt_processors)


if __name__ == '__main__':
    main()
