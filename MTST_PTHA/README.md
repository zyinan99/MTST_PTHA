
# Scale Up Composed Image Retrieval Learning via Modification Text Generation



### Prerequisites (Following SPRC)

	
The following commands will create a local Anaconda environment with the necessary packages installed.

```bash
conda create -n mtst -y python=3.9
conda activate mtst
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install -r requirements.txt
```


### Data Preparation (Following SPRC)

To properly work with the codebase FashionIQ and CIRR datasets should have the following structure:

```
project_base_path
└───  fashionIQ_dataset
      └─── captions
            | cap.dress.test.json
            | cap.dress.train.json
            | cap.dress.val.json
            | ...
            
      └───  images
            | B00006M009.jpg
            | B00006M00B.jpg
            | B00006M6IH.jpg
            | ...
            
      └─── image_splits
            | split.dress.test.json
            | split.dress.train.json
            | split.dress.val.json
            | ...

└───  cirr_dataset  
       └─── train
            └─── 0
                | train-10108-0-img0.png
                | train-10108-0-img1.png
                | train-10108-1-img0.png
                | ...
                
            └─── 1
                | train-10056-0-img0.png
                | train-10056-0-img1.png
                | train-10056-1-img0.png
                | ...
                
            ...
            
       └─── dev
            | dev-0-0-img0.png
            | dev-0-0-img1.png
            | dev-0-1-img0.png
            | ...
       
       └─── test1
            | test1-0-0-img0.png
            | test1-0-0-img1.png
            | test1-0-1-img0.png 
            | ...
       
       └─── cirr
            └─── captions
                | cap.rc2.test1.json
                | cap.rc2.train.json
                | cap.rc2.val.json
                
            └─── image_splits
                | split.rc2.test1.json
                | split.rc2.train.json
                | split.rc2.val.json
```

change ``base_path = Path("path_to_data_dir")'' in MTST_PTHA/src/data_utils.py
Download pretrained path from "https://huggingface.co/yinanzhou1/mtst_ptha/blob/main/tuned_mtst_cirr.pt"
Download generated finetuning data from "https://huggingface.co/yinanzhou1/mtst_ptha/blob/main/cirr_train_modifier_rev_long.json"
### Pre-Training



python src/blip_fine_tune_2.py \
   --dataset {'CIRRGen' or 'FashionIQGen'} \
   --blip-model-name 'blip2_cir_cat' \
   --num-epochs {'10' for CIRR, '30' for fashionIQ} \
   --num-workers 4 \
   --learning-rate {'1e-5' for CIRR, '2e-5' for fashionIQ} \
   --batch-size 128 \
   --transform targetpad \
   --target-ratio 1.25  \
   --save-training \
   --save-best \
   --validation-frequency 1 

### Training

change the path of ``pretrained: "PATH_TO_PRETRAINED_WEIGHTS" '' in MTST_PTHA/src/lavis/configs/models/blip2/blip2_pretrain_mtst.yaml for finetuning MTST pretrained model

```sh
python src/blip_fine_tune_2.py \
   --dataset {'CIRR' or 'FashionIQ'} \
   --blip-model-name 'blip2_tgir_ce_sim' \
   --num-epochs {'50' for CIRR, '30' for fashionIQ} \
   --num-workers 4 \
   --learning-rate {'5e-6' for CIRR, '1e-5' for fashionIQ} \
   --batch-size 128 \
   --transform targetpad \
   --target-ratio 1.25  \
   --save-training \
   --save-best \
   --validation-frequency 1 
   --backbone {"pretrain_mtst" for pretrained model using mtst, "pretrain" for initialized model from blip-2}
```

### Evaluation


```sh
python src/blip_validate.py \
   --dataset {'CIRR' or 'FashionIQ'} \
   --blip-model-name {trained model name} \
   --model-path {for path} 
```

### CIRR Testing

```sh
python src/cirr_test_submission.py \
   --blip-model-name {trained model name} \
   --model-path {for path} 
```


