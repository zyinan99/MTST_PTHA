import json
from pathlib import Path
from typing import Union, List, Dict, Literal

import PIL
import PIL.Image
from PIL import Image
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch 
import torch.utils.data
import torchvision
import os
base_path = Path("path_to_data_dir")

def collate_fn(batch):
    '''
    function which discard None images in a batch when using torch DataLoader
    :param batch: input_batch
    :return: output_batch = input_batch - None_values
    '''
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def _convert_image_to_rgb(image):
    return image.convert("RGB")


class SquarePad:
    """
    Square pad the input image with zero padding
    """

    def __init__(self, size: int):
        """
        For having a consistent preprocess pipeline with CLIP we need to have the preprocessing output dimension as
        a parameter
        :param size: preprocessing output dimension
        """
        self.size = size

    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = [hp, vp, hp, vp]
        return F.pad(image, padding, 0, 'constant')


class TargetPad:
    """
    Pad the image if its aspect ratio is above a target ratio.
    Pad the image to match such target ratio
    """

    def __init__(self, target_ratio: float, size: int):
        """
        :param target_ratio: target ratio
        :param size: preprocessing output dimension
        """
        self.size = size
        self.target_ratio = target_ratio

    def __call__(self, image):
        w, h = image.size
        actual_ratio = max(w, h) / min(w, h)
        if actual_ratio < self.target_ratio:  # check if the ratio is above or below the target ratio
            return image
        scaled_max_wh = max(w, h) / self.target_ratio  # rescale the pad to match the target ratio
        hp = max(int((scaled_max_wh - w) / 2), 0)
        vp = max(int((scaled_max_wh - h) / 2), 0)
        padding = [hp, vp, hp, vp]
        return F.pad(image, padding, 0, 'constant')


def squarepad_transform(dim: int):
    """
    CLIP-like preprocessing transform on a square padded image
    :param dim: image output dimension
    :return: CLIP-like torchvision Compose transform
    """
    return Compose([
        SquarePad(dim),
        Resize(dim, interpolation=PIL.Image.BICUBIC),
        CenterCrop(dim),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def targetpad_transform(target_ratio: float, dim: int):
    """
    CLIP-like preprocessing transform computed after using TargetPad pad
    :param target_ratio: target ratio for TargetPad
    :param dim: image output dimension
    :return: CLIP-like torchvision Compose transform
    """
    return Compose([
        TargetPad(target_ratio, dim),
        Resize(dim, interpolation=PIL.Image.BICUBIC),
        CenterCrop(dim),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


class FashionIQDataset(Dataset):
    """
    FashionIQ dataset class which manage FashionIQ data.
    The dataset can be used in 'relative' or 'classic' mode:
        - In 'classic' mode the dataset yield tuples made of (image_name, image)
        - In 'relative' mode the dataset yield tuples made of:
            - (reference_image, target_image, image_captions) when split == train
            - (reference_name, target_name, image_captions) when split == val
            - (reference_name, reference_image, image_captions) when split == test
    The dataset manage an arbitrary numbers of FashionIQ category, e.g. only dress, dress+toptee+shirt, dress+shirt...
    """

    def __init__(self, split: str, dress_types: List[str], mode: str, preprocess: callable):
        """
        :param split: dataset split, should be in ['test', 'train', 'val']
        :param dress_types: list of fashionIQ category
        :param mode: dataset mode, should be in ['relative', 'classic']:
            - In 'classic' mode the dataset yield tuples made of (image_name, image)
            - In 'relative' mode the dataset yield tuples made of:
                - (reference_image, target_image, image_captions) when split == train
                - (reference_name, target_name, image_captions) when split == val
                - (reference_name, reference_image, image_captions) when split == test
        :param preprocess: function which preprocesses the image
        """
        self.mode = mode
        self.dress_types = dress_types
        self.split = split

        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")
        if split not in ['test', 'train', 'val']:
            raise ValueError("split should be in ['test', 'train', 'val']")
        for dress_type in dress_types:
            if dress_type not in ['dress', 'shirt', 'toptee']:
                raise ValueError("dress_type should be in ['dress', 'shirt', 'toptee']")

        self.preprocess = preprocess

        # get triplets made by (reference_image, target_image, a pair of relative captions)
        self.triplets: List[dict] = []
        if split != 'train':
            for dress_type in dress_types:
                with open(base_path / 'fashionIQ_dataset' / 'captions' / f'cap.{dress_type}.{split}.json') as f:
                    self.triplets.extend(json.load(f))
        else:    
            for dress_type in dress_types:
                with open(base_path / 'fashionIQ_dataset' / 'captions' / f'cap_{split}_{dress_type}_rev.json') as f:
                    for line in f:
                        json_data = json.loads(line)
                        self.triplets.append(json_data)



        # get the image names
        self.image_names: list = []
        for dress_type in dress_types:
            with open(base_path / 'fashionIQ_dataset' / 'image_splits' / f'split.{dress_type}.{split}.json') as f:
                self.image_names.extend(json.load(f))
        for image_name in self.image_names[:]:
            file_path = base_path / 'fashionIQ_dataset' / 'images' /f"{image_name}.png"
            if not file_path.exists():
                self.image_names.remove(image_name)

        print(f"FashionIQ {split} - {dress_types} dataset in {mode} mode initialized")

    def __getitem__(self, index):
        try:
            if self.mode == 'relative':
                image_captions = self.triplets[index]['captions']
                reference_name = self.triplets[index]['candidate'].replace("jpg","png")

                if self.split == 'train':
                    image_captions_rev = self.triplets[index]['modifier_generated']

                    reference_image_path = base_path / 'fashionIQ_dataset' / 'images' / f"{reference_name}"

                    reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                    target_name = self.triplets[index]['target'].replace("jpg","png")
                    target_image_path = base_path / 'fashionIQ_dataset' / 'images' / f"{target_name}"


                    target_image = self.preprocess(PIL.Image.open(target_image_path))
                    return reference_image, target_image, image_captions,image_captions_rev

                elif self.split == 'val':
                    target_name = self.triplets[index]['target'].replace("jpg","png")
                    return reference_name, target_name, image_captions

                elif self.split == 'test':
                    reference_image_path = base_path / 'fashionIQ_dataset' / 'images' / f"{reference_name}"
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                    return reference_name, reference_image, image_captions

            elif self.mode == 'classic':
                image_name = self.image_names[index]
                image_path = base_path / 'fashionIQ_dataset' / 'images' / f"{image_name}.png"
                image = self.preprocess(PIL.Image.open(image_path))
                return image_name, image

            else:
                raise ValueError("mode should be in ['relative', 'classic']")
        except Exception as e:
            print(f"Exception: {e}")

    def __len__(self):
        if self.mode == 'relative':
            return len(self.triplets)
        elif self.mode == 'classic':
            return len(self.image_names)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")


class CIRRDataset(Dataset):
    """
       CIRR dataset class which manage CIRR data
       The dataset can be used in 'relative' or 'classic' mode:
           - In 'classic' mode the dataset yield tuples made of (image_name, image)
           - In 'relative' mode the dataset yield tuples made of:
                - (reference_image, target_image, rel_caption) when split == train
                - (reference_name, target_name, rel_caption, group_members) when split == val
                - (pair_id, reference_name, rel_caption, group_members) when split == test1
    """

    def __init__(self, split: str, mode: str, preprocess: callable):
        """
        :param split: dataset split, should be in ['test', 'train', 'val']
        :param mode: dataset mode, should be in ['relative', 'classic']:
                  - In 'classic' mode the dataset yield tuples made of (image_name, image)
                  - In 'relative' mode the dataset yield tuples made of:
                        - (reference_image, target_image, rel_caption) when split == train
                        - (reference_name, target_name, rel_caption, group_members) when split == val
                        - (pair_id, reference_name, rel_caption, group_members) when split == test1
        :param preprocess: function which preprocesses the image
        """
        self.preprocess = preprocess
        self.mode = mode
        self.split = split

        if split not in ['test1', 'train', 'val']:
            raise ValueError("split should be in ['test1', 'train', 'val']")
        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")
        self.triplets: List[dict] = []

        if split!='train':
            with open(base_path / 'cirr_dataset' / 'cirr' / 'captions' / f'cap.rc2.{split}.json') as f:
            # with open(base_path /  'cirr' / 'captions' / f'cap.rc2.{split}.json') as f:
                self.triplets = json.load(f)

            
        else: 
            with open(Path(base_path) / f'cirr_dataset/cirr/captions/cirr_train_modifier_rev_long.json') as f:
                for line in f:
                    json_data = json.loads(line)
                    self.triplets.append(json_data)


        # get a mapping from image name to relative path
        with open(base_path / 'cirr_dataset' / 'cirr' / 'image_splits' / f'split.rc2.{split}.json') as f:
            self.name_to_relpath = json.load(f)

        print(f"CIRR {split} dataset in {mode} mode initialized")

    def __getitem__(self, index):
        try:
            if self.mode == 'relative':
                group_members = self.triplets[index]['img_set']['members']
                reference_name = self.triplets[index]['reference']
                rel_caption = self.triplets[index]['caption']

                if self.split == 'train':
                    reference_image_path = base_path/ 'cirr_dataset' / self.name_to_relpath[reference_name]
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                    target_hard_name = self.triplets[index]['target_hard']
                    target_image_path = base_path / 'cirr_dataset' / self.name_to_relpath[target_hard_name]
                    target_image = self.preprocess(PIL.Image.open(target_image_path))
                    modifier_generated = self.triplets[index]['modifier_generated']

                    return reference_image, target_image, rel_caption, modifier_generated

                elif self.split == 'val':
                    target_hard_name = self.triplets[index]['target_hard']
                    # rel_caption = self.triplets[index]['modifier_generated']

                    return reference_name, target_hard_name, rel_caption, group_members

                elif self.split == 'test1':
                    pair_id = self.triplets[index]['pairid']
                    return pair_id, reference_name, rel_caption, group_members

            elif self.mode == 'classic':
                image_name = list(self.name_to_relpath.keys())[index]
                image_path = base_path/ 'cirr_dataset'  / self.name_to_relpath[image_name]
                im = PIL.Image.open(image_path)
                image = self.preprocess(im)
                return image_name, image

            else:
                raise ValueError("mode should be in ['relative', 'classic']")

        except Exception as e:
            print(f"Exception: {e}")

    def __len__(self):
        if self.mode == 'relative':
            return len(self.triplets)
        elif self.mode == 'classic':
            return len(self.name_to_relpath)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")

class CIRRDatasetGen(Dataset):
    """
       CIRR dataset class which manage CIRR data
       The dataset can be used in 'relative' or 'classic' mode:
           - In 'classic' mode the dataset yield tuples made of (image_name, image)
           - In 'relative' mode the dataset yield tuples made of:
                - (reference_image, target_image, rel_caption) when split == train
                - (reference_name, target_name, rel_caption, group_members) when split == val
                - (pair_id, reference_name, rel_caption, group_members) when split == test1
    """
    def __init__(self, split: str, mode: str, preprocess: callable):
        """
        :param split: dataset split, should be in ['test', 'train', 'val']
        :param mode: dataset mode, should be in ['relative', 'classic']:
                  - In 'classic' mode the dataset yield tuples made of (image_name, image)
                  - In 'relative' mode the dataset yield tuples made of:
                        - (reference_image, target_image, rel_caption) when split == train
                        - (reference_name, target_name, rel_caption, group_members) when split == val
                        - (pair_id, reference_name, rel_caption, group_members) when split == test1
        :param preprocess: function which preprocesses the image
        """
        self.preprocess = preprocess
        self.mode = mode
        self.split = split

        if split not in ['test1', 'train', 'val']:
            raise ValueError("split should be in ['test1', 'train', 'val']")
        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")
        self.triplets: List[dict] = []

        if split != 'train':
            with open(base_path / 'cirr' / 'captions' / f'cap.rc2.{split}.json') as f:
                self.triplets.extend(json.load(f))
        else:    

            with open(Path(base_path) / f'cirr_dataset/cirr/captions/cirrr_modifier_generated_5p_completed_rev.json') as f:

                for line in f:
                    json_data = json.loads(line)
                    self.triplets.append(json_data)


        # get a mapping from image name to relative path
        if split != 'train':

            with open(base_path /  'cirr' / 'image_splits' / f'split.rc2.{split}.json') as f:
                self.name_to_relpath = json.load(f)
        else:
            with open(base_path /  'cirr' / 'image_splits' / f'split.rc2.{split}.generated.json') as f:
                self.name_to_relpath = json.load(f)

        print(f"CIRR {split} dataset in {mode} mode initialized")

    def __getitem__(self, index):
        try:
            if self.mode == 'relative':
                reference_name = self.triplets[index]['reference']


                if self.split == 'train':
                    rel_caption = self.triplets[index]['modifier_generated']
                    rev_caption = self.triplets[index]['modifier_reversed']
                    reference_image_path = base_path / 'cirr' / self.name_to_relpath[reference_name]
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                    target_hard_name = self.triplets[index]['target']
                    target_image_path = base_path / 'cirr' / self.name_to_relpath[target_hard_name]
                    target_image = self.preprocess(PIL.Image.open(target_image_path))
                    return reference_image, target_image, rel_caption, rev_caption

                elif self.split == 'val':
                    rel_caption = self.triplets[index]['caption']
                    # rel_caption = self.triplets[index]['modifier_generated']

                    target_hard_name = self.triplets[index]['target_hard']
                    group_members = self.triplets[index]['img_set']['members']

                    return reference_name, target_hard_name, rel_caption, group_members

                elif self.split == 'test1':
                    rel_caption = self.triplets[index]['caption']
                    pair_id = self.triplets[index]['pairid']
                    group_members = self.triplets[index]['img_set']['members']
                    return pair_id, reference_name, rel_caption, group_members

            elif self.mode == 'classic':
                image_name = list(self.name_to_relpath.keys())[index]
                image_path = base_path / 'cirr' / self.name_to_relpath[image_name]
                im = PIL.Image.open(image_path)
                image = self.preprocess(im)
                return image_name, image

            else:
                raise ValueError("mode should be in ['relative', 'classic']")

        except Exception as e:
            print(f"Exception: {e}")

    def __len__(self):
        if self.mode == 'relative':
            return len(self.triplets)
        elif self.mode == 'classic':
            return len(self.name_to_relpath)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")


class CIRCODataset(Dataset):
    """
    CIRCO dataset
    """

    def __init__(self, data_path: Union[str, Path], split: Literal['val', 'test'],
                 mode: Literal['relative', 'classic'], preprocess: callable):
        """
        Args:
            data_path (Union[str, Path]): path to CIRCO dataset
            split (str): dataset split, should be in ['test', 'val']
            mode (str): dataset mode, should be in ['relative', 'classic']
            preprocess (callable): function which preprocesses the image
        """

        # Set dataset paths and configurations
        data_path = Path(data_path)
        self.mode = mode
        self.split = split
        self.preprocess = preprocess
        self.data_path = data_path

        # Ensure input arguments are valid
        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")
        if split not in ['test', 'val']:
            raise ValueError("split should be in ['test', 'val']")

        # Load COCO images information
        with open(data_path / 'unlabeled2017' / "annotations" / "image_info_unlabeled2017.json", "r") as f:
            imgs_info = json.load(f)

        self.img_paths = [data_path  / "unlabeled2017" / img_info["file_name"] for img_info in
                          imgs_info["images"]]
        self.img_ids = [img_info["id"] for img_info in imgs_info["images"]]
        self.img_ids_indexes_map = {str(img_id): i for i, img_id in enumerate(self.img_ids)}

        # get CIRCO annotations
        with open(data_path / 'annotations' / f'{split}.json', "r") as f:
            self.annotations: List[dict] = json.load(f)


        # Get maximum number of ground truth images (for padding when loading the images)
        self.max_num_gts = 23  # Maximum number of ground truth images

        print(f"CIRCODataset {split} dataset in {mode} mode initialized")

    def get_target_img_ids(self, index) -> Dict[str, int]:
        """
        Returns the id of the target image and ground truth images for a given query

        Args:
            index (int): id of the query

        Returns:
             Dict[str, int]: dictionary containing target image id and a list of ground truth image ids
        """

        return {
            'target_img_id': self.annotations[index]['target_img_id'],
            'gt_img_ids': self.annotations[index]['gt_img_ids']
        }

    def __getitem__(self, index) -> dict:
        """
        Returns a specific item from the dataset based on the index.

        In 'classic' mode, the dataset yields a dictionary with the following keys: [img, img_id]
        In 'relative' mode, the dataset yields dictionaries with the following keys:
            - [reference_img, reference_img_id, target_img, target_img_id, relative_caption, shared_concept, gt_img_ids,
            query_id] if split == val
            - [reference_img, reference_img_id, relative_caption, shared_concept, query_id]  if split == test
        """

        if self.mode == 'relative':
            # Get the query id
            query_id = str(self.annotations[index]['id'])

            # Get relative caption and shared concept
            relative_caption = self.annotations[index]['relative_caption']
            shared_concept = self.annotations[index]['shared_concept']

            # Get the reference image
            reference_img_id = str(self.annotations[index]['reference_img_id'])
            reference_img_path = self.img_paths[self.img_ids_indexes_map[reference_img_id]]
            reference_img = self.preprocess(PIL.Image.open(reference_img_path))

            if self.split == 'val':
                # Get the target image and ground truth images
                target_img_id = str(self.annotations[index]['target_img_id'])
                gt_img_ids = [str(x) for x in self.annotations[index]['gt_img_ids']]
                target_img_path = self.img_paths[self.img_ids_indexes_map[target_img_id]]
                target_img = self.preprocess(PIL.Image.open(target_img_path))

                # Pad ground truth image IDs with zeros for collate_fn
                gt_img_ids += [''] * (self.max_num_gts - len(gt_img_ids))

                return {
                    'reference_img': reference_img,
                    'reference_imd_id': reference_img_id,
                    'target_img': target_img,
                    'target_img_id': target_img_id,
                    'relative_caption': relative_caption,
                    'shared_concept': shared_concept,
                    'gt_img_ids': gt_img_ids,
                    'query_id': query_id,
                }

            elif self.split == 'test':
                return {
                    'reference_img': reference_img,
                    'reference_imd_id': reference_img_id,
                    'relative_caption': relative_caption,
                    'shared_concept': shared_concept,
                    'query_id': query_id,
                }

        elif self.mode == 'classic':
            # Get image ID and image path
            img_id = str(self.img_ids[index])
            img_path = self.img_paths[index]

            # Preprocess image and return
            img = self.preprocess(PIL.Image.open(img_path))
            return {
                'img': img,
                'img_id': img_id
            }

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        if self.mode == 'relative':
            return len(self.annotations)
        elif self.mode == 'classic':
            return len(self.img_ids)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")

class COCODataset(Dataset):

    def __init__(self, root_dir, preprocess: callable) -> None:
        super().__init__()

        self.root_dir = Path(base_path) / 'genecis_data/val2017'
        
        self.preprocess = preprocess


    def load_sample(self, sample):

        val_img_id = sample['val_image_id']
        fpath = os.path.join(self.root_dir, f'{val_img_id:012d}.jpg')
        img = Image.open(fpath)
        image = self.preprocess(img)
        
        # if self.transform is not None:
        #     img = self.transform(img)

        return image

class COCOValSubset(COCODataset):

    def __init__(self, val_split_path, tokenizer=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        with open(val_split_path) as f:
            val_samples = json.load(f)

        self.val_samples = val_samples
        # self.tokenizer = tokenizer

    def __getitem__(self, index):
        
        """
        Follow same return signature as CIRRSubset
        """

        sample = self.val_samples[index]
        reference = sample['reference']

        target = sample['target']
        gallery = sample['gallery']
        caption = sample['condition']

        reference, target = [self.load_sample(i) for i in (reference, target)]
        gallery = [self.load_sample(i) for i in gallery]

        if self.preprocess is not None:
            gallery = torch.stack(gallery)
            gallery_and_target = torch.cat([target.unsqueeze(0), gallery])
        else:
            gallery_and_target = [target] + gallery

        # if self.tokenizer is not None:
        #     caption = self.tokenizer(caption)

        # By construction, target_rank = 0
        return reference, caption, gallery_and_target, 0  

    def __len__(self):
        return len(self.val_samples)

        

DILATION = 0.7
PAD_CROP = True

def expand2square(pil_img, background_color=(0, 0, 0)):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

class VAWDataset(Dataset):

    def __init__(self,  image_dir, preprocess: callable) -> None:
        super().__init__()

        self.image_dir = Path(base_path) /"genecis_data/VG_100K"
        self.preprocess = preprocess



    def load_cropped_image(self, img):

        image_id = img['image_id']
        # bbox = img['instance_bbox']
        
        # Get image
        path = os.path.join(self.image_dir, f'{image_id}.jpg')
        im = Image.open(path)
        image = self.preprocess(im)
        

        return image

class VAWValSubset(VAWDataset):

    def __init__(self, val_split_path, tokenizer=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        with open(val_split_path) as f:
            val_samples = json.load(f)

        self.val_samples = val_samples
        # self.tokenizer = tokenizer

    def __getitem__(self, index):
        
        """
        Follow same return signature as CIRRSubset
            (Except for returning reference object at the end)
        """

        sample = self.val_samples[index]
        reference = sample['reference']

        target = sample['target']
        gallery = sample['gallery']
        caption = sample['condition']

        reference, target = [self.load_cropped_image(i) for i in (reference, target)]
        gallery = [self.load_cropped_image(i) for i in gallery]

        if self.preprocess is not None:
            gallery = torch.stack(gallery)
            gallery_and_target = torch.cat([target.unsqueeze(0), gallery])
        else:
            gallery_and_target = [target] + gallery

        # if self.tokenizer is not None:
        #     caption = self.tokenizer(caption)

        # By construction, target_rank = 0
        return reference, caption, gallery_and_target, 0

    def __len__(self):
        return len(self.val_samples)

