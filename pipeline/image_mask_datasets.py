import os
from typing import List, Union, Optional, Literal, Dict, Any
from dataclasses import dataclass
import torch
from torchvision import transforms
from PIL import Image
import re
import numpy as np
import pickle as pkl
import json
from env_vars import IMAGENET_PATH, HARDIMAGENET_PATH, COCO_PATH
from pycocotools.coco import COCO


@dataclass
class ImageMask():
	image: torch.Tensor
	mask: Optional[torch.Tensor]


class ImageMaskDataset():
	to_tens = transforms.ToTensor()

	def __init__(
		self,
		images: Union[str, List[str]],
		masks: Optional[Union[str, List[str]]] = None,
		class_name: Optional[str] = None
	):
		if isinstance(images, str):
			images = os.listdir(images)
		self.images = images
		if masks is not None and isinstance(masks, str):
			masks = os.listdir(masks)
		self.masks = masks
		if self.masks is not None:
			assert len(self.images) == len(self.masks), \
				f"Not the same number of images and masks: {len(self.images)} images, {len(self.masks)} masks"
		self.class_name = class_name

	def __len__(self) -> int:
		return len(self.images)

	def __getitem__(self, idx: int) -> ImageMask:
		return ImageMask(self.get_image(idx), self.get_mask(idx))

	def get_image(self, idx: int) -> torch.Tensor:
		return self.to_tens(self.get_pil_image(idx))

	def get_pil_image(self, idx: int) -> Image:
		return Image.open(self.images[idx]).convert('RGB')

	def get_mask(self, idx: int) -> Optional[torch.Tensor]:
		return self.to_tens(self.get_pil_mask(idx))

	def get_pil_mask(self, idx: int) -> Optional[torch.Tensor]:
		return None if self.masks is None else Image.open(self.masks[idx])
	
	def get_class_name(self) -> str:
		if self.class_name is None:
			raise Exception('Dataset class name not provided')
		return self.class_name


class CocoImageMaskDataset(ImageMaskDataset):
	to_pil = transforms.ToPILImage()

	def __init__(self, coco_category_idx: int, split: Union[Literal['train'], Literal['val']] = 'train'):
		self.coco_category_idx = coco_category_idx
		annotation_file = os.path.join(COCO_PATH, 'annotations', f'instances_{split}2017.json')
		self.coco = COCO(annotation_file=annotation_file)
		with open(annotation_file, 'r') as f:
			self.instances_data = json.load(f)
		self.img_ids = self.get_coco_images_ids_with_obj(coco_category_idx)
		self.images = self.get_coco_image_paths_from_ids(self.img_ids, split)
		relevant_annotations = self.get_relevant_annotations(self.img_ids, coco_category_idx)
		self.masks = list(map(self.combine_segmentations, relevant_annotations))
		self.class_name = list(filter(lambda d: d['id'] == coco_category_idx, self.instances_data['categories']))[0]['name']
	
	def get_mask(self, idx: int) -> Image:
		return torch.tensor(self.masks[idx])[torch.newaxis, :, :]
	
	def get_pil_mask(self, idx):
		return self.to_pil(self.get_mask(idx))
	
	def get_coco_images_ids_with_obj(self, category_id: int) -> List[int]:
		s = set()
		for annotation in self.instances_data['annotations']:
			if annotation['category_id'] == category_id:
				s.add(annotation['image_id'])
		lst = list(s)
		lst.sort()
		return lst

	def get_coco_image_paths_from_ids(self, lst: List[int], split: Union[Literal['train'], Literal['val']] = 'train') -> List[str]:
		ans = [None] * len(lst)
		id_to_idx = {id: i for i, id in enumerate(lst)}
		for img_data in self.instances_data['images']:
			if img_data['id'] in id_to_idx:
				os.path.join(COCO_PATH, 'images', )
				ans[id_to_idx[img_data['id']]] = f"/fs/cml-datasets/coco/images/{split}2017/{img_data['file_name']}"
		assert all(x is not None for x in ans), "Illegal ids passed in lst"
		return ans

	def get_relevant_annotations(self, img_id_lst, cat_id):
		ans = [list() for _ in range(len(img_id_lst))]
		id_to_idx = {id: i for i, id in enumerate(img_id_lst)}
		for annotation in self.instances_data['annotations']:
			if annotation['category_id'] == cat_id and annotation['image_id'] in id_to_idx:
				j = id_to_idx[annotation['image_id']]
				ans[j].append(annotation)
		assert all(len(x) > 0 for x in ans), "At least one ID does not have an associated annotation"
		return ans
	
	def combine_segmentations(self, seg_lst):
		if len(seg_lst) == 0:
			raise Exception()
		first = self.coco.annToMask(seg_lst[0])
		if len(seg_lst) == 1:
			return first
		rest = self.combine_segmentations(seg_lst[1:])
		return np.maximum(first, rest)


def get_image_mask_dataset(name: str) -> ImageMaskDataset:
	hmn_pat = re.compile(r"hardimagenet-(\d+)")
	hmn_res = hmn_pat.match(name)
	if hmn_res is not None:
		class_idx = int(hmn_res[1])
		with open(f'{HARDIMAGENET_PATH}/meta/paths_by_rank.pkl', 'rb') as f:
			paths_by_rank: List[str] = pkl.load(f)
		with open(f'{HARDIMAGENET_PATH}/meta/imagenet_classnames.pkl', 'rb') as f:
			imagenet_classnames: List[str] = pkl.load(f)
		with open(f'{HARDIMAGENET_PATH}/meta/hard_imagenet_idx.pkl', 'rb') as f:
			hard_imagenet_idx: List[int] = pkl.load(f)
		images_lst, masks_lst = [], []
		for sample_idx in range(len(paths_by_rank[hard_imagenet_idx[class_idx]])):
			split, x, xy = paths_by_rank[hard_imagenet_idx[class_idx]][sample_idx].split('/')
			_, y = xy.split('.')[0].split('_')
			split, x, y
			images_lst.append(IMAGENET_PATH + '/' + paths_by_rank[hard_imagenet_idx[class_idx]][sample_idx])
			masks_lst.append(os.path.join(HARDIMAGENET_PATH, split, f"{x}_{x}_{y}.JPEG"))
		class_name = imagenet_classnames[hard_imagenet_idx[class_idx]]
		return ImageMaskDataset(images_lst, masks_lst, class_name)
	
	coco_pat = re.compile(r"coco-(\d+)")
	coco_res = coco_pat.match(name)
	if coco_res is not None:
		class_idx = int(coco_res[1])
		return CocoImageMaskDataset(class_idx, 'train')

	raise Exception(f"Dataset '{name}' not available")

