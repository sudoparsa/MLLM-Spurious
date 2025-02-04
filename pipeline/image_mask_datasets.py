import os
from typing import List, Union, Optional, Literal, Dict, Any
from dataclasses import dataclass
import torch
from torchvision import transforms
from PIL import Image
import re
from collections import namedtuple
import numpy as np
import pickle as pkl
import json
from env_vars import IMAGENET_PATH, HARDIMAGENET_PATH, COCO_PATH, CITYSCAPES_PATH
from pycocotools.coco import COCO


@dataclass
class ImageMask():
	image: torch.Tensor
	mask: Optional[torch.Tensor]


class ImageMaskDataset():
	to_tens = transforms.ToTensor()
	to_pil = transforms.ToPILImage()

	def __init__(
		self,
		images: Union[str, List[str]],
		masks: Optional[Union[str, List[str]]] = None,
		class_name: Optional[str] = None
	):
		if isinstance(images, str):
			images = sorted(os.listdir(images))
		self.images = images
		if masks is not None and isinstance(masks, str):
			masks = sorted(os.listdir(masks))
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


# class LargeImageMaskDataset(ImageMaskDataset):
# 	def get_class_name(self, idx: int) -> str:
# 		if self.class_name is None:
# 			raise Exception('Dataset class name not set')
# 		return self.class_name[idx]


class CocoCategoryImageMaskDataset(ImageMaskDataset):
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


class CityscapesImageMaskDataset(ImageMaskDataset):
	# https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
	Label = namedtuple(
		'Label',
		['name', 'id', 'trainId', 'category', 'categoryId', 'hasInstances', 'ignoreInEval', 'color']
	)
	labels = [
		#       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
		Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
		Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
		Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
		Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
		Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
		Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
		Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
		Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
		Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
		Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
		Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
		Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
		Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
		Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
		Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
		Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
		Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
		Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
		Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
		Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
		Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
		Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
		Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
		Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
		Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
		Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
		Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
		Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
		Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
		Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
		Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
		Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
		Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
		Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
		Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
	]

	def __init__(self, cityscape_label_id: int, split: Union[Literal['train'], Literal['val']] = 'train'):
		self.cityscape_label_id = cityscape_label_id
		self.class_name = list(filter(lambda lab: lab.id == cityscape_label_id, self.labels))[0].name

		split = 'val'
		cities = sorted(os.listdir(os.path.join(CITYSCAPES_PATH, f'gtFine/gtFine/{split}/')))
		path_prefixes = []
		for city in cities:
			for filename in sorted(os.listdir(os.path.join(CITYSCAPES_PATH, f'gtFine/gtFine/{split}/', city))):
				if filename.split('.')[-1] == 'json':
					path = os.path.join(CITYSCAPES_PATH, f'gtFine/gtFine/{split}/', city, filename)
					path_prefixes.append(path.split('_gtFine')[0] + '_gtFine')
		
		def has_label_id(path_prefix: str, label_id: int) -> bool:
			path = path_prefix + '_labelIds.png'
			img = Image.open(path)
			img = self.to_tens(img)
			return (torch.round(img * 255).int() == label_id).any()
		self.path_prefixes = list(filter(lambda p: has_label_id(p, cityscape_label_id), path_prefixes))
	
	def __len__(self) -> int:
		return len(self.path_prefixes)

	def get_pil_image(self, idx):
		path = self.path_prefixes[idx].replace('gtFine', 'leftImg8bit') + '.png'
		return Image.open(path)
	
	def get_mask(self, idx):
		path = self.path_prefixes[idx] + '_labelIds.png'
		mask = Image.open(path)
		mask = transforms.ToTensor()(mask)
		return (torch.round(mask * 255).int() == self.cityscape_label_id).int()
	
	def get_pil_mask(self, idx):
		return self.to_pil(self.get_mask(idx))


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
		return CocoCategoryImageMaskDataset(class_idx, 'train')
	
	cityscape_pat = re.compile(r"cityscape-(\d+)")
	cityscape_res = cityscape_pat.match(name)
	if cityscape_res is not None:
		class_idx = int(cityscape_res[1])
		return CityscapesImageMaskDataset(class_idx, 'train')

	raise Exception(f"Dataset '{name}' not available")

