import os
from typing import List, Union, Optional
from dataclasses import dataclass
import torch
from torchvision import transforms
from PIL import Image
import re
import pickle as pkl

@dataclass
class ImageMask():
	image: torch.Tensor
	mask: Optional[torch.Tensor]

class ImageMaskDataset():
	to_tens = transforms.ToTensor()

	def __init__(
		self,
		images: Union[str, List[str]],
		masks: Optional[Union[str, List[str]]] = None
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
	
	def reorder(self, idx_lst: List[int]):
		assert len(idx_lst) == len(self)
		self.images = [self.images[i] for i in idx_lst]
		if self.masks is not None:
			self.masks = [self.masks[i] for i in idx_lst]


def get_image_mask_dataset(name: str) -> ImageMaskDataset:
	hmn_pat = re.compile(r"hardimagenet-(\d+)")
	hmn_res = hmn_pat.match(name)
	if hmn_res is not None:
		class_idx = int(hmn_res[1])
		from env_vars import IMAGENET_PATH, HARDIMAGENET_PATH
		with open(f'{HARDIMAGENET_PATH}/meta/paths_by_rank.pkl', 'rb') as f:
			paths_by_rank = pkl.load(f)
		with open(f'{HARDIMAGENET_PATH}/meta/hard_imagenet_idx.pkl', 'rb') as f:
			hard_imagenet_idx = pkl.load(f)
		images_lst, masks_lst = [], []
		for sample_idx in range(len(paths_by_rank[hard_imagenet_idx[class_idx]])):
			split, x, xy = paths_by_rank[hard_imagenet_idx[class_idx]][sample_idx].split('/')
			_, y = xy.split('.')[0].split('_')
			split, x, y
			images_lst.append(IMAGENET_PATH + '/' + paths_by_rank[hard_imagenet_idx[class_idx]][sample_idx])
			masks_lst.append(os.path.join(HARDIMAGENET_PATH, split, f"{x}_{x}_{y}.JPEG"))
		return ImageMaskDataset(images_lst, masks_lst)
	raise Exception(f"Dataset '{name}' not available")

