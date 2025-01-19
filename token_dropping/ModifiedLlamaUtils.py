import numpy as np
import torch
import transformers

def upscale(img: torch.Tensor, processor) -> torch.Tensor:
	from transformers.models.mllama.image_processing_mllama import ChannelDimension, to_channel_dimension_format
	data_format = ChannelDimension.FIRST
	upscaled_img = to_channel_dimension_format(img.numpy(), data_format)
	upscaled_img, aspect_ratio = processor.image_processor.resize(
		image=upscaled_img,
		size=processor.image_processor.size,
		resample=processor.image_processor.resample,
		max_image_tiles=processor.image_processor.max_image_tiles,
		input_data_format=data_format,
		data_format=data_format,
	)
	upscaled_img = processor.image_processor.pad(
		image=upscaled_img,
		size=processor.image_processor.size,
		aspect_ratio=aspect_ratio,
		input_data_format=data_format,
		data_format=data_format,
	)
	return torch.tensor(upscaled_img)


def morph_mask(mask: torch.Tensor, ps: int = 14) -> torch.Tensor:
	if len(mask.shape) == 3:
		mask = mask[0]
	assert len(mask.shape) == 2
	assert mask.shape[0] % (ps) == 0 and mask.shape[1] % ps == 0

	morphed_mask = []
	for i in range(mask.shape[0] // ps):
		row = []
		for j in range(mask.shape[1] // ps):
			patch = mask[ps*i:ps*(i+1), ps*j:ps*(j+1)]
			row.append(0 if patch.prod() == 0 else 1)
		morphed_mask.append(row)
	return torch.tensor(morphed_mask)


def expand_morphed_mask(morphed_mask: torch.Tensor, factor: int = 14) -> torch.Tensor:
	new_mask = []
	for i in range(morphed_mask.shape[0]):
		row = []
		for j in range(morphed_mask.shape[1]):
			for _ in range(factor):
				row.append([morphed_mask[i, j]])
		for _ in range(factor):
			new_mask.append(row)
	new_mask = torch.tensor(new_mask)
	
	return torch.stack([new_mask, new_mask, new_mask])
