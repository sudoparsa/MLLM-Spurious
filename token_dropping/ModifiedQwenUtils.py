import torch
import numpy
import numpy.typing as npt

def rescale_tensor(img: npt.NDArray, processor, upscale_factor = 1, patch_size: int = 14, mf : int = 2) -> npt.NDArray:
	"""Does the same rescaling as performed by Qwen2VLImageProcessor, with optional upscaling"""
	from transformers.models.qwen2_vl.image_processing_qwen2_vl import infer_channel_dimension_format, get_image_size, smart_resize, resize
	from transformers.image_transforms import resize
	import PIL

	# from Qwen2VLImageProcessor._preprocess in do_resize section
	input_data_format = infer_channel_dimension_format(img)
	height, width = get_image_size(img, channel_dim=input_data_format)
	hp, wp = smart_resize(
		height,
		width,
		factor=processor.image_processor.patch_size * processor.image_processor.merge_size,
		min_pixels=processor.image_processor.min_pixels,
		max_pixels=processor.image_processor.max_pixels,
	)
	rescaled_img = resize(
		img, size=(hp, wp), resample=PIL.Image.Resampling.BICUBIC, input_data_format=input_data_format
	)

	# optional upscaling; set upscale_factor to 1 to do nothing
	upscaled_img = resize(
		image=rescaled_img,
		size=(rescaled_img.shape[1]*upscale_factor, rescaled_img.shape[2]*upscale_factor),
		resample=PIL.Image.Resampling.BICUBIC
	)

	return upscaled_img



def morph_mask(mask: torch.Tensor, ps: int = 14, mf: int = 2) -> torch.Tensor:
	"""
	Patchification for masks: takes the mask and creates mask of patches which is a superset of the mask.
	'ps' is patch size, 'mf' is merge factor (Qwen merges 2x2 patches)
	"""
	if len(mask.shape) == 3:
		mask = mask[0]
	assert len(mask.shape) == 2
	assert mask.shape[0] % (ps*mf) == 0 and mask.shape[1] % (ps*mf) == 0

	morphed_mask = []
	for i in range(mask.shape[0] // (ps*mf)):
		row = []
		for j in range(mask.shape[1] // (ps*mf)):
			patch = mask[(ps*mf)*i:(ps*mf)*(i+1), (ps*mf)*j:(ps*mf)*(j+1)]
			for _ in range(mf):
				row.append(0 if patch.prod() == 0 else 1)
		for _ in range(mf):
			morphed_mask.append(row)
	return torch.tensor(morphed_mask)



def expand_morphed_mask(morphed_mask: torch.Tensor, factor: int = 14) -> torch.Tensor:
	"""
	Takes the output of morph_mask() and expands the patches back to pixels.
	Useful for displaying the dropped tokens on the original image.
	"""
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



