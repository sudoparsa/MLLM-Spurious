import torch
import math

def morph_mask(mask: torch.Tensor, ps: int = 14) -> torch.Tensor:
	if len(mask.shape) == 3:
		mask = mask[0]
	assert len(mask.shape) == 2
	# assert mask.shape[0] % (ps) == 0 and mask.shape[1] % ps == 0

	morphed_mask = []
	for i in range(int(math.ceil(mask.shape[0] / ps))):
		row = []
		for j in range(int(math.ceil(mask.shape[1] / ps))):
			patch = mask[ps*i:ps*(i+1), ps*j:ps*(j+1)]
			row.append(0 if patch.prod() == 0 else 1)
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
