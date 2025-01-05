import numpy as np

def format_name(name: str) -> str:
	return '_'.join(name.split(' '))

def get_corners(arr):
    on_pixels = np.where(arr != 0)
    x_max, y_max = [np.max(on_pixels[i]) for i in [1,2]]
    x_min, y_min = [np.min(on_pixels[i]) for i in [1,2]]
    return x_min, x_max, y_min, y_max

def get_bbox(arr, expand=False):
    out = np.zeros_like(arr)
    if arr.sum() > 0:
        x_min, x_max, y_min, y_max = get_corners(arr)
        out[:, x_min:x_max, y_min:y_max] = 1
    return out
