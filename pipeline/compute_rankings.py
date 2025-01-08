from typing import List, Any, Callable, Union
import os
import random

def extract_feature_score_owl(owl_res: Any, spur_feat_idx: int) -> float:
	owl_res = owl_res[0]
	res_idxs = [j for j in range(len(owl_res['labels'])) if owl_res['labels'][j] == spur_feat_idx]
	owl_scores: List[float] = [owl_res['scores'][j] for j in res_idxs]
	return max(owl_scores) if len(owl_scores) > 0 else 0

def extract_feature_score_dino(dino_res: Any, spur_feat_idx: int) -> float:
	dino_scores: List[float] = dino_res[spur_feat_idx][0]['scores']
	return max(dino_scores) if len(dino_scores) > 0 else 0

def rank_images(
		results: Union[str, List[str]],
		spur_feat_idx: str,
		extract_feature_score: Callable[[Any, int], float],
		randomize_before: bool
	) -> List[int]:
	if isinstance(results, str):
		results = [os.path.join(results, fname) for fname in os.listdir(results)]
	# print(f"first path {results[0]=}")
	lst = []
	for i, res_path in enumerate(results):
		with open(res_path, 'rb') as f:
			res = pkl.load(f)
		score = extract_feature_score(res, spur_feat_idx)
		j = int(res_path.split('/')[-1].split('.')[0])
		lst.append((score, j))
	if randomize_before:
		random.shuffle(lst)
	lst.sort()
	return list(map(lambda t: t[1], lst))


if __name__ == '__main__':
	from env_vars import CACHE_DIR, PIPELINE_STORAGE_DIR
	from image_mask_datasets import get_image_mask_dataset
	import math
	import gc
	import pickle as pkl
	import argparse
	import os
	import pathlib
	from utils import format_name

	parser = argparse.ArgumentParser(description="Ranking Dataset by Object Detection Scores")
	parser.add_argument(
		"--dataset",
		type=str,
		help="Dataset to run on (must be registered in `image_mask_dataset.py`)",
		required=True
	)
	parser.add_argument(
		"--model",
		type=str,
		choices=['owl', 'dino'],
		help="Object detection model to use",
		required=True
	)
	parser.add_argument(
		"--spur_feat_file",
		type=str,
		help="File with a list of spurious features (same as that used to generate object detection results)",
		required=True
	)
	parser.add_argument(
		"--no_randomize_before",
		help="Don't randomize the image order before sorting. Randomizing aims to prevent the same images showing up in all ranking extremes",
		default=False,
		action='store_true'
	)
	args = parser.parse_args()
	dataset_name = args.dataset
	model_name = args.model
	spur_feat_file = args.spur_feat_file
	randomize_before = (not args.no_randomize_before)
	print(f"{randomize_before=}")

	if model_name == 'owl':
		extract_feat_score = extract_feature_score_owl
	elif model_name == 'dino':
		extract_feat_score = extract_feature_score_dino
	else:
		raise Exception(f"Model '{model_name}' is not supported")

	results_dir = os.path.join(PIPELINE_STORAGE_DIR, 'object_detection', dataset_name, model_name)
	with open(os.path.join(PIPELINE_STORAGE_DIR, 'spurious_features', spur_feat_file), 'r') as f:
		all_spur_features = [line.strip() for line in f.readlines()]
	pathlib.Path(os.path.join(PIPELINE_STORAGE_DIR, 'rankings', dataset_name, model_name)).mkdir(parents=True, exist_ok=True)

	for i, spur_feat in enumerate(all_spur_features):
		# print(f"{i=}, {spur_feat=}", flush=True)
		sorted_idxs = rank_images(results_dir, i, extract_feat_score, randomize_before)
		with open(os.path.join(PIPELINE_STORAGE_DIR, 'rankings', dataset_name, model_name, f"{format_name(spur_feat)}.pkl"), 'wb') as f:
			pkl.dump(sorted_idxs, f)
		if i % 5 == 0:
			gc.collect()

