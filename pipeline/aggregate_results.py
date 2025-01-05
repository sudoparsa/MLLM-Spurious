import os
import re
import pandas as pd
import functools
from typing import List
import pickle as pkl
import random
from env_vars import PIPELINE_STORAGE_DIR
from run_experiments import get_prompts
from utils import format_name


pat = re.compile(r"i=(\d+), img_type=(\w+), prompt_id=(\w+)-(\w+)-(\d+) :: res='(.*)'")

def get_results_df(dataset_name: str, mllm_name: str) -> pd.DataFrame:
	experiment_results_dir = f"/cmlscratch/snawathe/MLLM-Spurious/pipeline_storage/experiment_results/{dataset_name}/{mllm_name}/"
	result_filenames = os.listdir(experiment_results_dir)

	lst = []
	for filename in result_filenames:
		with open(os.path.join(experiment_results_dir, filename), 'r') as f:
			for line in f.readlines():
				info = pat.match(line.strip())
				lst.append((int(info[1]), info[2], info[3], info[4], int(info[5]), info[6]))
	df = pd.DataFrame(lst, columns=["img_idx", "img_type", "prompt_type", "prompt_obj_status", "prompt_variant", "resp"])

	all_prompts = get_prompts('', True) + get_prompts('', False)
	@functools.cache
	def get_correct_answer_from_id(id: str) -> str:
		return list(filter(lambda d: d['id'] == id, all_prompts))[0]['correct_answer']
	df['correct_ans'] = df.apply(lambda row: get_correct_answer_from_id('-'.join([row.prompt_type, row.prompt_obj_status, str(row.prompt_variant)])), axis=1)
	assert (~pd.isnull(df['correct_ans'])).all()
	df['correct'] = df.apply(lambda row: row.correct_ans in row.resp, axis=1)
	return df


def compute_single_score(df: pd.DataFrame, spur_feature_name: str, sorted_idxs: List[int], K: int = 50):
	bot_idxs = sorted_idxs[:K]
	top_idxs = sorted_idxs[-K:]

	score_natural_unbiased_plus = df[(df.img_idx.isin(top_idxs)) & (df.img_type == 'natural') & (df.prompt_type == 'unbiased')].correct.mean()
	score_natural_unbiased_minus = df[(df.img_idx.isin(bot_idxs)) & (df.img_type == 'natural') & (df.prompt_type == 'unbiased')].correct.mean()
	score_natural_unbiased = score_natural_unbiased_plus - score_natural_unbiased_minus

	score_natural_sycophantic_plus = df[(df.img_idx.isin(top_idxs)) & (df.img_type == 'natural') & (df.prompt_type == 'sycophantic')].correct.mean()
	score_natural_sycophantic_minus = df[(df.img_idx.isin(bot_idxs)) & (df.img_type == 'natural') & (df.prompt_type == 'sycophantic')].correct.mean()
	score_natural_sycophantic = score_natural_sycophantic_plus - score_natural_sycophantic_minus	

	score_masked_unbiased_plus = df[(df.img_idx.isin(bot_idxs)) & (df.img_type == 'masked') & (df.prompt_type == 'unbiased')].correct.mean()
	score_masked_unbiased_minus = df[(df.img_idx.isin(top_idxs)) & (df.img_type == 'masked') & (df.prompt_type == 'unbiased')].correct.mean()
	score_masked_unbiased = score_masked_unbiased_plus - score_masked_unbiased_minus

	score_masked_sycophantic_plus = df[(df.img_idx.isin(bot_idxs)) & (df.img_type == 'masked') & (df.prompt_type == 'sycophantic')].correct.mean()
	score_masked_sycophantic_minus = df[(df.img_idx.isin(top_idxs)) & (df.img_type == 'masked') & (df.prompt_type == 'sycophantic')].correct.mean()
	score_masked_sycophantic = score_masked_sycophantic_plus - score_masked_sycophantic_minus

	score_dropped_unbiased_plus = df[(df.img_idx.isin(bot_idxs)) & (df.img_type == 'dropped') & (df.prompt_type == 'unbiased')].correct.mean()
	score_dropped_unbiased_minus = df[(df.img_idx.isin(top_idxs)) & (df.img_type == 'dropped') & (df.prompt_type == 'unbiased')].correct.mean()
	score_dropped_unbiased = score_dropped_unbiased_plus - score_dropped_unbiased_minus

	score_dropped_sycophantic_plus = df[(df.img_idx.isin(bot_idxs)) & (df.img_type == 'dropped') & (df.prompt_type == 'sycophantic')].correct.mean()
	score_dropped_sycophantic_minus = df[(df.img_idx.isin(top_idxs)) & (df.img_type == 'dropped') & (df.prompt_type == 'sycophantic')].correct.mean()
	score_dropped_sycophantic = score_dropped_sycophantic_plus - score_dropped_sycophantic_minus

	return {
		'spur_feature_name': spur_feature_name,
		'hallucination_score': (score_natural_unbiased + score_masked_unbiased) / 2,
		'score_natural_unbiased': score_natural_unbiased,
		'score_natural_sycophantic': score_natural_sycophantic,
		'score_masked_unbiased': score_masked_unbiased,
		'score_masked_sycophantic': score_masked_sycophantic,
		'score_dropped_unbiased': score_dropped_unbiased,
		'score_dropped_sycophantic': score_dropped_sycophantic
	}

def compute_scores(dataset_name: str, mllm_name: str, ranking_model: str, spur_feat_file: str, K: int = 50, include_random: bool = False) -> pd.DataFrame:
	with open(os.path.join(PIPELINE_STORAGE_DIR, 'spurious_features', spur_feat_file), 'r') as f:
		all_spur_features = [line.strip() for line in f.readlines()]
	df = get_results_df(dataset_name, mllm_name)

	lst = []
	for spur_feat in all_spur_features:
		spur_feat_name = format_name(spur_feat)
		with open(os.path.join(PIPELINE_STORAGE_DIR, 'rankings', dataset_name, ranking_model, f"{spur_feat_name}.pkl"), 'rb') as f:
			sorted_idxs = pkl.load(f)
		lst.append(compute_single_score(df, spur_feat, sorted_idxs, K))
	
	if include_random:
		sorted_idxs = list(range(len(sorted_idxs)))
		random.shuffle(sorted_idxs)
		lst.append(compute_single_score(df, 'random ordering', sorted_idxs, K))

	res_df = pd.DataFrame(lst)
	res_df = res_df.sort_values(by='hallucination_score', ascending=False)
	return res_df




