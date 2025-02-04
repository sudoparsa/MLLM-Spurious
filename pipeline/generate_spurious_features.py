from env_vars import NLTK_CACHE_DIR, PIPELINE_STORAGE_DIR, OPENAI_API_KEY
import os
import pathlib
import nltk
from nltk.stem import WordNetLemmatizer
from typing import List
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
from openai import OpenAI
from functools import partial

def ask_gpt(system_prompt: str, user_prompt: str, client: OpenAI, model: str = 'gpt-4o-mini') -> str:
	messages = [
		{
			'role': 'system',
			'content': system_prompt
		},
		{
			'role': 'user',
			'content': user_prompt
		}
	]
	completion = client.chat.completions.create(
		model=model,
		messages=messages
	)
	return completion.choices[0].message.content

if NLTK_CACHE_DIR is not None:
	nltk.data.path.append(NLTK_CACHE_DIR)
wnl = WordNetLemmatizer()

def word_filter(class_name: str, w: str) -> bool:
	w = w.lower()
	for cw in class_name.split(' '):
		if cw in w:
			return False
	return True

system_prompt = "You are part of a study on spurious correlations in vision language models."

def check_exists_without(class_name: str, sf: str, client: OpenAI) -> bool:
	return 'No' not in ask_gpt(system_prompt, f"Can a {sf} exist without a {class_name}? Respond with 'Yes' or 'No'.", client)

def check_is_not_part(class_name: str, sf: str, client: OpenAI) -> bool:
	return 'Yes' not in ask_gpt(system_prompt, f"Is a {sf} part of a {class_name}? Respond with 'Yes' or 'No'.", client)

import inflect
p = inflect.engine()

def check_nonreliant(class_name: str, sf: str, client: OpenAI) -> bool:
	return 'Yes' not in ask_gpt(system_prompt, f"Do all or almost all {p.plural(class_name)} have a {sf}? Respond with 'Yes' or 'No'.", client)

def check_noninclusive(class_name: str, sf: str, client: OpenAI) -> bool:
	return 'Yes' not in ask_gpt(system_prompt, f"Do all or almost all {p.plural(sf)} have a {class_name}? Respond with 'Yes' or 'No'.", client)

def gen_spur_features(class_name: str, client: OpenAI, n: int = 16) -> List[str]:
	gpt_prompt_tail = "List exactly one item on a every consecutive line, followed by a period and a one sentence explanation. " + \
		"The object must be physical and discernable in an image. The object name must be less than two words. " + \
		"Do not number the responses. Do not output anything else."

	gpt_prompt_1 = f"List {n} objects that commonly appear in images of a {class_name}. The objects cannot be part of a {class_name}."
	gpt_resp1 = ask_gpt(system_prompt, gpt_prompt_1 + ' ' + gpt_prompt_tail, client)
	spur_features_obj = list(map(lambda s: s.split('.')[0].strip(), gpt_resp1.split('\n\n' if '\n\n' in gpt_resp1 else '\n')[:n]))

	gpt_prompt_2 = f"List {n} background elements that commonly appear in images of a {class_name}. The objects cannot be part of a {class_name}."
	gpt_resp2 = ask_gpt(system_prompt, gpt_prompt_2 + ' ' + gpt_prompt_tail, client)
	spur_features_bg = list(map(lambda s: s.split('.')[0].strip(), gpt_resp2.split('\n\n' if '\n\n' in gpt_resp2 else '\n')[:n]))
	
	all_spur_features = spur_features_obj + spur_features_bg
	processed = [wnl.lemmatize(w.lower()) for w in all_spur_features]
	processed = list(filter(lambda s: word_filter(class_name, s), processed))
	processed = list(filter(lambda s: check_exists_without(class_name, s, client), processed))
	processed = list(filter(lambda s: check_nonreliant(class_name, s, client), processed))
	processed = list(filter(lambda s: check_is_not_part(class_name, s, client), processed))
	processed = list(filter(lambda s: check_noninclusive(class_name, s, client), processed))
	return list(set(processed))

def get_spur_features_storage_dir(class_name: str) -> str:
	return os.path.join(PIPELINE_STORAGE_DIR, 'spurious_features', f"{class_name}.txt")

def write_spur_features(class_name: str, lst: List[str]):
	path = get_spur_features_storage_dir(class_name)
	with open(path, 'w') as f:
		for feat in lst:
			f.write(feat + '\n')

def read_spur_features(class_name: str) -> List[str]:
	path = get_spur_features_storage_dir(class_name)
	with open(path, 'r') as f:
		spur_features = [line.strip() for line in f.readlines()]
	return spur_features

if __name__ == '__main__':
	from utils import format_name
	import argparse
	parser = argparse.ArgumentParser(description="Generate Spurious Features")
	parser.add_argument(
		"--class_names",
		type=str,
		nargs='+',
		help="Class names to be passed to GPT to produce possible spurious features",
	)
	parser.add_argument(
		"--file_names",
		type=str,
		nargs="+",
		help="File names to write the results to. If not provided, will use the class names",
	)
	parser.add_argument(
		"--dataset_group",
		type=str,
		help="Fast way to generate spurious features for all classes in a registered dataset",
		choices=['hardimagenet', 'coco'],
	)
	parser.add_argument(
		"-n",
		type=int,
		default=16,
		help="Number of features to request in each prompt",
	)
	args = parser.parse_args()
	class_names = args.class_names
	n = args.n
	file_names = args.file_names
	dataset_group = args.dataset_group
	if dataset_group is not None: 
		if file_names is not None or class_names is not None:
			raise ValueError("Cannot provide class names or file names when using dataset group")
		if dataset_group == 'hardimagenet':
			from env_vars import HARDIMAGENET_PATH
			import pickle as pkl
			with open(f'{HARDIMAGENET_PATH}/meta/imagenet_classnames.pkl', 'rb') as f:
				imagenet_classnames: List[str] = pkl.load(f)
			with open(f'{HARDIMAGENET_PATH}/meta/hard_imagenet_idx.pkl', 'rb') as f:
				hard_imagenet_idx: List[int] = pkl.load(f)
			class_names, file_names = [], []
			for i in range(15):
				class_names.append(imagenet_classnames[hard_imagenet_idx[i]])
				file_names.append(f"hardimagenet-{i}")
		elif dataset_group == 'coco':
			pass # TODO
		else:
			raise ValueError(f"Dataset group '{dataset_group}' not supported")
	if file_names is not None:
		assert len(class_names) == len(file_names), \
			f"Number of class names and file names provided differ: {len(class_names)} class names, {len(file_names)} file names"

	client = OpenAI()

	for i, class_name in enumerate(class_names):
		lst = gen_spur_features(class_name, client, n)
		fname = format_name(class_name) if file_names is None else file_names[i]
		write_spur_features(fname, lst)
