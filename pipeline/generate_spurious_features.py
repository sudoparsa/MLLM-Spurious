from env_vars import NLTK_CACHE_DIR, PIPELINE_STORAGE_DIR, OPENAI_API_KEY
import os
import nltk
from nltk.stem import WordNetLemmatizer
from typing import List
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
from openai import OpenAI

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

nltk.data.path.append(NLTK_CACHE_DIR)
wnl = WordNetLemmatizer()

def gen_spur_features(class_name: str, client: OpenAI, n: int = 16) -> List[str]:
	system_prompt = "You are part of a study on spurious correlations in vision language models."
	gpt_prompt_tail = "List exactly one item on a every consecutive line, followed by a period and a one sentence explanation. The object must be physical and discernable in an image. The object name must be less than two words. Do not number the responses. Do not output anything else."
	def word_filter(w: str) -> bool:
		w = w.lower()
		for cw in class_name.split(' '):
			if cw in w:
				return False
		return True

	gpt_prompt_1 = f"List {n} objects that commonly appear in images of a {class_name}. " + gpt_prompt_tail
	gpt_resp1 = ask_gpt(system_prompt, gpt_prompt_1, client)
	spur_features_obj = list(map(lambda s: s.split('.')[0].strip(), gpt_resp1.split('\n\n' if '\n\n' in gpt_resp1 else '\n')[:n]))
	spur_features_obj = list(filter(word_filter, spur_features_obj))

	gpt_prompt_2 = f"List {n} background elements that commonly appear in images of a {class_name}. " + gpt_prompt_tail
	gpt_resp2 = ask_gpt(system_prompt, gpt_prompt_2, client)
	spur_features_bg = list(map(lambda s: s.split('.')[0].strip(), gpt_resp2.split('\n\n' if '\n\n' in gpt_resp2 else '\n')[:n]))
	spur_features_bg = list(filter(word_filter, spur_features_bg))
	
	all_spur_features = spur_features_obj + spur_features_bg
	processed = [wnl.lemmatize(w.lower()) for w in all_spur_features]
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
		required=True
	)
	parser.add_argument(
		"--file_names",
		type=str,
		nargs="+",
		help="File names to write the results to. If not provided, will use the class names",
		required=False
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
	if file_names is not None:
		assert len(class_names) == len(file_names), \
			f"Number of class names and file names provided differ: {len(class_names)} class names, {len(file_names)} file names"

	client = OpenAI()

	for i, class_name in enumerate(class_names):
		lst = gen_spur_features(class_name, client, n)
		fname = format_name(class_name) if file_names is None else file_names[i]
		write_spur_features(fname, lst)
