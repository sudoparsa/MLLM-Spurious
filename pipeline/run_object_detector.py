import torch
from torchvision import transforms
from typing import List, Any
from transformers import Owlv2ForObjectDetection, Owlv2Processor, GroundingDinoModel, GroundingDinoProcessor
from typing import Callable
import functools

to_pil = transforms.ToPILImage()

def run_owl(texts: List[str], img: torch.Tensor, owl_model: Owlv2ForObjectDetection, owl_processor: Owlv2Processor):
	pil_img = to_pil(img)
	with torch.no_grad():
		output = owl_model(**owl_processor(text=[texts], images=pil_img, return_tensors='pt').to(owl_model.device))
	res = owl_processor.post_process_object_detection(outputs=output, target_sizes=torch.Tensor([pil_img.size[::-1]]), threshold=0.05)
	del pil_img, output
	return res

def load_run_owl(owl_model: Owlv2ForObjectDetection, owl_processor: Owlv2Processor) -> Callable[[List[str], torch.Tensor], Any]:
	return functools.partial(run_owl, owl_model=owl_model, owl_processor=owl_processor)

def run_dino_sep(texts: List[str], img: torch.Tensor, dino_model: GroundingDinoModel, dino_processor: GroundingDinoProcessor):
	pil_img = to_pil(img)
	res_lst = []
	for t in texts:
		t = t + '.'
		inputs = dino_processor(text=t, images=pil_img, return_tensors='pt').to(dino_model.device)
		with torch.no_grad():
			output = dino_model(**inputs)
		res = dino_processor.post_process_grounded_object_detection(
			output,
			inputs.input_ids,
			target_sizes=[pil_img.size[::-1]]
		)
		res_lst.append(res)
	return res_lst

def load_run_dino_sep(dino_model: GroundingDinoModel, dino_processor: GroundingDinoProcessor) -> Callable[[List[str], torch.Tensor], Any]:
	return functools.partial(run_dino_sep, dino_model=dino_model, dino_processor=dino_processor)


if __name__ == '__main__':
	from env_vars import CACHE_DIR, PIPELINE_STORAGE_DIR
	from image_mask_datasets import get_image_mask_dataset
	import math
	import gc
	import pickle as pkl
	import argparse
	import os
	import pathlib

	parser = argparse.ArgumentParser(description="Running Object Detection")
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
		help="File with a list of spurious features",
		required=True
	)
	parser.add_argument(
		"--num_tot_chunks",
		type=int,
		default=1,
		help="Number of chunks to divide the dataset into"
	)
	parser.add_argument(
		"--chunk",
		type=int,
		default=0,
		help="Index of chunk to run"
	)
	args = parser.parse_args()
	dataset_name = args.dataset
	model_name = args.model
	spur_feat_file = args.spur_feat_file
	num_tot_chunks = args.num_tot_chunks
	chunk = args.chunk

	device = "cuda" if torch.cuda.is_available() else "cpu"
	if model_name == 'owl':
		model_id = "google/owlv2-base-patch16-ensemble"
		owl_model = Owlv2ForObjectDetection.from_pretrained(model_id, cache_dir=CACHE_DIR).to(device)
		owl_processor = Owlv2Processor.from_pretrained(model_id, cache_dir=CACHE_DIR)
		objdet_func = load_run_owl(owl_model, owl_processor)
	elif model_name == 'dino':
		model_id = "IDEA-Research/grounding-dino-base"
		dino_model = GroundingDinoModel.from_pretrained(model_id, cache_dir=CACHE_DIR).to(device)
		dino_processor = GroundingDinoProcessor.from_pretrained(model_id, cache_dir=CACHE_DIR)
		objdet_func = load_run_dino_sep(dino_model, dino_processor)
	else:
		raise Exception(f"Model '{model_name}' is not supported")
	
	dataset = get_image_mask_dataset(dataset_name)
	num_samples = len(dataset)
	num_samples_per_chunk = math.ceil(num_samples/num_tot_chunks)
	chunk_start = num_samples_per_chunk * chunk
	chunk_end = min(num_samples_per_chunk * (chunk + 1), num_samples)

	downsize = transforms.Compose([transforms.Resize(size=14*35), transforms.ToPILImage(), transforms.ToTensor()])
	with open(os.path.join(PIPELINE_STORAGE_DIR, 'spurious_features', spur_feat_file), 'r') as f:
		all_spur_features = [line.strip() for line in f.readlines()]
	pathlib.Path(os.path.join(PIPELINE_STORAGE_DIR, 'object_detection', dataset_name, model_name)).mkdir(parents=True, exist_ok=True)

	for i in range(chunk_start, chunk_end):
		img = dataset.get_image(i)
		if any(d > 14*35 for d in img.shape):
			img = downsize(img)
		res = objdet_func(all_spur_features, img)
		with open(os.path.join(PIPELINE_STORAGE_DIR, 'object_detection', dataset_name, model_name, f"{i}.pkl"), 'wb') as f:
			pkl.dump(res, f)
		del img, res
		if i % 30 == 0:
			gc.collect()
			torch.cuda.empty_cache()

