import os
import re
import logging
import argparse
from PIL import Image
from utils.utils import *
from openai import OpenAI
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers import MllamaForConditionalGeneration, AutoProcessor



device = 'cuda'
_MASK_ROOT = '/workspace/MLLM-Spurious/hardImageNet'
_IMAGENET_ROOT = '/workspace/MLLM-Spurious/HardImageNet_Images'
HARD_IMAGE_NET_DIR = '/workspace/MLLM-Spurious/hardImageNet'
SPURIOUS_IMAGENET_DIR = "/workspace/MLLM-Spurious/images"
COCO_PATH = "/p/vast1/cai6/sumit_storage/coco"
CACHE_DIR = "/p/vast1/cai6/parsahs/huggingface/hub"


def get_model(args):
    min_pixels = 256*28*28
    max_pixels = 2048*28*28
    if args.model == 'qwen':
        model_id = "Qwen/Qwen2-VL-7B-Instruct"
        if not args.drop_mask:
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_id, torch_dtype="auto", device_map=device, cache_dir=CACHE_DIR
            )
            processor = AutoProcessor.from_pretrained(model_id, min_pixels=min_pixels, max_pixels=max_pixels, cache_dir=CACHE_DIR)
        else:
            model = ModifiedQwen2VLForConditionalGeneration.from_pretrained(
                model_id, torch_dtype="auto", device_map=device,  cache_dir=CACHE_DIR
            )
            processor = ModifiedQwen2VLProcessor.from_pretrained(model_id,  cache_dir=CACHE_DIR)
    
    elif args.model =='llama':
        model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
        if not args.drop_mask:
            model = MllamaForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype="auto",
                device_map=device,
                cache_dir=CACHE_DIR
            )
        else:
            model = ModifiedMllamaForConditionalGeneration.from_pretrained(
				model_id, torch_dtype="auto", device_map=device,  cache_dir=CACHE_DIR
			).to(device)
        processor = AutoProcessor.from_pretrained(model_id, min_pixels=min_pixels, max_pixels=max_pixels, cache_dir=CACHE_DIR)
    
    elif args.model =='llava':
        model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
        if not args.drop_mask:
            model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype="auto",
            device_map=device,
            cache_dir=CACHE_DIR)
            processor = LlavaNextProcessor.from_pretrained(model_id, min_pixels=min_pixels, max_pixels=max_pixels, cache_dir=CACHE_DIR)
        else:
            model = ModifiedLlavaNextForConditionalGeneration.from_pretrained(
				model_id, torch_dtype="auto", device_map=device,  cache_dir=CACHE_DIR
			).to(device)
            processor = ModifiedLlavaNextProcessor.from_pretrained(model_id, min_pixels=min_pixels, max_pixels=max_pixels,cache_dir=CACHE_DIR)
        processor.patch_size = model.config.vision_config.patch_size
        processor.vision_feature_select_strategy = model.config.vision_feature_select_strategy
    
    elif args.model == 'gpt-4o':
        model = 'gpt-4o'
        processor = OpenAI()
    
    else:
        logger.info('Invalid Model')
    
    logger.info(f"{type(model)=}")
    logger.info(f"{type(processor)=}")

    return model, processor


######################
## Running the VLLM ##
######################

def vllm_standard_preprocessing(model, processor, prompt, image, **processor_kwargs):
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": ""}
            ]}
    ]
    messages[0]['content'][1]['text'] = prompt
    text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        text=[text_prompt], images=[image], padding=True, return_tensors="pt",
        **processor_kwargs
    ).to(device)
    return inputs


def vllm_decoding(inputs, output_ids, processor) -> str:
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    return output_text[0]


def get_vllm_output(model, processor, prompt, image, max_new_tokens=2048):
    if model == 'gpt-4o':
        return get_gpt_output(processor, prompt, image)
    # package inputs in expected format
    inputs = vllm_standard_preprocessing(model, processor, prompt, image)
    # Inference: Generation of the output
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    # decoding
    return vllm_decoding(inputs, output_ids, processor)



def regex_coco(file_path):
    results = []
    pattern = (
        r"^FAILURE ::: cat=(?P<cat>.*?) ::: idx=\d+ ::: path=(?P<path>.*?) ::: "
        r"annots=(?P<annots>\[.*?\]) ::: prompt=.* ::: res=.*$")
    
    with open(file_path, 'r') as file:
        for line in file:
            match = re.match(pattern, line.strip())
            if match:
                # Extract the named groups 'cat', 'path', and 'annots'
                results.append({
                    'cat': match.group('cat'),
                    'path': match.group('path'),
                    'annots': match.group('annots'),
                })
    
    return results

def analyze_coco(args, logger):
    logs_dir = f"log/{args.dataset}/{args.model}/"
    prompt = f"Is there a CLASSNAME in the image? Explain."


    failures = []
    for filename in os.listdir(logs_dir):
        if filename.startswith(f"{args.mode}-{args.experiment}") and filename.endswith('.log'):
            failures += regex_coco(os.path.join(logs_dir, filename))
    
    cat_path_dict = {}
    for failure in failures:
        cat = failure['cat']
        path = failure['path']
        annots = failure['annots']
        if cat in cat_path_dict.keys():
            if path not in cat_path_dict[cat]['path']:
                cat_path_dict[cat]['path'] += [path]
                cat_path_dict[cat]['annots'] += [annots]
        else:
            cat_path_dict[cat] = {'path': [path], 'annots': [annots]}
    
    model, processor = get_model(args)

    for cat in cat_path_dict.keys():
        cat_prompt = prompt.replace("CLASSNAME", cat)
        logger.info(f"### Category: {cat} ###")
        logger.info(f"Prompt: {cat_prompt}")
        for i, path in enumerate(cat_path_dict[cat]['path']):
            image = Image.open(path).convert('RGB')
            output = get_vllm_output(model, processor, cat_prompt, image, max_new_tokens=256)
            logger.info(f"### {i} ###")
            logger.info(f"Output: {output}")
            logger.info(f"Path: {path}")
        logger.info('\n')


def analyze(args, logger):
    if args.dataset == 'coco':
        return analyze_coco(args, logger)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Seeing What's Not There")
    parser.add_argument(
        "--model",
        type=str,
        default="qwen",
        choices=["qwen", "llama", "llava", "gpt-4o"],
        help="model name",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="hardimagenet",
        choices=["hardimagenet", "imagenet", "spurious_imagenet", "coco"],
        help="dataset name",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="unbiased",
        choices=["unbiased", "sycophantic", "cot"],
        help="prompt's type",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="object_spur",
        choices=["object_spur", "object_nospur", "noobject_spur", "noobject_nospur", "blank"],
        help="object and spurious cues are present or not",
    )
    parser.add_argument('--drop_mask', action='store_true')

    args = parser.parse_args()

    logging_level = logging.INFO

    # create folder
    os.makedirs(f"results", exist_ok=True)
    os.makedirs(f"results/{args.dataset}", exist_ok=True)
    os.makedirs(f"results/{args.dataset}/{args.model}", exist_ok=True)

    logging.basicConfig(format="### %(message)s ###")  # level=logging_level,

    logger = logging.getLogger("Spur")
    logger.setLevel(level=logging_level)
    LOG_NAME = f"{args.mode}-{args.experiment}"
    logger.addHandler(logging.FileHandler(f"results/{args.dataset}/{args.model}/{LOG_NAME}.log", mode='w'))

    analyze(args, logger)
