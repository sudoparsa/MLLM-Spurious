# Pipeline to Explore Spurious Features in MLLMs

### Getting Started

Assuming you have a Python instance with all necessary packages installed, GPU access, and have configured all environment variables in `env_vars.py`, and set up the basic folder structure, the following commands will run the pipeline on HardImageNet Class 0 "dog sled", using Qwen2 as the MLLM and OWLv2 as the object detector.

```bash
python ./pipeline/generate_spurious_features.py --class_names "dog sled" --file_names "hardimagenet-0"

for ityp in natural masked dropped; do
	python ./pipeline/run_experiments.py --dataset=hardimagenet-0 --class_name="dog sled" --img_type=$ityp --mllm=qwen
done

python ./pipeline/run_object_detector.py --dataset=hardimagenet-0 --model=owl --spur_feat_file=hardimagenet-0.txt 

python ./pipeline/compute_rankings.py --dataset=hardimagenet-0 --model=owl --spur_feat_file=hardimagenet-0.txt
```

Afterwards, the functions in `aggregate_results.py` can be used in a notebook to construct DataFrames.

Note that some of these scripts come with chunking capabilities: the dataset can be divided into disjoint pieces, so multiple GPUs can process the sections concurrently. See the command line arguments for further details.


### Recommended Folder Structure

The pipeline assumes the following structure: let PIPELINE_CACHE_DIR point to a folder 'pipeline_storage/' that is a sibling of 'pipeline/'.

- pipeline/
- pipeline_storage/
	- experiment_results/
	- object_detection/
	- rankings/
	- spurious_features/

For rankings, MLLM experiment results, and object detection results, they will be stored as pickle files under names like '<dataset_name>/<model_name>/asdf.pkl' in their respective directories.
	


### Models and Datasets

Available MLLMs:
- Qwen2VL

Available object detectors for ranking:
- OWLv2
- Grounding Dino

HardImageNet and COCO are available out-of-the-box. To add additional datasets, you must add a dataloader in `image_mask_dataset.py` and register it in `get_image_mask_dataset()`.
