#!/bin/bash

#SBATCH --job-name=log
#SBATCH --output=qwen_drop/qwen_nodrop.txt
#SBATCH --error=qwen_drop/qwen_nodrop_err.txt
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --account=nexus
#SBATCH --qos=medium
#SBATCH --mem=64GB
#SBATCH --ntasks=2
#SBATCH

cd /cmlscratch/snawathe/MLLM-Spurious/

# /nfshomes/snawathe/micromamba/envs/spurious-test/bin/python main.py --model qwen --dataset hardimagenet --mode unbiased --experiment noobject_spur --drop_mask --K 50 &> qwen_drop/qwen_drop_unbiased_noobject_spur
# /nfshomes/snawathe/micromamba/envs/spurious-test/bin/python main.py --model qwen --dataset hardimagenet --mode unbiased --experiment noobject_nospur --drop_mask --K 50 &> qwen_drop/qwen_drop_unbiased_noobject_nospur

/nfshomes/snawathe/micromamba/envs/spurious-test/bin/python main.py --model qwen --dataset hardimagenet --mode sycophantic --experiment noobject_spur --drop_mask --K 50 &> qwen_drop/qwen_drop_sycophantic_noobject_spur
# /nfshomes/snawathe/micromamba/envs/spurious-test/bin/python main.py --model qwen --dataset hardimagenet --mode sycophantic --experiment noobject_nospur --drop_mask --K 50 &> qwen_drop/qwen_drop_sycophantic_noobject_nospur
