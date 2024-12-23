#!/bin/bash

#SBATCH --job-name=log
#SBATCH --output=qwen_nodrop/qwen_nodrop.txt
#SBATCH --error=qwen_nodrop/qwen_nodrop_err.txt
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:rtxa6000:2
#SBATCH --account=nexus
#SBATCH --qos=medium
#SBATCH --mem=64GB
#SBATCH --ntasks=2
#SBATCH

cd /cmlscratch/snawathe/MLLM-Spurious/

# /nfshomes/snawathe/micromamba/envs/spurious-test/bin/python main.py --model qwen --dataset hardimagenet --mode unbiased --experiment object_spur --K 50 &> qwen_nodrop/qwen_nodrop_unbiased_object_spur
# /nfshomes/snawathe/micromamba/envs/spurious-test/bin/python main.py --model qwen --dataset hardimagenet --mode unbiased --experiment object_nospur --K 50 &> qwen_nodrop/qwen_nodrop_unbiased_object_nospur
# /nfshomes/snawathe/micromamba/envs/spurious-test/bin/python main.py --model qwen --dataset hardimagenet --mode unbiased --experiment noobject_spur --K 50 &> qwen_nodrop/qwen_nodrop_unbiased_noobject_spur
# /nfshomes/snawathe/micromamba/envs/spurious-test/bin/python main.py --model qwen --dataset hardimagenet --mode unbiased --experiment noobject_nospur --K 50 &> qwen_nodrop/qwen_nodrop_unbiased_noobject_nospur

/nfshomes/snawathe/micromamba/envs/spurious-test/bin/python main.py --model qwen --dataset hardimagenet --mode sycophantic --experiment object_spur --K 50 &> qwen_nodrop/qwen_nodrop_sycophantic_object_spur
/nfshomes/snawathe/micromamba/envs/spurious-test/bin/python main.py --model qwen --dataset hardimagenet --mode sycophantic --experiment object_nospur --K 50 &> qwen_nodrop/qwen_nodrop_sycophantic_object_nospur
/nfshomes/snawathe/micromamba/envs/spurious-test/bin/python main.py --model qwen --dataset hardimagenet --mode sycophantic --experiment noobject_spur --K 50 &> qwen_nodrop/qwen_nodrop_sycophantic_noobject_spur
/nfshomes/snawathe/micromamba/envs/spurious-test/bin/python main.py --model qwen --dataset hardimagenet --mode sycophantic --experiment noobject_nospur --K 50 &> qwen_nodrop/qwen_nodrop_sycophantic_noobject_nospur

/nfshomes/snawathe/micromamba/envs/spurious-test/bin/python main.py --model qwen --dataset hardimagenet --mode unbiased --experiment blank --K 50 &> qwen_nodrop/qwen_nodrop_unbiased_blank
/nfshomes/snawathe/micromamba/envs/spurious-test/bin/python main.py --model qwen --dataset hardimagenet --mode sycophantic --experiment blank --K 50 &> qwen_nodrop/qwen_nodrop_sycophantic_blank

