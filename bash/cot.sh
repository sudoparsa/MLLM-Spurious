#!/bin/bash

#SBATCH --job-name=CoT                                   # sets the job name
#SBATCH --output=CoT.out.%j                               # indicates a file to redirect STDOUT to; %j is the jobid. If set, must be set to a file instead of a directory or else submission will fail.
#SBATCH --error=CoT.out.%j                                # indicates a file to redirect STDERR to; %j is the jobid. If set, must be set to a file instead of a directory or else submission will fail.
#SBATCH --time=01-23:59:00                                          # how long you would like your job to run; format=dd-hh:mm:ss
#SBATCH --account=nexus
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --mem=32g
#SBATCH --ntasks=1                                                   # cpu cores be reserved for your node total
#SBATCH --partition=tron
#SBATCH --qos=default


cd /fs/nexus-scratch/parsahs/spurious/vlm
pwd

. /usr/share/Modules/init/bash
. /etc/profile.d/ummodules.sh

module add Python3/3.12.7
module add cuda
source env/bin/activate


python --version

nvidia-smi

cd MLLM-Spurious


# python main.py --model llava --dataset hardimagenet --K 50 --experiment noobject_nospur --mode cot --drop_mask
# python main.py --model llava --dataset hardimagenet --K 50 --experiment noobject_spur --mode cot --drop_mask

python main.py --model qwen --dataset hardimagenet --K 50 --experiment object_nospur --mode cot
python main.py --model qwen --dataset hardimagenet --K 50 --experiment object_spur --mode cot


python main.py --model qwen --dataset hardimagenet --K 50 --experiment noobject_nospur --mode cot --drop_mask
python main.py --model qwen --dataset hardimagenet --K 50 --experiment noobject_spur --mode cot --drop_mask


# python main.py --model llava --dataset hardimagenet --K 50 --experiment object_nospur --mode cot
# python main.py --model llava --dataset hardimagenet --K 50 --experiment object_spur --mode cot

# python main.py --model llama --dataset hardimagenet --K 50 --experiment object_nospur --mode cot
# python main.py --model llama --dataset hardimagenet --K 50 --experiment object_spur --mode cot


# python main.py --model qwen --dataset hardimagenet --K 5 --experiment blank --mode cot 
# python main.py --model llava --dataset hardimagenet --K 5 --experiment blank --mode cot 
# python main.py --model llama --dataset hardimagenet --K 5 --experiment blank --mode cot

# python main.py --model qwen --dataset imagenet --K 50 --experiment object_spur --mode cot 
# python main.py --model qwen --dataset imagenet --K 50 --experiment object_nospur --mode cot 
# python main.py --model qwen --dataset imagenet --K 5 --experiment blank --mode cot 


# python main.py --model llava --dataset imagenet --K 5 --experiment blank --mode cot 
# python main.py --model llava --dataset imagenet --K 50 --experiment object_nospur --mode cot 
# python main.py --model llava --dataset imagenet --K 50 --experiment object_spur --mode cot 


# python main.py --model llama --dataset imagenet --K 5 --experiment blank --mode cot 
# python main.py --model llama --dataset imagenet --K 50 --experiment object_spur --mode cot 
# python main.py --model llama --dataset imagenet --K 50 --experiment object_nospur --mode cot 


# python main.py --model llama --dataset hardimagenet --K 50 --experiment noobject_nospur --mode unbiased --drop_mask
# python main.py --model llama --dataset hardimagenet --K 50 --experiment noobject_spur --mode unbiased --drop_mask

# python main.py --model llama --dataset hardimagenet --K 50 --experiment noobject_nospur --mode cot --drop_mask
# python main.py --model llama --dataset hardimagenet --K 50 --experiment noobject_spur --mode cot --drop_mask

