#!/bin/bash

#SBATCH --job-name=SpNet                                   # sets the job name
#SBATCH --output=SpNet.out.%j                               # indicates a file to redirect STDOUT to; %j is the jobid. If set, must be set to a file instead of a directory or else submission will fail.
#SBATCH --error=SpNet.out.%j                                # indicates a file to redirect STDERR to; %j is the jobid. If set, must be set to a file instead of a directory or else submission will fail.
#SBATCH --time=01-00:00:00                                          # how long you would like your job to run; format=dd-hh:mm:ss
#SBATCH --account=nexus
#SBATCH --gres=gpu:rtxa5000:3
#SBATCH --mem=128g
#SBATCH --ntasks=1                                                   # cpu cores be reserved for your node total
#SBATCH --partition=tron
#SBATCH --qos=high


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


CUDA_VISIBLE_DEVICES=0 python main.py --model qwen --dataset spurious_imagenet --K 75 --experiment noobject_spur --mode twostepv1 &
CUDA_VISIBLE_DEVICES=1 python main.py --model qwen --dataset spurious_imagenet --K 75 --experiment noobject_spur --mode twostepv2 &
CUDA_VISIBLE_DEVICES=2 python main.py --model qwen --dataset spurious_imagenet --K 75 --experiment noobject_nospur --mode twostepv1

wait

CUDA_VISIBLE_DEVICES=0 python main.py --model qwen --dataset spurious_imagenet --K 75 --experiment noobject_nospur --mode twostepv2 &
CUDA_VISIBLE_DEVICES=1 python main.py --model llava --dataset spurious_imagenet --K 75 --experiment noobject_spur --mode twostepv1 &
CUDA_VISIBLE_DEVICES=2 python main.py --model llava --dataset spurious_imagenet --K 75 --experiment noobject_spur --mode twostepv2

wait

CUDA_VISIBLE_DEVICES=0 python main.py --model qwen --dataset spurious_imagenet --K 75 --experiment blank --mode unbiased &
CUDA_VISIBLE_DEVICES=1 python main.py --model qwen --dataset spurious_imagenet --K 75 --experiment blank --mode twostepv1 &
CUDA_VISIBLE_DEVICES=2 python main.py --model qwen --dataset spurious_imagenet --K 75 --experiment blank --mode twostepv2

wait


CUDA_VISIBLE_DEVICES=0 python main.py --model llava --dataset spurious_imagenet --K 75 --experiment noobject_nospur --mode twostepv1 &
CUDA_VISIBLE_DEVICES=1 python main.py --model llava --dataset spurious_imagenet --K 75 --experiment noobject_nospur --mode twostepv2 &
CUDA_VISIBLE_DEVICES=2 python main.py --model llama --dataset spurious_imagenet --K 75 --experiment noobject_spur --mode twostepv2

wait

CUDA_VISIBLE_DEVICES=0 python main.py --model llava --dataset spurious_imagenet --K 75 --experiment blank --mode unbiased &
CUDA_VISIBLE_DEVICES=1 python main.py --model llava --dataset spurious_imagenet --K 75 --experiment blank --mode twostepv1 &
CUDA_VISIBLE_DEVICES=2 python main.py --model llava --dataset spurious_imagenet --K 75 --experiment blank --mode twostepv2

wait

CUDA_VISIBLE_DEVICES=0 python main.py --model llama --dataset spurious_imagenet --K 75 --experiment noobject_spur --mode twostepv1 &
CUDA_VISIBLE_DEVICES=1 python main.py --model llama --dataset spurious_imagenet --K 75 --experiment noobject_nospur --mode twostepv1 &
CUDA_VISIBLE_DEVICES=2 python main.py --model llama --dataset spurious_imagenet --K 75 --experiment noobject_nospur --mode twostepv2

wait

CUDA_VISIBLE_DEVICES=0 python main.py --model llama --dataset spurious_imagenet --K 75 --experiment blank --mode unbiased &
CUDA_VISIBLE_DEVICES=1 python main.py --model llama --dataset spurious_imagenet --K 75 --experiment blank --mode twostepv1 &
CUDA_VISIBLE_DEVICES=2 python main.py --model llama --dataset spurious_imagenet --K 75 --experiment blank --mode twostepv2

wait