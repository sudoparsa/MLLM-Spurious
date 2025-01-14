#!/bin/bash

#SBATCH --job-name=COCO                                   # sets the job name
#SBATCH --output=COCO.out.%j                               # indicates a file to redirect STDOUT to; %j is the jobid. If set, must be set to a file instead of a directory or else submission will fail.
#SBATCH --error=COCO.out.%j                                # indicates a file to redirect STDERR to; %j is the jobid. If set, must be set to a file instead of a directory or else submission will fail.
#SBATCH --time=23:59:00                                          # how long you would like your job to run; format=dd-hh:mm:ss
#SBATCH --account=scavenger
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --mem=32g
#SBATCH --ntasks=1                                                   # cpu cores be reserved for your node total
#SBATCH --partition=scavenger


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

python main.py --model qwen --dataset coco --mode unbiased --experiment noobject_spur --K 500
python main.py --model llava --dataset coco --mode unbiased --experiment noobject_spur --K 500
python main.py --model llama --dataset coco --mode unbiased --experiment noobject_spur --K 500

