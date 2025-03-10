#!/bin/bash

#SBATCH --job-name=IN_SYCO                                   # sets the job name
#SBATCH --output=IN_SYCO.out.%j                               # indicates a file to redirect STDOUT to; %j is the jobid. If set, must be set to a file instead of a directory or else submission will fail.
#SBATCH --error=IN_SYCO.out.%j                                # indicates a file to redirect STDERR to; %j is the jobid. If set, must be set to a file instead of a directory or else submission will fail.
#SBATCH --time=01-23:00:00                                          # how long you would like your job to run; format=dd-hh:mm:ss
#SBATCH --account=nexus
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --qos=medium
#SBATCH --mem=32g


cd /fs/nexus-scratch/parsahs/spurious/vlm
pwd

. /usr/share/Modules/init/bash
. /etc/profile.d/ummodules.sh

module add Python3/3.12.7
module add cuda
source env/bin/activate


python --version

nvidia-smi


python main.py --model qwen --dataset imagenet --mode sycophantic --experiment object_spur --K 300

