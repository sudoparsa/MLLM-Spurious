import os
import re
import shutil


log_path = '/p/vast1/cai6/parsahs/MLLM-Spurious/results/coco/qwen/unbiased-noobject_spur.log'
out_dir = '/p/vast1/cai6/parsahs/MLLM-Spurious/results_images/coco/qwen'



with open(log_path, 'r') as file:
    for line in file:
        if line.startswith('Path: '):
            image_path = line[6:].strip()
            file_name = os.path.basename(image_path)
            print(image_path)
            print(file_name)
            shutil.copy(image_path, os.path.join(out_dir, file_name))


            
