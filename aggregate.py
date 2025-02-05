import os
import pandas as pd

dataset_name = 'spurious_imagenet'

# model_name = 'qwen'
# model_name = 'llava'
model_name = 'llama'


# experiment = 'noobject_spur'
experiment = 'blank'
# experiment = 'noobject_nospur'


# mode = "twostepv1"
mode = 'twostepv2'
# mode = "unbiased"

dir_path = os.path.join("log", dataset_name, model_name, experiment, mode)
# dir_path = os.path.join("log", dataset_name, model_name, experiment)


aggregated_df = None
total = []
for file_name in sorted(os.listdir(dir_path)):
    if file_name.endswith('.csv'):
        file_path = os.path.join(dir_path, file_name)
        df = pd.read_csv(file_path)
        print(f"{file_name} Total={df['total'].iloc[-1]}")
        total.append(df['total'].iloc[-1])
        df = df.drop(columns=['total', 'prompt', 'target'], errors='ignore')        
        if aggregated_df is None:
            aggregated_df = df
        else:
            aggregated_df = pd.concat([aggregated_df, df], axis=1)

avg = sum(total) / len(total)
print(f"Avg: {avg}")
aggregated_df['total'] = [avg, avg]
output_file = os.path.join(dir_path, 'agg.csv')
if os.path.exists(output_file):
    os.remove(output_file)
if aggregated_df is not None:
    aggregated_df.to_csv(output_file, index=False)
    print(f"Aggregated file saved to {output_file}")
else:
    print("No CSV files found in the folder.")