import matplotlib.pyplot as plt
import pandas as pd
import base64
from io import BytesIO



def plot_data(data, name):
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Calculate accuracy gap and sort by it in descending order
    df['acc_gap'] = df['high_spur_acc'] - df['low_spur_acc']
    df = df.sort_values(by='acc_gap', ascending=False)
    
    # Plotting
    fig, ax = plt.subplots()
    
    # Set width of bar
    bar_width = 0.35
    
    # Define positions of bars on x-axis
    x = range(len(df))
    
    # Plot bars for high spur and low spur accuracy
    ax.bar([p - bar_width/2 for p in x], df['high_spur_acc'], width=bar_width, label='High Spur Acc')
    ax.bar([p + bar_width/2 for p in x], df['low_spur_acc'], width=bar_width, label='Low Spur Acc')
    
    # Add x-ticks and labels
    ax.set_xticks(x)
    ax.set_xticklabels(df['classname'], rotation=45, ha='right')
    ax.set_xlabel('Class Name')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy Comparison (High Spur vs Low Spur)')
    ax.legend()
    
    # Display the plot
    plt.savefig(name)


def encode_image(image):
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    image.save('test.jpg')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    return image_base64


def get_gpt_output(client, prompt, image):
    base64_image = encode_image(image)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                            },
                            ],
                            }
                            ],
                            )
    return response.choices[0].message.content
    