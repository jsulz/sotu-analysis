import gradio as gr
from datasets import load_dataset

# Load the dataset and convert it to a Pandas dataframe
sotu_dataset = 'jsulz/state-of-the-union-addresses'
dataset = load_dataset(sotu_dataset)
df = dataset['train'].to_pandas()

print(df.head(10))

def greet(name):
    return "Hello " + name + ", you're cool!!"

# Create a Gradio interface with blocks
with  gr.Blocks() as demo:
    with gr.Row():
        gr.Markdown("# A Dashboard to Analyze the State of the Union Addresses")
        demo = gr.Interface(fn=greet, inputs="text", outputs="text")

demo.launch()
