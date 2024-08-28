import gradio as gr
from datasets import load_dataset
from nltk.util import ngrams
from collections import Counter

# Load the dataset and convert it to a Pandas dataframe
sotu_dataset = 'jsulz/state-of-the-union-addresses'
dataset = load_dataset(sotu_dataset)
df = dataset['train'].to_pandas()
df['word_count'] = df['speech_html'].apply(lambda x: len(x.split()))
written = df[df['categories'] == 'Written']
spoken = df[df['categories'] == 'Spoken']

# Create a Gradio interface with blocks
with  gr.Blocks() as demo:
    gr.Markdown(
        """
        # A Dashboard to Analyze the State of the Union Addresses
        """)
    # get all unique president names
    presidents = df['potus'].unique()
    # convert presidents to a list
    presidents = presidents.tolist()
    # create a dropdown to select a president
    president = gr.Dropdown(label="Select a President", choices=["All"] + presidents)
    with gr.Row():
        # if president is not of type string
        @gr.render(inputs=president)
        def show_text(potus):
            if potus is not None:
                gr.Markdown(f"{potus} was the first president of the United States.")
        
        @gr.render(inputs=president)
        def word_length_bar(potus):
            # calculate the total number of words in the speech_html column and add it to a new column
            # if the president is "All", show the word count for all presidents
            if potus == "All":
                gr.BarPlot(df, x="date", y="word_count", title="Total Number of Words in the Speeches")
            else:
                # if the president is not "All", show the word count for the selected president
                gr.BarPlot(df[df['potus'] == potus], x="date", y="word_count", title="Total Number of Words in the Speeches")
    with gr.Row():

        @gr.render(inputs=president)
        def ngram_bar(potus):
            # create a Counter object from the trigrams
            potus_df = df[df["potus"] == potus]
            trigrams = (
                potus_df["tokens-nostop"].apply(lambda x: list(ngrams(x, 3))).apply(Counter).sum()
            )
            # get the most common trigrams
            common_trigrams = trigrams.most_common(20)
            # unzip the list of tuples and plot the trigrams and counts as a bar chart
            trigrams, counts = zip(*common_trigrams)
            # join the trigrams into a single string
            trigrams = [" ".join(trigram) for trigram in trigrams]
            # create a dataframe from the trigrams and counts
            trigrams_df = pd.DataFrame({"trigrams": trigrams, "counts": counts})
            # plot the trigrams and counts as a bar chart
            gr.BarPlot(trigrams_df, x="trigrams", y="counts", title="Most Common Trigrams")

demo.launch()
