import gradio as gr
from datasets import load_dataset
from nltk.util import ngrams
from collections import Counter
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

# Load the dataset and convert it to a Pandas dataframe
sotu_dataset = "jsulz/state-of-the-union-addresses"
dataset = load_dataset(sotu_dataset)
df = dataset["train"].to_pandas()
# decode the tokens-nostop column from a byte array to a list of string
df["tokens-nostop"] = df["tokens-nostop"].apply(
    lambda x: x.decode("utf-8")
    .replace('"', "")
    .replace("[", "")
    .replace("]", "")
    .split(",")
)
df["word_count"] = df["speech_html"].apply(lambda x: len(x.split()))
# calculate the automated readibility index reading ease score for each address
# automated readability index = 4.71 * (characters/words) + 0.5 * (words/sentences) - 21.43
df["ari"] = df["no-contractions"].apply(
    lambda x: (4.71 * (len(x.replace(" ", "")) / len(x.split())))
    + (0.5 * (len(x.split()) / len(x.split("."))))
    - 21.43
)

written = df[df["categories"] == "Written"]
spoken = df[df["categories"] == "Spoken"]

# Create a Gradio interface with blocks
with gr.Blocks() as demo:
    gr.Markdown(
        """
        # A Dashboard to Analyze the State of the Union Addresses
        """
    )
    gr.BarPlot(
        df,
        x="date",
        y="word_count",
        title="Total Number of Words in the Speeches",
        color="categories",
    )
    # group by president and category and calculate the average word count sort by date
    avg_word_count = (
        df.groupby(["date", "potus", "categories"])["word_count"].mean().reset_index()
    )
    # create a bar chart
    gr.BarPlot(
        avg_word_count,
        x="potus",
        y="word_count",
        title="Average Number of Words in the Speeches",
        color="categories",
        x_label_angle=-45,
        height=400,
        min_width=160,
        fill_height=True,
        container=True,
        scale=2,
    )
    with gr.Row():
        ari = df[["potus", "date", "ari", "categories"]]
        gr.LinePlot(
            ari,
            x="date",
            y="ari",
            title="Automated Readability Index",
        )
    # get all unique president names
    presidents = df["potus"].unique()
    # convert presidents to a list
    presidents = presidents.tolist()
    # create a dropdown to select a president
    president = gr.Dropdown(label="Select a President", choices=["All"] + presidents)
    grams = gr.Slider(minimum=1, maximum=4, step=1, label="N-grams", interactive=True)
    with gr.Row():
        # if president is not of type string
        @gr.render(inputs=president)
        def show_text(potus):
            if potus != "All" and potus is not None:
                ari = df[df["potus"] == potus][
                    ["date", "categories", "word_count", "ari"]
                ]
                gr.DataFrame(ari, height=200)

        @gr.render(inputs=president)
        def word_length_bar(potus):
            # calculate the total number of words in the speech_html column and add it to a new column
            # if the president is "All", show the word count for all presidents
            # if the president is not "All", show the word count for the selected president
            if potus != "All" and potus is not None:
                gr.LinePlot(
                    df[df["potus"] == potus],
                    x="date",
                    y="word_count",
                    title="Total Number of Words in the Speeches",
                )

    with gr.Row():

        @gr.render(inputs=[president, grams])
        def ngram_bar(potus, n_grams):
            if potus != "All" and potus is not None:
                if type(n_grams) is not int:
                    n_grams = 1
                print(n_grams)
                # create a Counter object from the trigrams
                potus_df = df[df["potus"] == potus]
                # decode the tokens-nostop column from a byte array to a list of string
                trigrams = (
                    potus_df["tokens-nostop"]
                    .apply(lambda x: list(ngrams(x, n_grams)))
                    .apply(Counter)
                    .sum()
                )
                # get the most common trigrams
                common_trigrams = trigrams.most_common(20)
                # unzip the list of tuples and plot the trigrams and counts as a bar chart
                trigrams, counts = zip(*common_trigrams)
                # join the trigrams into a single string
                trigrams = [" ".join(trigram) for trigram in trigrams]
                # create a dataframe from the trigrams and counts
                trigrams_df = pd.DataFrame({"trigrams": trigrams, "counts": counts})
                # plot the trigrams and counts as a bar chart from matplotlib
                fig, ax = plt.subplots(figsize=(12, 4))
                ax.barh(trigrams_df["trigrams"], trigrams_df["counts"])
                ax.set_title("Top 20 Trigrams")
                ax.set_ylabel("Count")
                ax.set_xlabel("Trigrams")
                plt.xticks(rotation=45)
                # make it tight layout
                plt.tight_layout()
                gr.Plot(value=fig, container=True)


demo.launch()
