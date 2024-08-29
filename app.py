import gradio as gr
from datasets import load_dataset
from nltk.util import ngrams
from collections import Counter
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

# Load the dataset and convert it to a Pandas dataframe
sotu_dataset = "jsulz/state-of-the-union-addresses"
dataset = load_dataset(sotu_dataset)
df = dataset["train"].to_pandas()
# decode the tokens-nostop column from a byte array to a list of string
"""
df["tokens-nostop"] = df["tokens-nostop"].apply(
    lambda x: x.decode("utf-8")
    .replace('"', "")
    .replace("[", "")
    .replace("]", "")
    .split(",")
)
"""
df["word_count"] = df["speech_html"].apply(lambda x: len(x.split()))
# calculate the automated readibility index reading ease score for each address
# automated readability index = 4.71 * (characters/words) + 0.5 * (words/sentences) - 21.43
df["ari"] = df["no-contractions"].apply(
    lambda x: (4.71 * (len(x.replace(" ", "")) / len(x.split())))
    + (0.5 * (len(x.split()) / len(x.split("."))))
    - 21.43
)
df = df.sort_values(by="date")
written = df[df["categories"] == "Written"]
spoken = df[df["categories"] == "Spoken"]

# Create a Gradio interface with blocks
with gr.Blocks() as demo:
    gr.Markdown(
        """
        # A Dashboard to Analyze the State of the Union Addresses
        """
    )
    fig1 = px.line(
        df,
        x="date",
        y="word_count",
        title="Total Number of Words in Addresses",
        line_shape="spline",
    )
    fig1.update_layout(
        xaxis=dict(title="Date of Address"),
        yaxis=dict(title="Word Count"),
    )
    gr.Plot(fig1)
    # group by president and category and calculate the average word count sort by date
    avg_word_count = (
        df.groupby(["potus", "categories"])["word_count"].mean().reset_index()
    )
    fig2 = px.bar(
        avg_word_count,
        x="potus",
        y="word_count",
        title="Average Number of Words in Addresses by President",
        color="categories",
        barmode="group",
    )
    fig2.update_layout(
        xaxis=dict(
            title="President",
            tickangle=-45,  # Rotate labels 45 degrees counterclockwise
        ),
        yaxis=dict(
            title="Average Word Count",
            tickangle=0,  # Default label angle (horizontal)
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    gr.Plot(fig2)
    with gr.Row():
        ari = df[["potus", "date", "ari", "categories"]]
        fig3 = px.line(
            ari,
            x="date",
            y="ari",
            title="Automated Readability Index in each Address",
            line_shape="spline",
        )
        fig3.update_layout(
            xaxis=dict(title="Date of Address"),
            yaxis=dict(title="ARI Score"),
        )
        gr.Plot(fig3)
    # get all unique president names
    presidents = df["potus"].unique()
    # convert presidents to a list
    presidents = presidents.tolist()
    # create a dropdown to select a president
    president = gr.Dropdown(label="Select a President", choices=presidents)
    grams = gr.Slider(minimum=1, maximum=4, step=1, label="N-grams", interactive=True)

    def plotly_bar(n_grams, potus):
        if potus is not None:
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
            common_trigrams = trigrams.most_common(10)
            # unzip the list of tuples and plot the trigrams and counts as a bar chart
            trigrams, counts = zip(*common_trigrams)
            # join the trigrams into a single string
            trigrams = [" ".join(trigram) for trigram in trigrams]
            # create a dataframe from the trigrams and counts
            trigrams_df = pd.DataFrame({"trigrams": trigrams, "counts": counts})
            fig4 = px.bar(
                trigrams_df,
                x="counts",
                y="trigrams",
                title=f"{potus}'s top {n_grams}-grams",
                orientation="h",
                height=400,
            )
            return fig4

    if president != "All" and president is not None:
        gr.Plot(plotly_bar, inputs=[grams, president])

    def plotly_line(president):
        if president != "All" and president is not None:
            potus_df = df[df["potus"] == president]
            fig5 = make_subplots(specs=[[{"secondary_y": True}]])
            fig5.add_trace(
                go.Scatter(
                    x=potus_df["date"],
                    y=potus_df["word_count"],
                    name="Word Count",
                ),
                secondary_y=False,
            )
            fig5.add_trace(
                go.Scatter(
                    x=potus_df["date"],
                    y=potus_df["ari"],
                    name="ARI",
                ),
                secondary_y=True,
            )
            # Add figure title
            fig5.update_layout(title_text="Address Word Count and ARI")

            # Set x-axis title
            fig5.update_xaxes(title_text="Date of Address")

            # Set y-axes titles
            fig5.update_yaxes(title_text="Word Count", secondary_y=False)
            fig5.update_yaxes(title_text="ARI", secondary_y=True)
            return fig5

    # calculate the total number of words in the speech_html column and add it to a new column
    # if the president is "All", show the word count for all presidents
    # if the president is not "All", show the word count for the selected president
    if president != "All" and president is not None:
        gr.Plot(plotly_line, inputs=[president])


demo.launch(share=True)
