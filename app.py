from collections import Counter
import math
import gradio as gr
from datasets import load_dataset
from nltk.util import ngrams
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib import pyplot as plt
from wordcloud import WordCloud
from huggingface_hub import InferenceClient
import matplotlib

matplotlib.use("agg")


def load_transform_dataset():
    # Load the dataset and convert it to a Pandas dataframe
    sotu_dataset = "jsulz/state-of-the-union-addresses"
    dataset = load_dataset(sotu_dataset)
    _df = dataset["train"].to_pandas()
    # Do some on-the-fly calculations
    # calcualte the number of words in each address
    _df["word_count"] = _df["speech_html"].apply(lambda x: len(x.split()))
    # calculate the automated readibility index reading ease score for each address
    # automated readability index = 4.71 * (characters/words) + 0.5 * (words/sentences) - 21.43
    _df["ari"] = _df["no-contractions"].apply(
        lambda x: (4.71 * (len(x.replace(" ", "")) / len(x.split())))
        + (0.5 * (len(x.split()) / len(x.split("."))))
        - 21.43
    )
    # create a column that is the year the speach was given from the date column
    _df["year"] = _df["date"].dt.year
    # create a column that is a concatenation of the president's name, year, and category
    _df["speech_key"] = (
        _df["potus"] + " - " + _df["year"].astype(str) + " (" + _df["categories"] + ")"
    )
    # Sort the dataframe by date because Plotly doesn't do any of this automatically
    _df = _df.sort_values(by="date")
    _written = _df[_df["categories"] == "Written"]
    _spoken = _df[_df["categories"] == "Spoken"]
    return _df, _written, _spoken


"""
Helper functions for Plotly charts
"""


def filter_potus(potus, _df):
    if potus != "All":
        # Filter on the potus
        potus_df = _df[_df["potus"] == potus]
    else:
        potus_df = _df
    return potus_df


def plotly_ngrams(n_grams, potus, _df):
    if potus is not None:
        potus_df = filter_potus(potus, _df)
        # Create a counter generator for the n-grams
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
        if potus == "All":
            potus = "All Presidents"
        fig4 = px.bar(
            trigrams_df,
            x="counts",
            y="trigrams",
            title=f"{potus}'s top {n_grams}-grams",
            orientation="h",
            height=400,
        )
        return fig4


def plotly_word_and_ari(president, _df):
    potus_df = filter_potus(president, _df)
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


def plt_wordcloud(president, _df):
    potus_df = filter_potus(president, _df)
    lemmatized = potus_df["lemmatized"].apply(lambda x: " ".join(x))
    # build a single string from lemmatized
    lemmatized = " ".join(lemmatized)
    # create a wordcloud from the lemmatized column of the dataframe
    wordcloud = WordCloud(background_color="white", width=800, height=400).generate(
        lemmatized
    )
    # create a matplotlib figure
    fig6 = plt.figure(figsize=(8, 4))
    # add the wordcloud to the figure
    plt.tight_layout()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    return fig6


def summarization(speech_key, _df):
    client = InferenceClient(model="facebook/bart-large-cnn")
    chunk_len = 4000
    speech = _df[_df["speech_key"] == speech_key]["speech_html"].values[0]
    sotu_chunks = int(math.ceil(len(speech) / chunk_len))
    response = []
    for chunk in range(1, sotu_chunks + 1):
        if chunk * 4000 < len(speech):
            chunk_text = speech[(chunk - 1) * chunk_len : chunk * chunk_len]
        else:
            chunk_text = speech[(chunk - 1) * chunk_len :]
        try:
            summarization_chunk = client.summarization(
                chunk_text, parameters={"truncation": "do_not_truncate"}
            )
        except Exception as e:
            print(e)
        response.append(summarization_chunk.summary_text)

    return "\n\n".join(response)


# Create a Gradio interface with blocks
with gr.Blocks() as demo:
    df, written, spoken = load_transform_dataset()
    # store the dataframe in a state object before passing to component functions
    df_state = gr.State(df)

    # Build out the top level static charts and content
    gr.Markdown(
        """
        # A Dashboard to Analyze the State of the Union Addresses
        This dashboard provides an analysis of all State of the Union (SOTU) addresses from 1790 to 2020 including written and spoken addresses. The data is sourced from the [State of the Union Addresses dataset](https://huggingface.co/datasets/jsulz/state-of-the-union-addresses) on the Hugging Face Datasets Hub. You can read more about how the data was gathered and cleaned on the dataset card. To read the speeches, you can visit the [The American Presidency Project's State of the Union page](https://www.presidency.ucsb.edu/documents/presidential-documents-archive-guidebook/annual-messages-congress-the-state-the-union) where this data was sourced.
        """
    )
    # Basic line chart showing the total number of words in each address
    with gr.Row():
        gr.Markdown(
            """
                    ## The shape of words
                    The line chart to the right shows the total number of words in each address. However, not all SOTUs are created equally. From 1801 to 1916, each address was a written message to Congress. In 1913, Woodrow Wilson broke with tradition and delivered his address in person. Since then, the addresses have been a mix of written and spoken (mostly spoken). 

                    The spikes you see in the early 1970's and early 1980's are from written addresses by Richard Nixon and Jimmy Carter respectively.

                    Now that we have a little historical context, what does this data look like if we split things out by president? The bar chart below shows the average number of words in each address by president. The bars are grouped by written and spoken addresses.
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
        gr.Plot(fig1, scale=2)
    # group by president and category and calculate the average word count sort by date
    avg_word_count = (
        df.groupby(["potus", "categories"])["word_count"].mean().reset_index()
    )
    # Build a bar chart showing the average number of words in each address by president
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

    # Create a line chart showing the Automated Readability Index in each address
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
        gr.Plot(fig3, scale=2)
        gr.Markdown(
            """
                   The line chart to the left shows the Automated Redibility Index (ARI) for each speech by year. The ARI is calculated using the formula: 4.71 * (characters/words) + 0.5 * (words/sentences) - 21.43. In general, ARI scores correspond to U.S. grade levels. For example, an ARI of 8.0 corresponds to an 8th grade reading level.

                   While there are other scores that are more representative of attributes we might want to measure, they require values like syllables. The ARI is a simple score to compute with our data. 

                   The drop off is quite noticeable, don't you think? ;) 
            """
        )
    gr.Markdown("## Summarize a Speech")
    speeches = df["speech_key"].unique()
    speeches = speeches.tolist()
    speech = gr.Dropdown(label="Select a Speech", choices=speeches)
    # create a dropdown to select a speech from a president
    run_summarization = gr.Button(value="Summarize")
    fin_speech = gr.Textbox(label="Summarized Speech", type="text", lines=10)
    run_summarization.click(
        summarization, inputs=[speech, df_state], outputs=[fin_speech]
    )
    gr.Markdown(
        """
            ## Dive Deeper on Each President

            Use the dropdown to select a president a go a little deeper. 
            
            To begin with, there is an [n-gram](https://en.wikipedia.org/wiki/N-gram) bar chart built from all of the given president's addresses. An n-gram is a contiguous sequence of n items from a given sample of text or speech. Because written and spoken speech is littered with so-called "stop words" such as "and", "the", and "but", they've been removed to provide a more rich (albeit sometimes more difficult to read) view of the text. 
            
            The slider only goes up to 4-grams because the data is sparse beyond that. I personally found the n-grams from our last three presidents to be less than inspiring and full of platitudes. Earlier presidents have more interesting n-grams.

            Next up is a word cloud of the lemmatized text from the president's addresses. [Lemmatization](https://en.wikipedia.org/wiki/Lemmatization) is the process of grouping together the inflected forms of a word so they can be analyzed as a single item. Think of this as a more advanced version of [stemming](https://en.wikipedia.org/wiki/Stemming) where we can establish novel links between words like "better" and "good" that might otherwise be overlooked in stemming.
            
            You can also see a line chart of word count and ARI for each address.
    """
    )
    # get all unique president names
    presidents = df["potus"].unique()
    presidents = presidents.tolist()
    presidents.append("All")

    # create a dropdown to select a president
    president = gr.Dropdown(label="Select a President", choices=presidents, value="All")
    # create a text area to display the summarized speech
    # create a slider for number of word grams
    grams = gr.Slider(
        minimum=1, maximum=4, step=1, label="N-grams", interactive=True, value=1
    )

    # show a bar chart of the top n-grams for a selected president
    gr.Plot(plotly_ngrams, inputs=[grams, president, df_state])

    gr.Plot(plt_wordcloud, scale=2, inputs=[president, df_state])

    # show a line chart of word count and ARI for a selected president
    gr.Plot(plotly_word_and_ari, inputs=[president, df_state])


demo.launch()
