---
title: State of the Union Analysis
emoji: 📊
colorFrom: purple
colorTo: yellow
sdk: gradio
sdk_version: 4.42.0
app_file: app.py
pinned: false
license: mit
short_description: A set of data visualizations for all recorded U.S. State of the Union speeches and messages.
datasets: jsulz/state-of-the-union-addresses
---

# State of the Union Analysis

This Space is a Gradio data dashboard for visualizing different aspects of State of the Union addresses over the years.

The data comes from a Hugging Face dataset - [jsulz/state-of-the-union-addresses](https://huggingface.co/datasets/jsulz/state-of-the-union-addresses). To read more about how the data was collected and transformed, visit the [dataset card](https://huggingface.co/datasets/jsulz/state-of-the-union-addresses).

The Space makes use of:

- Gradio
- Plotly (for the charts)
- nltk (to create an n-gram visualization)
- datasets (for loading the dataset from Hugging Face)
