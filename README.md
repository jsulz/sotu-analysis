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
short_description: A data dashboard for all U.S. State of the Unions.
datasets:
  - jsulz/state-of-the-union-addresses
models:
  - Qwen/Qwen2.5-72B-Instruct
---

# State of the Union Analysis

This Space is a Gradio data dashboard for visualizing different aspects of State of the Union addresses over the years.

The data comes from a Hugging Face dataset - [jsulz/state-of-the-union-addresses](https://huggingface.co/datasets/jsulz/state-of-the-union-addresses). To read more about how the data was collected and transformed, visit the [dataset card](https://huggingface.co/datasets/jsulz/state-of-the-union-addresses).

The Space makes use of:

- Gradio
- Plotly (for the charts)
- nltk (to create an n-gram visualization)
- datasets (for loading the dataset from Hugging Face)
- [Quen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) model for summarization
