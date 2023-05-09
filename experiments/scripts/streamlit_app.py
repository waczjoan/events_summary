# streamlit run streamlit_app.py
import streamlit as st
from typing import Dict

import click
import json
import yaml

from events_mod.utils import load_model_from_config
from events_mod.dataloader.summary import load_texts

st.set_page_config(layout="wide")

hparams_path = "experiments/config/models.yaml"
with open(hparams_path, "r") as fin:
    hparams = yaml.safe_load(fin)

model = load_model_from_config(cfg=hparams['summarizer_for_news']["model"])
model_one_line_summary = load_model_from_config(
    cfg=hparams['one_line_summary']["model"]
)
model_key_phrase_summary = load_model_from_config(
    cfg=hparams['key_phrase_summary']["model"]
)


col1, col2, col3, col4 = st.columns(4)

with col1:
    text1 = st.text_area('First text', '... input 1st text')
    st.write('The current movie title is', text1[:30])

with col2:
    text2 = st.text_area('2nd text', '... input 2nd text')
    st.write('The current movie title is', text2)

with col3:
    text3 = st.text_area('3rd text', '... input 3rd text')
    st.write('The current movie title is', text3)


with col4:
    text4 = st.text_area('4th text', '... input 4th text')
    st.write('The current movie title is', text4)


decoded_output1 = "..."
decoded_output2 = "..."
decoded_output3 = "..."

col1, col2, col3 = st.columns(3)

with col1:
    if st.checkbox("Summary:"):

        text = text1
        input_ids = model.tokenize(text)
        generated_ids = model.generate(
            input_ids,
            num_return_sequences=1
        )
        decoded_output = model.decode(generated_ids)
        decoded_output1 = str(decoded_output)

    st.write(decoded_output1)

with col2:
    if st.checkbox("Model one line summary:"):

        text = text1
        input_ids = model_one_line_summary.tokenize(text)
        generated_ids = model_one_line_summary.generate(
            input_ids,
            num_return_sequences=1
        )
        decoded_output2 = model_one_line_summary.decode(generated_ids)
    st.write(decoded_output2)

with col3:
    if st.checkbox("Key phrase:"):

        text = text1
        input_ids = model_key_phrase_summary.tokenize(text)
        generated_ids = model_key_phrase_summary.generate(input_ids)
        decoded_output3 = model_key_phrase_summary.decode(generated_ids)
    st.write(decoded_output3)
