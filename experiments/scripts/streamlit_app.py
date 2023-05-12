"""Super easy application/prototype to text summarization.

Run: streamlit run streamlit_app.py.
"""
from pathlib import Path
import re


from annotated_text import annotated_text
import click
import configparser
from eventregistry import EventRegistry
import streamlit as st
import yaml

from events_mod.evaluate.metrics import calc_rouge
from events_mod.utils import (
    load_model_from_config, truncate_texts
)
from events_mod.dataloader.eventregistry import (
    detailed_about_event,
)


def rgb_to_hex(r, g, b):
    """Convert RGB to HEX."""
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)


def calc_rouge_metrics(
    summary_text: str,
    text: str,
    metric: str = 'rougeLsum'
):
    """Calculate rouge metrics and create annotated text."""
    body = []
    sentences = text.split(".")
    score_avg = []
    for sentence in sentences:
        if len(sentence) > 5:
            score = calc_rouge(
                summary_text, sentence, metrics=[metric],
            )
            precision = round(score[metric].precision, 2)
            color = rgb_to_hex(
                round(255 * (1 - precision)),
                255,
                round(255 * (1 - precision)),
            )
            body.append(
                (re.sub('[^a-zA-Z ., \n]+', '', sentence),
                 f'{metric}.precision={precision}', color)
            )
            score_avg.append(precision)

    st.write(f'Rouge.precision.avg: {sum(score_avg) / len(score_avg)}')
    annotated_text(body)

    return sum(score_avg) / len(score_avg)


def create_bullet_points(model_bullet_point_summary, text):
    """Create bullets point to selected text."""
    input_ids = model_bullet_point_summary.tokenize(text)
    generated_ids = model_bullet_point_summary.generate(
        input_ids)
    decoded_output = model_bullet_point_summary.decode(
        generated_ids
    )
    return decoded_output


def create_summary(model, text):
    """Create one line summary."""
    input_ids = model.tokenize(text)
    generated_ids = model.generate(
        input_ids,
        num_return_sequences=1
    )
    decoded_output = model.decode(
        generated_ids
    )
    return decoded_output


def checkbox_and_summary(checkbox_name, model, text):
    """Create checkbox and text summary."""
    checkbox_summary = st.checkbox(checkbox_name)
    summary = [""]
    if checkbox_summary:
        summary = create_summary(
            model, text
        )
        st.write(summary)
    return summary[0]


def text_bullet_points(
    text,
    model_bullet_point_summary,
    text_area_text,
    text_area_placeholder,
    name_checkopoint,
    session_summary,
):
    """Use model to create text bullet points summary."""
    text1 = st.text_area(
        text_area_text, text, placeholder=text_area_placeholder,
    )
    st.write('text len:', len(text1))
    if st.checkbox(name_checkopoint):
        if session_summary is False:
            if len(text1) > 10:
                out = create_bullet_points(model_bullet_point_summary, text1)
                st.write(out.split('\n - '))
                session_summary = True
    else:
        session_summary = False
    return text1, session_summary


@st.cache_resource
def load_model(model_type, hparams):
    """Load model and cache."""
    model = load_model_from_config(
        cfg=hparams[model_type]["model"]
    )
    return model


@st.cache_resource
def load_event_registry(config_path="config/config.local"):
    """To use Appi, config.local should have apikey."""
    config = configparser.RawConfigParser()
    config.read(config_path)

    details_dict = dict(config.items('eventRegistry'))
    er = EventRegistry(apiKey=details_dict['apikey'])
    return er


def take_text(arts, idx):
    """Take articles body to display text."""
    if len(arts) >= idx:
        text = arts[idx - 1]['body']
    else:
        text = ''
    return text


@click.command()
@click.option(
    "--hparams_path",
    help="Path to selected model hparams",
    type=click.Path(exists=True, path_type=Path),
    default=Path("experiments/config/models.yaml")
)
def main(
    hparams_path: Path,
):
    """Streamlit app."""
    st.set_page_config(layout="wide")

    with open(hparams_path, "r") as fin:
        hparams = yaml.safe_load(fin)

    model_summarizer_for_news = load_model('summarizer_for_news', hparams)

    model_bertseq1seq = load_model('bert_summarizer_for_news', hparams)

    model_one_line_summary = load_model('one_line_summary', hparams)

    model_bullet_point_summary = load_model('bullet_point_summary', hparams)

    model_key_phrase_summary = load_model('key_phrase_summary', hparams)

    model_compare = load_model('semantic_similarity_roberta', hparams)

    er = load_event_registry()

    st.header('API')

    concat_text_option = "original selected texts"
    summary_model_input_type_options = [
        concat_text_option, "summary from selected texts"
    ]

    summary_model_types = [
        "t5-summarizer-for-news", "bert2bert_cnn_daily_mail"
    ]

    # settings
    st.session_state.disabled = False
    st.summary_model_input_type = summary_model_input_type_options[0]

    st.session_state.calc_summary_1 = False
    st.session_state.calc_summary_2 = False
    st.session_state.calc_summary_3 = False
    st.session_state.calc_summary_4 = False

    text_input = st.text_input(
        "Enter eventUri 👇",
        disabled=st.session_state.disabled,
        placeholder="eng-8608850",
    )

    checkbox_find_detailed = st.checkbox(
        "Find detailed information about a specific event:"
    )
    if checkbox_find_detailed:
        arts = detailed_about_event(
            eventregistry=er,
            event_id=text_input,
            max_items=4,
            lang='eng'
        )
        st.session_state.cached = True
    else:
        arts = []
        st.session_state.cached = False

    text1 = take_text(arts, 1)
    text2 = take_text(arts, 2)
    text3 = take_text(arts, 3)
    text4 = take_text(arts, 4)

    st.header('Texts')

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        text1, st.session_state.calc_summary_1 = text_bullet_points(
            text=text1,
            model_bullet_point_summary=model_bullet_point_summary,
            text_area_text='1st text',
            text_area_placeholder="1st text",
            name_checkopoint="Summary bullet points 1:",
            session_summary=st.session_state.calc_summary_1
        )

    with col2:

        text2, st.session_state.calc_summary_2 = text_bullet_points(
            text=text2,
            model_bullet_point_summary=model_bullet_point_summary,
            text_area_text='2nd text',
            text_area_placeholder="2nd text",
            name_checkopoint="Summary bullet points 2:",
            session_summary=st.session_state.calc_summary_2

        )

    with col3:
        text3, st.session_state.calc_summary_3 = text_bullet_points(
            text=text3,
            model_bullet_point_summary=model_bullet_point_summary,
            text_area_text='3rd text',
            text_area_placeholder="3rd text",
            name_checkopoint="Summary bullet points 3:",
            session_summary=st.session_state.calc_summary_3

        )

    with col4:
        text4, st.session_state.calc_summary_4 = text_bullet_points(
            text=text4,
            model_bullet_point_summary=model_bullet_point_summary,
            text_area_text='4th text',
            text_area_placeholder="4th text",
            name_checkopoint="Summary bullet points 4:",
            session_summary=st.session_state.calc_summary_4
        )

    st.radio(
        "Which model should be used to summarizing 👉",
        key="summary_model_type",
        options=summary_model_types,
    )
    if st.session_state.summary_model_type == summary_model_types[0]:
        model = model_summarizer_for_news
    elif st.session_state.summary_model_type == summary_model_types[1]:
        model = model_bertseq1seq

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        summary1 = checkbox_and_summary(
            "Summary 1st text:", model, text1
        )

    with col2:

        summary2 = checkbox_and_summary("Summary 2nd text:", model, text2)

    with col3:

        summary3 = checkbox_and_summary("Summary 3rd text:", model, text3)

    with col4:
        summary4 = checkbox_and_summary("Summary 4th text:", model, text4)

    st.header('All texts summarization')

    col1, col2 = st.columns([1, 3])
    with col1:
        st.radio(
            "What should be concatenated 👉",
            key="summary_model_input_type",
            options=summary_model_input_type_options,
        )
    with col2:

        if st.session_state.summary_model_input_type == concat_text_option:
            text_concat = truncate_texts(
                texts_batch=[text1, text2, text3, text4]
            )
        else:
            text_concat = truncate_texts(
                texts_batch=[summary1, summary2, summary3, summary4]
            )

        st.write(text_concat)
    s1 = 0
    s2 = 0
    s3 = 0
    s4 = 0
    decoded_output1 = ''
    col1, col2, col3 = st.columns(3)

    with col1:
        checkbox1 = st.checkbox("Summary:")
        if checkbox1:
            decoded_output = create_summary(
                model, text_concat
            )
            decoded_output1 = str(decoded_output)

            st.write(decoded_output1)
            st.session_state.cached = True
        else:
            st.session_state.cached = False

    with col2:
        if st.checkbox("Model one line summary:"):
            decoded_output = create_summary(
                model_one_line_summary, text_concat
            )
            decoded_output2 = str(decoded_output)
            st.write(decoded_output2)

    with col3:
        if st.checkbox("Key phrase:"):
            decoded_output = create_summary(
                model_key_phrase_summary, text_concat
            )
            decoded_output3 = str(decoded_output)
            st.write(decoded_output3)

    if checkbox1:
        col1, col2, col3, col4 = st.columns(4)

        cosine = model_compare.compare_embeddings(
            summary=decoded_output1,
            texts=[text1, text2, text3, text4])

        with col1:
            st.write(f'cosine_similarity: {cosine[0]}')

            s1 = calc_rouge_metrics(
                decoded_output1,
                text1,
            )

        with col2:
            st.write(f'cosine_similarity: {cosine[1]}')

            s2 = calc_rouge_metrics(
                decoded_output1,
                text2,
            )

        with col3:
            st.write(f'cosine_similarity: {cosine[2]}')

            s3 = calc_rouge_metrics(
                decoded_output1,
                text3,
            )

        with col4:
            st.write(f'cosine_similarity: {cosine[3]}')

            s4 = calc_rouge_metrics(
                decoded_output1,
                text4,
            )

    st.write(f'sum: {s1 + s2 + s3 + s4}')


if __name__ == "__main__":
    main()
