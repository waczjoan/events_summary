"""Super easy application/prototype to text summarization.

Run: streamlit run streamlit_app.py.
"""
from pathlib import Path

from annotated_text import annotated_text
import click
import streamlit as st
from events_mod.utils import truncate_texts
import yaml

from events_mod.evaluate.metrics import calc_rouge
from events_mod.utils import load_model_from_config


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
    for sentence in sentences:
        score = calc_rouge(
            summary_text, sentence, metrics=[metric],
        )
        precision = round(score[metric].precision, 2)
        color = rgb_to_hex(
            round(255 * (1 - precision)),
            255,
            round(255 * (1 - precision)),
        )
        body.append((sentence, f'{metric}.precision={precision}', color))

    annotated_text(body)


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


@st.cache_resource
def load_model(model_type, hparams):
    """Load model and cache."""
    model = load_model_from_config(
        cfg=hparams[model_type]["model"]
    )
    return model


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

    model = load_model('summarizer_for_news', hparams)

    model_one_line_summary = load_model('one_line_summary', hparams)

    model_bullet_point_summary = load_model('bullet_point_summary', hparams)

    model_key_phrase_summary = load_model('key_phrase_summary', hparams)

    st.header('Texts')

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        text1 = st.text_area('First text', '... input 1st text')
        st.write('1st text len:', len(text1))
        if st.checkbox("Summary 1:"):
            out = create_bullet_points(model_bullet_point_summary, text1)
            st.write(out.split('\n - '))

    with col2:
        text2 = st.text_area('2nd text', '... input 2nd text')
        st.write('2nd text len:', len(text2))
        if st.checkbox("Summary 2:"):
            out = create_bullet_points(model_bullet_point_summary, text2)
            st.write(out.split('\n - '))

    with col3:
        text3 = st.text_area('3rd text', '... input 3rd text')
        st.write('3rd text len:', len(text3))
        if st.checkbox("Summary 3:"):
            out = create_bullet_points(model_bullet_point_summary, text3)
            st.write(out.split('\n - '))

    with col4:
        text4 = st.text_area('4th text', '... input 4th text')
        st.write('4th text len:', len(text4))
        if st.checkbox("Summary 4:"):
            out = create_bullet_points(model_bullet_point_summary, text4)
            st.write(out.split('\n - '))

    st.header('All texts summarization')

    col1, col2, col3 = st.columns(3)

    text_concat = truncate_texts(
        texts_batch=[text1, text2, text3, text4]
    )
    st.write(text_concat)

    with col1:
        checkbox1 = st.checkbox("Summary:")
        if checkbox1:
            decoded_output = create_summary(
                model, text_concat
            )
            decoded_output1 = str(decoded_output)

            st.write(decoded_output1)

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

        with col1:
            calc_rouge_metrics(
                decoded_output1,
                text1,
            )

        with col2:
            calc_rouge_metrics(
                decoded_output1,
                text2,
            )

        with col3:
            calc_rouge_metrics(
                decoded_output1,
                text3,
            )

        with col4:
            calc_rouge_metrics(
                decoded_output1,
                text3,
            )


if __name__ == "__main__":
    main()
