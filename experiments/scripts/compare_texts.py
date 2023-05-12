"""A script to run contextual text comparison on summary and texts."""
import json
import os

from events_mod.models import SimSemRoberta

from pathlib import Path

import click


@click.command()
@click.option(
    "--path_to_summary",
    help="Input directory for summary",
    type=click.Path(exists=True, path_type=Path),
    default=Path("data/input/summaries/sample_summary.txt")
)
@click.option(
    "--input_dir_texts",
    help="Data directory for raw/concatenated texts",
    type=click.Path(exists=True, path_type=Path),
    default=Path("data/input")
)
@click.option(
    "--output_dir",
    help="Directory to save data.",
    type=click.Path(exists=True, path_type=Path),
    default=Path("output")
)
def main(
        path_to_summary: str,
        input_dir_texts: str,
        output_dir: str) -> None:
    """Method running semantic similarity and serializing result."""
    filename = ''
    texts = []
    with open(path_to_summary, 'r') as s:
        summary = s.readlines()[0]
    for text_file in os.listdir(input_dir_texts):
        if 'concat' in text_file:
            with open(os.path.join(input_dir_texts, text_file), 'r') as t:
                texts.extend(list(json.load(t).values()))
            filename = '.'.join([text_file.split('.')[0], 'sim_report.json'])
            break
    model = SimSemRoberta(
        experiment_name='semantic_similarity',
        model_name='semsim_roberta')
    similarities = model.compare_embeddings(
        summary=summary,
        texts=texts)
    result = {'summary': summary,
              'texts': texts,
              'similarities': list(similarities)}

    with open(os.path.join(output_dir, filename), 'w') as f_out:
        json.dump(obj=result, fp=f_out)


if __name__ == '__main__':
    main()
