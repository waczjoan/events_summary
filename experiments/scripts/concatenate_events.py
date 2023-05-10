"""Script for preparing the texts concatenation."""

import click
from pathlib import Path
from events_mod.utils import truncate_texts
import os
import json


@click.command()
@click.option(
    "--input_dir",
    help="Input data directory",
    type=click.Path(exists=True, path_type=Path),
    default=Path("output")
)
@click.option(
    "--output_dir",
    help="Data directory for final text output",
    type=click.Path(exists=True, path_type=Path),
    default=Path("data/input")
)
@click.option(
    "--add_eos",
    help="Add EOS tokens at the end of each text",
    type=bool,
    default=True
)
def main(
        input_dir: Path,
        output_dir: Path,
        add_eos: bool
):
    """Process and write data concatenated by keyword."""
    result = {}
    for directory in os.listdir(input_dir):
        for subdirectory in os.listdir(os.path.join(input_dir, directory)):
            for filename in os.listdir(os.path.join(input_dir,
                                                    directory,
                                                    subdirectory)):
                with open(os.path.join(input_dir,
                                       directory,
                                       subdirectory,
                                       filename), 'r') as f_in:
                    texts = []
                    data = json.load(f_in)
                    for data_dict in data:
                        texts.append(data_dict['body'])
                    result[subdirectory] = truncate_texts(texts_batch=texts,
                                                          add_eos_token=add_eos)
    filename = 'example_concat.json'
    if add_eos:
        filename = '_'.join(['sep', filename])
    with open(os.path.join(output_dir, filename), 'w') as f_out:
        json.dump(obj=result, fp=f_out)


if __name__ == '__main__':
    main()
