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
    default=Path("data/input")
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
        if os.path.isdir(os.path.join(input_dir,
                                      directory)):
            for filename in os.listdir(os.path.join(input_dir,
                                                    directory)):
                with open(os.path.join(input_dir,
                                       directory,
                                       filename), 'r') as f_in:
                    data = json.load(f_in)
                    for k, v in data.items():
                        result[k] = truncate_texts(texts_batch=v,
                                                   add_eos_token=add_eos)

    filename = 'concat.json'
    if add_eos:
        filename = '_'.join(['sep', filename])
    with open(os.path.join(output_dir, filename), 'w') as f_out:
        json.dump(obj=result, fp=f_out)


if __name__ == '__main__':
    main()
