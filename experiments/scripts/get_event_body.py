"""Script for extracting texts classified as belonging to the same event."""

from typing import List, Dict
import warnings
import click
from pathlib import Path
import os
import json
import time


def extract_texts(input_dir: Path,
                  api_dir: str,
                  subdir: str) -> Dict[str, List[str]]:
    """Function parsing text into a dictionary."""
    result = {}
    for filename in os.listdir(os.path.join(input_dir,
                                            api_dir,
                                            subdir)):
        with open(os.path.join(input_dir,
                               api_dir,
                               subdir,
                               filename), 'r') as f_in:
            texts = []
            data = json.load(f_in)
            for data_dict in data:
                texts.append(data_dict['body'])
            result[subdir] = texts
    return result


@click.command()
@click.option(
    '--input_dir',
    help='Directory with input data',
    type=click.Path(exists=True, path_type=Path),
    default=Path("output")
)
@click.option(
    '--output_dir',
    help='Directory to write the texts to',
    type=click.Path(exists=True, path_type=Path),
    default='data/input'
)
@click.option(
    '--event_ids',
    help='Provide list of event types for news filtering',
    default=None
)
def main(
        input_dir: Path,
        output_dir: Path,
        event_ids: List[str] = None
):
    """Process and write data with or without event selection."""
    texts_dict = {}
    non_existent = []
    for directory in os.listdir(input_dir):
        for subdirectory in os.listdir(os.path.join(input_dir, directory)):
            if event_ids:
                for event_id in event_ids:
                    if subdirectory == event_id:
                        dict_tmp = extract_texts(input_dir=input_dir,
                                                 api_dir=directory,
                                                 subdir=event_id)
                        texts_dict = {**texts_dict, **dict_tmp}
                    else:
                        non_existent.append(event_id)
            else:
                texts_dict = {**texts_dict,
                              **extract_texts(input_dir=input_dir,
                                              api_dir=directory,
                                              subdir=subdirectory)}
    assert len(texts_dict) > 0, "No events of the selected types were found"
    if len(non_existent) > 0:
        warnings.warn(f'The following events were not found:\n{non_existent}')
    filename = f'events.{time.time()}.json'
    with open(os.path.join(output_dir, 'events', filename), 'w') as f_out:
        json.dump(obj=texts_dict, fp=f_out)


if __name__ == "__main__":
    main()
