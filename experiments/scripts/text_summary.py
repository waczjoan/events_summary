"""Script for creating the events summary."""
import os
from pathlib import Path
from typing import Dict

import click
import json
import yaml

from events_mod.utils import load_model_from_config


@click.command()
@click.option(
    "--hparams_path",
    help="Path to selected model hparams",
    type=click.Path(exists=True, path_type=Path),
    default=Path("experiments/config/models.yaml")
)
@click.option(
    "--model",
    help="Name of selected base model",
    type=str,
    default="one_line_summary"
)
@click.option(
    "--output_dir",
    help="Directory to save data.",
    type=click.Path(path_type=Path),
    default=Path("output")
)
def main(
    hparams_path: Path,
    model: str,
    output_dir: Path,
) -> Dict[str, Dict[str, float]]:
    with open(hparams_path, "r") as fin:
        hparams = yaml.safe_load(fin)[model]

    abstract = """After a tense and closely watched election season, the results of the 2020 US Presidential Election have been announced. Democratic candidate Joe Biden has been declared the winner, defeating incumbent President Donald Trump. Biden secured the presidency after a hard-fought campaign that saw record voter turnout and widespread controversy over issues such as mail-in ballots, voter fraud, and election interference. 
    Despite Trump's claims of voter fraud and legal challenges in several key swing states,
    Biden was able to secure enough electoral votes to win the election. Biden's win was 
    supported by a diverse coalition of voters, including people of color, women,
    and young people, who turned out in record numbers to support the Democratic ticket. 
    Biden's running mate, Kamala Harris, also made history as the first woman and first 
    person of color to be elected Vice President of the United States
    """

    id_key = "1"
    model = load_model_from_config(cfg=hparams["model"])
    input_ids = model.tokenize(abstract)
    generated_ids = model.generate(input_ids)
    decoded_output = model.decode(generated_ids)

    output_file = Path(os.path.join(
        output_dir, Path(model.experiment_name),
        Path(model.model_name),
        Path(f"{id_key}.json")
    ))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w") as f_out:
        json.dump(obj=decoded_output, fp=f_out, indent=4)


if __name__ == "__main__":
    main()
