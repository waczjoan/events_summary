"""Script for creating the events summary."""
import os
from pathlib import Path
from typing import Dict

import click
import json
import yaml

from events_mod.utils import load_model_from_config
from events_mod.dataloader.summary import load_texts


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
    default="summarizer_for_news"
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

    model = load_model_from_config(cfg=hparams["model"])
    data_to_summary = load_texts(hparams["data_text"])

    for id_key in data_to_summary:
        text = data_to_summary[id_key]
        input_ids = model.tokenize(text)
        generated_ids = model.generate(input_ids)
        decoded_output = model.decode(generated_ids)

        output_file = Path(os.path.join(
            output_dir, Path(model.experiment_name),
            Path(model.model_name),
            Path(f"{id_key}.json")
        ))
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w") as f_out:
            json.dump(obj=decoded_output, fp=f_out, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
