"""Script for creating the events summary."""
import os
from pathlib import Path
from typing import Dict

import click
import json
import yaml

from events_mod.utils import load_model_from_config, split_text_into_paragraphs
from events_mod.dataloader.summary import load_texts
from typing import Any, List


def model_inference(model: Any, text: str) -> str:
    """Single model full infernce step."""
    input_ids = model.tokenize(text)
    generated_ids = model.generate(input_ids)
    return model.decode(generated_ids)


@click.command()
@click.option(
    "--hparams_path",
    help="Path to selected model hparams",
    type=click.Path(exists=True, path_type=Path),
    default=Path("experiments/config/models.yaml")
)
@click.option(
    "--model_name",
    help="Name of selected base model",
    type=str,
    default="key_phrase_summary"
)
@click.option(
    "--output_dir",
    help="Directory to save data.",
    type=click.Path(path_type=Path),
    default=Path("output")
)
def main(
    hparams_path: Path,
    model_name: str,
    output_dir: Path,
) -> Dict[str, Dict[str, float]]:
    """Creating the events summary using selected model."""
    with open(hparams_path, "r") as fin:
        hparams = yaml.safe_load(fin)[model_name]

    model = load_model_from_config(cfg=hparams["model"])
    data_to_summary = load_texts(hparams["data_text"])

    for id_key in data_to_summary:
        if model_name == "bullet_point_summary":
            text_splits: List[str] = split_text_into_paragraphs(
                data_to_summary[id_key], hparams["split_strategy"]
            )
            decoded_output = [
                f" - {model_inference(model, text_split)[0]}\n"
                for text_split in text_splits
            ]

        else:
            decoded_output = model_inference(model, data_to_summary[id_key])

        output_file = Path(os.path.join(
            output_dir, Path(model.experiment_name),
            Path(model.model_name),
            Path(f"{id_key}.json")
        ))
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w") as f_out:
            json.dump(
                obj=decoded_output, fp=f_out, indent=4, ensure_ascii=False
            )


if __name__ == "__main__":
    main()
