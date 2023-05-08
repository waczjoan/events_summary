"""Script for creating the events summary."""
import click

@click.command()
@click.option(
    "--hparams_path",
    help="Path to selected model hparams",
    type=click.Path(exists=True, path_type=Path),
    default=Path("scripts/config/models.yaml")
)
@click.option(
    "--model",
    help="Name of selected base model",
    type=str,
)
def main(
    hparams_path: Path,
    model: str
) -> Dict[str, Dict[str, float]]:
    """TODO

    """
    with open(hparams_path, "r") as fin:
        hparams = yaml.safe_load(fin)[model]


if __name__ == "__main__":
    main()
