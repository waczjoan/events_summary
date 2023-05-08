"""Script for creating the events summary."""
import click
from pathlib import Path

import configparser
from eventregistry import *

from events_mod.dataloader.eventregistry import newest_data, latest_events


@click.command()
@click.option(
    "--config_path",
    help="Path to config.",
    type=click.Path(exists=True, path_type=Path),
    default=Path("config/config.local")
)
@click.option(
    "--max_items",
    help="Max articles downloaded from Event Registry.",
    type=int,
    default=10
)
@click.option(
    "--output_dir",
    help="Directory to save data.",
    type=click.Path(path_type=Path),
    default=Path("output")
)
@click.option(
    "--keywords",
    help="Key words using to search articles",
    type=str,
    default="tesla"
)
@click.option(
    "--keywords_loc",
    help="Key words location",
    type=str,
    default="title"
)
def main(
    keywords: str,
    keywords_loc: str,
    max_items: int,
    config_path: Path,
    output_dir: Path
):
    config = configparser.RawConfigParser()
    config.read(config_path)

    details_dict = dict(config.items('eventRegistry'))

    er = EventRegistry(apiKey=details_dict['apikey'])
    #arts = newest_data(
    #    eventregistry=er,
    #    max_items=max_items,
    #    keywords=keywords,
    #    keywords_loc=keywords_loc
    #)

    arts = latest_events(
        topic=keywords,
        eventregistry=er,
        n_items=max_items,
    )
    output_file = Path(os.path.join(
        output_dir, Path(f"{keywords}"),
        Path(f"{keywords_loc}.json")
    ))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w") as f_out:
        json.dump(obj=arts, fp=f_out, indent=4)


if __name__ == "__main__":
    main()
