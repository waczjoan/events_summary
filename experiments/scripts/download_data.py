"""Script for creating the events summary."""
import click
import json
import os
from pathlib import Path

import configparser
from eventregistry import EventRegistry

from events_mod.dataloader.eventregistry import (
    newest_data, latest_events, recently_added_data
)


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
@click.option(
    "--api_type",
    help="Type to api endpoint",
    type=str,
    default="recently_added"
)
def main(
    keywords: str,
    keywords_loc: str,
    max_items: int,
    config_path: Path,
    output_dir: Path,
    api_type: str
):
    """Download data from Event Registry."""
    config = configparser.RawConfigParser()
    config.read(config_path)

    details_dict = dict(config.items('eventRegistry'))
    er = EventRegistry(apiKey=details_dict['apikey'])

    if api_type == "newest_data":
        arts = newest_data(
            eventregistry=er,
            max_items=max_items,
            keywords=keywords,
            keywords_loc=keywords_loc
        )
    elif api_type == "latest_events":
        arts = latest_events(
            topic=keywords,
            eventregistry=er,
            n_items=max_items,
        )
    elif api_type == "recently_added":
        arts = recently_added_data(
            eventregistry=er,
            topic=keywords,
            max_items=max_items
        )
    output_file = Path(os.path.join(
        output_dir,
        Path(api_type),
        Path(keywords),
        Path(f"{keywords_loc}.json")
    ))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w") as f_out:
        json.dump(obj=arts, fp=f_out, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
