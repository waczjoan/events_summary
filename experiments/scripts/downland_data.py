"""Script for creating the events summary."""
import click

from eventregistry import *

from events_mod.dataloader.eventregistry import newest_data


@click.command()
@click.option(
    "--api_key",
    help="Api key to Event Registry",
    type=str,
)
@click.option(
    "--max_items",
    help="Max articles downloaded from Event Registry.",
    type=int,
)
def main(
    api_key: str,
    max_items: str
):
    er = EventRegistry(apiKey=api_key)
    newest_data(
        eventregistry=er,
        max_items=max_items
    )


if __name__ == "__main__":
    main()
