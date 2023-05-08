from eventregistry import *


def newest_data(
    max_items: int,
    eventregistry: EventRegistry
):
    """Find news articles that mention Tesla in the article title"""
    q = QueryArticlesIter(
        keywords="tesla",
        keywordsLoc="title"
    )
    for art in q.execQuery(
        eventregistry,
        sortBy="date",
        maxItems=max_items
    ):
        print(art)