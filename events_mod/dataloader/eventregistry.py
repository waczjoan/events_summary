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


def recently_added_data(
    topic: str,
    eventregistry: EventRegistry
):
    """Print a list of recently added articles mentioning the topic"""
    q = QueryArticlesIter(
        conceptUri=eventregistry.getConceptUri(topic)
    )
    for art in q.execQuery(
            eventregistry, sortBy = "date"
    ):
        print(art)


def latest_events(
    topic: str,
    eventregistry: EventRegistry,
    n_items: int,
):
    """Search for latest events related to the topic"""
    q = QueryEvents(
        conceptUri=eventregistry.getConceptUri(topic)
    )
    q.setRequestedResult(
        RequestEventsInfo(
            sortBy="date",
            count=n_items
        )
    )
    print(eventregistry.execQuery(q))
