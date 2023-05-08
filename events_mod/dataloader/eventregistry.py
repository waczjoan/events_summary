from eventregistry import *


def newest_data(
    max_items: int,
    eventregistry: EventRegistry,
    keywords_loc: str,
    keywords: str
):
    """Find news articles that mention Tesla in the article title"""
    q = QueryArticlesIter(
        keywords=keywords,
        keywordsLoc=keywords_loc
    )
    exec_query = q.execQuery(
        eventregistry,
        sortBy="date",
        maxItems=max_items
    )
    arts = []
    for art in exec_query:
        arts.append(art)
    return arts


def recently_added_data(
    topic: str,
    eventregistry: EventRegistry
):
    """Print a list of recently added articles mentioning the topic"""
    q = QueryArticlesIter(
        conceptUri=eventregistry.getConceptUri(topic)
    )
    arts = []
    for art in q.execQuery(
            eventregistry, sortBy="date"
    ):
        arts.append(art)
    return arts


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
    return eventregistry.execQuery(q)
