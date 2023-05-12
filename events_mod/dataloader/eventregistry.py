"""Methods that allow to obtain articles or events from Event Registry."""
from eventregistry import (
    EventRegistry,
    QueryArticlesIter,
    QueryEvents,
    RequestEventsInfo,
    QueryEventArticlesIter
)


def newest_data(
    max_items: int,
    eventregistry: EventRegistry,
    keywords_loc: str,
    keywords: str
):
    """Find news articles that mention Tesla in the article title."""
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
    max_items: int,
    eventregistry: EventRegistry,
    lang: str = 'eng'
):
    """Print a list of recently added articles mentioning the topic."""
    q = QueryArticlesIter(
        conceptUri=eventregistry.getConceptUri(topic), lang=lang,
    )
    arts = []
    for art in q.execQuery(
            eventregistry,
            sortBy="date",
            maxItems=max_items
    ):
        arts.append(art)
    return arts


def latest_events(
    topic: str,
    eventregistry: EventRegistry,
    n_items: int,
):
    """Search for latest events related to the topic."""
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


def detailed_about_event(
    eventregistry: EventRegistry,
    lang: str,
    event_id: str,
    max_items: int
):
    """Find detailed information about a specific event."""
    arts = []
    iter = QueryEventArticlesIter(
        event_id, lang=lang
    )
    q = iter.execQuery(
        eventregistry,
        sortBy="sourceImportance",
        maxItems=max_items
    )
    for art in q:
        arts.append(art)
    return arts
