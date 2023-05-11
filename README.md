# 
Repository created as part of a collaboration between students from Wrocław University of Technology and Event Registry. The task is to create a summary of several texts, assuming that the texts come from the same event. 

For full use of scripts, yous should copy the config file from config/config to config/config.local and complete with your key.
```
[eventRegistry]
    apiKey = yourapiKey
```

### Setup env
```bash
$ python3.9 -m venv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
```

# Used methods

## Bullet point summarization

This methodology involves segmenting the articles into bullet points in order to condense their content. The pipeline can be broken down into two primary steps:

1. Segmentation of the text into individual paragraphs
2. Summarization of each paragraph into a concise bullet point.

Text segmentation into paragraphs can be achieved through either semantic or lexical means. Semantic splitting, while ideal for summarization, can be challenging to implement. With the use of the EventRegistry API, article division can be accomplished through the inclusion of an empty line character (`\n\n`) or HTML tags such as `\<p\>` when scrapping the data. More advanced machine learning techniques require further research and testing, particularly in terms of efficiency.

Lexical splitting, on the other hand, is a more simplistic approach, which can be done with the division of the text into `n` equal chunks of words. A superior alternative would be to divide the text into sentences, although this method necessitates greater effort than separating the text by dots.

Existing pretrained models, like `snrspeaks/t5-one-line-summary` or `anikethdev/t5-summarizer-for-news`, can be efficiently utilized for the summarization component. For multi-lingual applications, the `mT5` model, such as `ctu-aic/mt5-base-multilingual-summarization-multilarge-cs`, should be considered but will need to undergo testing.