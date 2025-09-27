# Note
## One sample look like
### Twitter Dataset
```
{
    "words": [      "And",      "this",      "is",      "proof",      "that",      "the",      "US",      "is",      "behind",
      "in",      "fashion",      ",",      "I",      "bought",      "this",      "sweater",      "in",      "Europe",      "last",
      "winter",      "(",      "except",      "in",      "grey",      ")"
    ],
    "image_id": "15649.jpg",
    "aspects": [
      { "from": 6, "to": 7, "polarity": "NEG", "term": ["US"] },
      { "from": 17, "to": 18, "polarity": "POS", "term": ["Europe"] }
    ],
    "opinions": [{ "term": [] }],
    "caption": "im not a fan of the black t  shirt but i love the black",
    "image_path": "./data/twitter2015_images/15649.jpg",
    "aspects_num": 2
}
```

## Difference
- `term`: Term of Twitter Dataset are very various while Hotel Dataset may be use a pair of `(aspect_name, aspect_term)`. E.g.: `("service", "Dang Thanh An")`.
- In Hotel Dataset, `sentiment` and `aspect` will share a `term` (or `aspect_term` = `sentiment_term`). 
- Twitter Dataset has only 1 image while Hotel Dataset have multi-images or nothing. -> Need to find a way to concanate them
- Twitter Dataset has the location of aspect
