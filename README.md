![spacy syllables](https://raw.githubusercontent.com/sloev/sentimental-onix/master/.github/onix.webp) <a href="https://www.buymeacoffee.com/sloev" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-pink.png" alt="Buy Me A Coffee" height="51px" width="217px"></a>

![example workflow](https://github.com/sloev/sentimental-onix/actions/workflows/test.yml/badge.svg) [![Latest Version](https://img.shields.io/pypi/v/sentimental-onix.svg)](https://pypi.python.org/pypi/sentimental-onix) [![Python Support](https://img.shields.io/pypi/pyversions/sentimental-onix.svg)](https://pypi.python.org/pypi/sentimental-onix)

# Sentimental Onix

Sentiment Analysis using [onnx](https://github.com/onnx/onnx) for python with a focus on being [spacy](https://github.com/explosion/spaCy) compatible *and EEEEEASY to use*.

**Features**
- [x] English sentiment analysis
- [x] Spacy pipeline component
- [x] Sentiment model downloading from github

## Install

```bash
$ pip install sentimental_onix
# download english sentiment model
$ python -m sentimental_onix download en
```

## Usage

```python
import spacy
from sentimental_onix import pipeline

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("sentencizer")
nlp.add_pipe("sentimental_onix", after="sentencizer")

sentences = [
    (sent.text, sent._.sentiment)
    for doc in nlp.pipe(
        [
            "i hate pasta on tuesdays",
            "i like movies on wednesdays",
            "i find your argument ridiculous",
            "soda with straws are my favorite",
        ]
    )
    for sent in doc.sents
]

assert sentences == [
    ("i hate pasta on tuesdays", "Negative"),
    ("i like movies on wednesdays", "Positive"),
    ("i find your argument ridiculous", "Negative"),
    ("soda with straws are my favorite", "Positive"),
]

```

## Benchmark

|         library|   result|
|----------------|---------|
|   spacytextblob|    58.9%|
|sentimental_onix|      69%|
 
See [./benchmark/](./benchmark/) for info

## Dev setup / testing

<details><summary>expand</summary>


### Install

install the dev package and pyenv versions

```bash
$ pip install -e ".[dev]"
$ python -m spacy download en_core_web_sm
$ python -m sentimental_onix download en
```

### Run tests

```bash
$ black .
$ pytest -vvl
```


### Packaging and publishing

```bash
python3 -m pip install --upgrade build twine
python3 -m build
python3 -m twine upload dist/*
```

</details>