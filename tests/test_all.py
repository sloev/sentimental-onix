import spacy
from sentimental_onix import pipeline


def test_simple_english():
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
