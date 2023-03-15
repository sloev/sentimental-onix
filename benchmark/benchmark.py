import csv
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
from sentimental_onix import pipeline
import json
import collections

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("sentencizer")
nlp.add_pipe("spacytextblob")
nlp.add_pipe("sentimental_onix")


output = {"sentences": [], "results": {"textblob": 0, "sentimental_onix": 0}}


results = collections.Counter()

with open("dataset.csv", "r") as datasetfile:
    csv_reader = csv.DictReader(datasetfile)
    for row in csv_reader:
        doc = nlp(row["sentence"])
        for sent in doc.sents:
            true_sentiment = row["sentiment"]
            textblob_polarity = sent._.blob.polarity
            textblob_sentiment = (
                "negative"
                if textblob_polarity < -0.05
                else ("positive" if textblob_polarity > 0.05 else "neutral")
            )
            sentimental_onix_sentiment = sent._.sentiment.lower()

            results.update(
                {
                    "total": 1,
                    "textblob": true_sentiment == textblob_sentiment,
                    "sentimental_onix": true_sentiment == sentimental_onix_sentiment,
                }
            )


for k,v in results.items():
    print(k, v/results["total"], "%")
