from typing import Optional
from spacy.tokens import Span
from spacy.language import Language
from spacy.util import minibatch

import sentimental_onix.inference.en


@Language.factory(
    "sentimental_onix",
    assigns=["span._.sentiment"],
    default_config={"threshold": 0.7, "lang": "en"},
    requires=["doc.sents"],
)
def __sentimental_onix(nlp, name: str, lang: str, threshold):
    return SentimentalOnix(nlp, name, lang, threshold)


class SentimentalOnix:
    def __init__(
        self,
        nlp: Language,
        name: str = "sentimental_onix",
        lang: Optional[str] = "en",
        threshold: float = 0.7,
    ):
        self.name = name
        self.threshold = threshold

        if lang == "en":
            self.infer = sentimental_onix.inference.en.create_infererence_function(
                threshold=threshold
            )
        else:
            raise NotImplementedError(
                f"sentimental_onix has no support for language: {lang}"
            )

        Span.set_extension("sentiment", default=None, force=True)

    def __call__(self, doc):
        sentences = [str(sent) for sent in doc.sents]
        sentiments = self.infer(sentences)
        for sentence, sentiment in zip(doc.sents, sentiments):
            sentence._.set("sentiment", sentiment)
        return doc

    def pipe(self, stream, batch_size=1000):
        for docs in minibatch(stream, size=batch_size):
            sentences = [sent.text.lower() for doc in docs for sent in doc.sents]
            sentiments = self.infer(sentences)
            for doc in docs:
                for sent in doc.sents:
                    sentiment = sentiments.pop(0)
                    sent._.set("sentiment", sentiment)
                yield doc
