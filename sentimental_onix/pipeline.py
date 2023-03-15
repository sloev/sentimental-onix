from typing import Optional
from spacy.tokens import Span
from spacy.language import Language
from spacy.util import minibatch
import onnx
import numpy as np
from onnxruntime import InferenceSession

from sentimental_onix import models
from sentimental_onix import util

_label_to_index = {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}
_index_to_label = {v: k for k, v in _label_to_index.items()}


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
            tokenizer_path = models.EN_SENTIMENT_TOKENIZER_PATH
            onnx_model_path = models.EN_SENTIMENT_MODEL_PATH
        else:
            raise NotImplementedError(
                f"sentimental_onix has no support for language: {lang}"
            )

        with open(tokenizer_path, "r") as handle:
            self.tokenizer = util.tokenizer_from_json(handle.read())

        onnx_model = onnx.load(onnx_model_path)

        self.onnx_session = InferenceSession(onnx_model.SerializeToString())

        Span.set_extension("sentiment", default=None, force=True)

    def infer(self, texts):
        tokenized = self.tokenizer.texts_to_sequences(texts)

        padded_tokenized = util.pad_sequences(tokenized, maxlen=54)

        predictions = self.onnx_session.run(
            None, {"Tokenized_Sent_Input:0": padded_tokenized}
        )
        predictions = predictions[0]

        results = []
        for pred in predictions:
            if max(pred) < self.threshold:
                results.append("Neutral")
            else:
                predicted_token = np.argmax(pred)
                predicted_decoded = _index_to_label[predicted_token]
                results.append(predicted_decoded.lower().capitalize())
        return results

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
