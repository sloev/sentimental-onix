import os
import onnx
import numpy as np
from onnxruntime import InferenceSession

from sentimental_onix import util

_LABEL_TO_INDEX = {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}
_INDEX_TO_LABEL = {v: k for k, v in _LABEL_TO_INDEX.items()}

ARTIFACTS_FOLDER = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_ZIP_URL = "https://github.com/sloev/sentimental-onix/releases/download/artifacts.en.v1.0.0/sentimental_onix.artifacts.en.zip"

_onnx_model_path = os.path.join(ARTIFACTS_FOLDER, "model.onnx")
_tokenizer_path = os.path.join(ARTIFACTS_FOLDER, "tokenizer.json")


def create_infererence_function(threshold=0.7, **kwargs):
    with open(_tokenizer_path, "r") as handle:
        tokenizer = util.tokenizer_from_json(handle.read())

    onnx_model = onnx.load(_onnx_model_path)

    onnx_session = InferenceSession(onnx_model.SerializeToString())

    def infer(texts):
        tokenized = tokenizer.texts_to_sequences(texts)

        padded_tokenized = util.pad_sequences(tokenized, maxlen=54)

        predictions = onnx_session.run(
            None, {"Tokenized_Sent_Input:0": padded_tokenized}
        )
        predictions = predictions[0]

        results = []
        for pred in predictions:
            if max(pred) < threshold:
                results.append("Neutral")
            else:
                predicted_token = np.argmax(pred)
                predicted_decoded = _INDEX_TO_LABEL[predicted_token]
                results.append(predicted_decoded.lower().capitalize())
        return results

    return infer


def download():
    util.download_and_extract_zip(ARTIFACTS_ZIP_URL, ARTIFACTS_FOLDER)
