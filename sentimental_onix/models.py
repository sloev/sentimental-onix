import os
from importlib.metadata import version

try:
    __version__ = version(__package__)
except:
    __version__ = "unknown"

__DATA_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

EN_SENTIMENT_MODEL_PATH = os.path.join(__DATA_FOLDER, "en.sentiment.model.onnx")
EN_SENTIMENT_TOKENIZER_PATH = os.path.join(__DATA_FOLDER, "en.sentiment.tokenizer.json")
EN_SENTIMENT_DOWNLOAD_TEMPLATE = f"{__version__}"
