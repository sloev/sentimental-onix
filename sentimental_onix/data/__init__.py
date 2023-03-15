import os
from importlib.metadata import version
__version__ = version(__package__)

__MODELS_FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))

EN_SENTIMENT_MODEL_PATH = os.path.join(__MODELS_FOLDER_PATH, "en.sentiment.model.onnx")
EN_SENTIMENT_TOKENIZER_PATH = os.path.join(__MODELS_FOLDER_PATH, "en.sentiment.tokenizer.json")
EN_SENTIMENT_DOWNLOAD_TEMPLATE = f"{__version__}"
