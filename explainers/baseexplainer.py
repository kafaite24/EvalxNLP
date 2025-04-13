"""Explainers API"""

import warnings
from abc import ABC, abstractmethod
from utils.text_classifier_utils import TextClassifierEngine
from typing import Optional, Tuple, Union, Dict, List
from utils.generate_heatmap import generate_explanation_heatmap
from utils.saliency_utils import lp_normalize

# Base class for all explainers, enforcing a consistent API
class BaseExplainer(ABC):
    # Abstract property to define the name of the explainer
    @property
    @abstractmethod
    def NAME(self):
        pass

    # Constructor to initialize the explainer with a model and tokenizer
    def __init__(
        self, model, tokenizer, **kwargs
    ):
        if model is None or tokenizer is None:
            raise ValueError("Please specify a model and a tokenizer.")

        # Store additional arguments for initialization
        self.init_args = kwargs

        # Initialize the text classifier engine with the provided model and tokenizer
        self.text_classifier_engine = TextClassifierEngine(model, tokenizer)

    # Property to get the device (e.g., CPU or GPU) of the model
    @property
    def device(self):
        return self.text_classifier_engine.model.device

    # Property to access the model
    @property
    def model(self):
        return self.text_classifier_engine.model

    # Property to access the tokenizer
    @property
    def tokenizer(self):
        return self.text_classifier_engine.tokenizer

    # Tokenize the input text using the text classifier engine
    def _tokenize(self, text, **tok_kwargs):
        return self.text_classifier_engine._tokenize(text, **tok_kwargs)

    # Split the input text into tokens
    def get_tokens(self, text):
        return self.text_classifier_engine.split_input_tokens(text)

    # Extract input embeddings for the given text
    def get_input_embeds(self, text):
        return self.text_classifier_engine.extract_input_embeddings(text)

    # Abstract method to compute feature importance, to be implemented by subclasses
    @abstractmethod
    def compute_feature_importance(
        self, text: str, target: int, target_token: Optional[str], **explainer_args
    ):
        pass

    # Callable interface to compute feature importance
    def __call__(
        self,
        text: str,
        target: Union[str, int],
        target_token: Optional[str] = None,
        **explainer_args
    ):
        return self.compute_feature_importance(
            text, target, target_token, **explainer_args
        )
