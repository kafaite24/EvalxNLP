import pdb
from typing import Dict, Optional, Text, Union
import logging

import numpy as np
import shap
from shap.maskers import Text as TextMasker

from .baseexplainer import BaseExplainer
from .explanation import Explanation


class SHAPExplainer(BaseExplainer):
    NAME = "SHAP"

    def __init__(
        self,
        model,
        tokenizer,
        silent: bool = True,
        algorithm: str = "partition",
        seed: int = 42,
        **kwargs,
    ):
        """
        Initializes the SHAPExplainer with the given model, tokenizer, and SHAP-specific arguments.
        """
        super().__init__(model, tokenizer, **kwargs)
        # Store SHAP-specific initialization arguments
        self.init_args["silent"] = silent
        self.init_args["algorithm"] = algorithm
        self.init_args["seed"] = seed

    def compute_feature_importance(
        self,
        text,
        target: Union[int, Text] = 1,
        target_token: Optional[Union[int, Text]] = None,
        **kwargs,
    ):
        """
        Computes feature importance scores for the given text using SHAP.

        Args:
            text: Input text to explain.
            target: Target class for which the explanation is generated.
            target_token: Optional specific token to focus on.
            kwargs: Additional arguments for SHAP explainer.

        Returns:
            Explanation object containing token-level importance scores.
        """
        # Validate the target class and input text
        target_pos_idx = self.text_classifier_engine.validate_target(target)
        text = self.text_classifier_engine.validate_input(text)

        # Remove irrelevant 'target_option' argument if passed
        if 'target_option' in kwargs:
            logging.warning("The 'target_option' argument is not used in SHAPExplainer and will be removed.")
            kwargs.pop('target_option')

        # Define a function to compute logits for SHAP explainer
        def func(texts: np.array):
            """
            Runs inference on the input texts and returns the softmax logits.
            """
            _, logits = self.text_classifier_engine.run_inference(texts.tolist())
            # Convert logits to probabilities and return as numpy array
            return logits.softmax(-1).detach().cpu().numpy()

        # Create a SHAP masker for text input
        masker = TextMasker(self.tokenizer)
        # Initialize the SHAP explainer with the model and masker
        explainer_partition = shap.Explainer(model=func, masker=masker, **self.init_args)
        # Compute SHAP values for the input text
        shap_values = explainer_partition(text, **kwargs)
        # Extract SHAP values for the target class
        attr = shap_values.values[0][:, target_pos_idx]

        # Tokenize the input text and prepare for token-level explanation
        item = self._tokenize(text, return_special_tokens_mask=True)
        token_ids = item['input_ids'][0].tolist()
        token_scores = np.zeros_like(token_ids, dtype=float)

        # Assign SHAP values to tokens, ignoring special tokens
        for i, (shap_value, is_special_token) in enumerate(zip(attr, item['special_tokens_mask'][0])):
            if not is_special_token:
                token_scores[i] = shap_value

        # Create an Explanation object with the computed scores
        output = Explanation(
            text=text,
            tokens=self.get_tokens(text),  # Extract tokens from the input text
            scores=token_scores,  # Token-level SHAP scores
            explainer=self.NAME,  # Name of the explainer
            target_pos_idx=target_pos_idx,  # Target class index
            target=self.text_classifier_engine.model.config.id2label[target_pos_idx],  # Target class label
            target_token=None  # No specific target token
        )
        return output
