import torch
from captum.attr import InputXGradient, Saliency
from typing import Optional, Tuple, Union
from .baseexplainer import BaseExplainer
from .explanation import Explanation
from utils.text_classifier_utils import TextClassifierEngine

class InputXGradientExplainer(BaseExplainer):
    # Class to compute feature importance using InputXGradient or Saliency methods
    NAME = "G"

    def __init__(
        self,
        model,
        tokenizer,
        device="cpu",
        multiply_by_inputs: bool = True,
        **kwargs,
    ):
        """
        Initialize the explainer with the model, tokenizer, and configuration.

        Args:
            model: The model to explain.
            tokenizer: Tokenizer for processing input text.
            device: Device to run computations on ('cpu' or 'cuda').
            multiply_by_inputs: Whether to use InputXGradient (True) or Saliency (False).
            **kwargs: Additional arguments for the base explainer.
        """
        super().__init__(model, tokenizer, **kwargs)

        self.multiply_by_inputs = multiply_by_inputs
        self._device = device

        # Update the explainer name based on the method used
        if self.multiply_by_inputs:
            self.NAME += "xI"
        else:
            self.NAME = 'Saliency'

    def compute_feature_importance(
        self,
        text: Union[str, Tuple[str, str]],
        target: Union[int, str] = 1,
        target_token: Optional[Union[int, str]] = None,
        **kwargs,
    ):
        """
        Compute feature importance scores for the input text.

        Args:
            text: Input text or a tuple of texts.
            target: Target class for which importance is computed.
            target_token: Specific token to target (optional).
            **kwargs: Additional arguments for the attribution method.

        Returns:
            Explanation object containing tokens, scores, and metadata.
        """
        def func(input_embeds):
            """
            Forward function for the model to compute logits.

            Args:
                input_embeds: Input embeddings for the model.

            Returns:
                Logits from the model.
            """
            outputs = self.text_classifier_engine.model(
                inputs_embeds=input_embeds, attention_mask=item["attention_mask"]
            )
            logits = outputs.logits
            return logits

        # Validate and preprocess the target and input text
        target_pos_idx = self.text_classifier_engine.validate_target(target)
        target_token_pos_idx = None
        text = self.text_classifier_engine.validate_input(text)

        # Tokenize the input text and move tensors to the specified device
        item = self._tokenize(text)
        item = {k: v.to(self._device) for k, v in item.items()}
        input_len = item["attention_mask"].sum().item()

        # Select the attribution method based on the configuration
        dl = (
            InputXGradient(func, **self.init_args)
            if self.multiply_by_inputs
            else Saliency(func, **self.init_args)
        )

        # Get input embeddings for the text
        inputs = self.get_input_embeds(text)

        # Compute attributions for the inputs
        attr = dl.attribute(inputs, target=target_pos_idx, **kwargs)
        attr = attr[0, :input_len, :].detach()

        # Pool over the hidden size dimension to get token-level scores
        attr = attr.sum(-1).cpu().numpy()
       
        # Create and return an Explanation object with the results
        output = Explanation(
            text=text,
            tokens=self.get_tokens(text),
            scores=attr,
            explainer=self.NAME,
            target_pos_idx=target_pos_idx,
            target_token_pos_idx=target_token_pos_idx,
            target=self.text_classifier_engine.model.config.id2label[target_pos_idx],
            target_token=None
        )
        return output
