from .baseexplainer import BaseExplainer
from typing import Optional, Tuple, Union
import torch
from captum.attr import InputXGradient, IntegratedGradients, Saliency
from explanation import Explanation

class IntegratedGradientsExplainer(BaseExplainer):
    NAME = "IG"

    def __init__(
        self,
        model,
        tokenizer,
        device="cpu",
        **kwargs,
    ):
        # Initialize the device (CPU or GPU) and call the parent constructor
        self._device = device
        super().__init__(model, tokenizer, **kwargs)
        # Ensure the model is moved to the specified device
        self.model.to(self._device)

    def _generate_baselines(self, input_len):
        # Generate baseline embeddings for the input length
        # Baselines are typically used as a reference point for attribution
        ids = (
            [self.text_classifier_engine.tokenizer.cls_token_id]  # CLS token
            + [self.text_classifier_engine.tokenizer.pad_token_id] * (input_len - 2)  # Padding tokens
            + [self.text_classifier_engine.tokenizer.sep_token_id]  # SEP token
        )
        # Convert IDs to a tensor and move it to the correct device
        ids_tensor = torch.tensor(ids, device=self._device)
        # Obtain embeddings for the baseline IDs
        embeddings = self.text_classifier_engine._get_embeddings_from_ids(ids_tensor)
        return embeddings.unsqueeze(0)  # Add batch dimension

    def compute_feature_importance(
        self,
        text: Union[str, Tuple[str, str]],
        target: Union[int, str] = 1,
        target_token: Optional[Union[int, str]] = None,
        show_progress: bool = False,
        **kwargs,
    ):
        # Validate the target and input text
        target_pos_idx = self.text_classifier_engine.validate_target(target)
        target_token_pos_idx = None
        text = self.text_classifier_engine.validate_input(text)

        # Define a function to compute logits given input embeddings
        def func(input_embeds):
            # Ensure input embeddings are on the correct device
            input_embeds = input_embeds.to(self._device)
            # Create an attention mask on the same device
            attention_mask = torch.ones(
                *input_embeds.shape[:2], dtype=torch.uint8, device=self._device
            )
            # Perform a forward pass with the input embeddings and attention mask
            _, logits = self.text_classifier_engine._forward_with_input_embeds(
                input_embeds, attention_mask, show_progress=show_progress
            )
            return logits

        # Tokenize the input text and calculate its length
        item = self._tokenize(text)
        input_len = item["attention_mask"].sum().item()

        # Initialize the Integrated Gradients explainer
        dl = IntegratedGradients(
            func, **self.init_args, multiply_by_inputs=False
        )
        # Get input embeddings for the text and move them to the correct device
        inputs = self.get_input_embeds(text).to(self._device)
        # Generate baselines and move them to the correct device
        baselines = self._generate_baselines(input_len).to(self._device)
        # Compute attributions using Integrated Gradients
        attr = dl.attribute(inputs, baselines=baselines, target=target_pos_idx, **kwargs)

        # Process the attribution results
        attr = attr[0, :input_len, :].detach().cpu()  # Move to CPU and detach
        attr = attr.sum(-1).numpy()  # Pool over the hidden size

        # Create an Explanation object to store the results
        output = Explanation(
            text=text,
            tokens=self.get_tokens(text),  # Tokenized representation of the text
            scores=attr,  # Attribution scores
            explainer=self.NAME,  # Explainer name
            target_pos_idx=target_pos_idx,  # Target position index
            target_token_pos_idx=target_token_pos_idx,  # Target token position index
            target=self.text_classifier_engine.model.config.id2label[target_pos_idx],  # Target label
            target_token=None  # Placeholder for target token
        )
        return output