from typing import Callable
import torch
from .baseevaluator import BaseEvaluator

class SparsenessEvaluator(BaseEvaluator):
    """Implementation of the sparseness metric.

    Computes the sparseness of the model based on the Gini index.

    Attributes:
        model (callable): model to explain.
    """
    NAME = "Sparseness ↑"

    def __init__(self, model, tokenizer, device="cpu"):
        """Initializes SparsenessEvaluator with the model, tokenizer, and device.

        Args:
            model (callable): The model to evaluate.
            tokenizer: Tokenizer used for preprocessing.
            device (str): Device to run computations on (e.g., "cpu" or "cuda").
        """
        self.model = model
        self.tokenizer = tokenizer
        self._device = device
        self.requires_human_rationale = False

    def get_sparsity(self, attribution: torch.Tensor) -> torch.Tensor:
        """Computes the sparseness of the model's attributions.

        Sparseness is calculated using a weighted sum of absolute attributions
        and normalized to produce a score between 0 and 1.

        Args:
            attribution (torch.Tensor): Attributions for each instance.

        Returns:
            torch.Tensor: Average sparseness score across all instances.
        """
        n_instances, n_features = attribution.shape

        # Compute the absolute value of attributions
        abs_attrib = torch.abs(attribution)

        # Calculate the total sum of absolute attributions for each instance
        attrib_sum = abs_attrib.sum(dim=1)

        # Compute the weighted sum for each instance
        weighted_sum = torch.zeros(n_instances).to(self._device)
        for j in range(n_features):
            # Weight decreases as feature index increases
            weighted_sum += (n_features - j + 0.5) * abs_attrib[:, j]

        # Normalize to obtain sparseness score
        sparsity = 1 - 2 * weighted_sum / (attrib_sum * n_features + 1e-8)
        
        # Return the mean sparseness score across all instances
        return sparsity.mean()

    def compute(self, explanations) -> dict:
        """Compute the sparseness metric for the model.

        This method processes the attributions from the explanations, pads them
        to the same length, sorts them by absolute value, and computes the sparseness.

        Args:
            explanations: List of explanation objects containing attributions.

        Returns:
            dict: Dictionary containing the sparseness metric.
        """
        # Ensure explanations is a list
        explanations = explanations if isinstance(explanations, list) else [explanations]

        # Extract attributions (importance scores) from each Explanation object
        attributions = [torch.tensor(explanation.scores).to(self._device) for explanation in explanations]
        
        # Find the maximum length of the attributions
        max_len = max(len(attr) for attr in attributions)
        
        # Pad the attributions to the same length
        padded_attributions = [torch.nn.functional.pad(attr, (0, max_len - len(attr))) for attr in attributions]
        
        # Stack padded attributions into a single tensor
        attributions = torch.stack(padded_attributions)

        # Sort attributions by absolute value (ascending order)
        sorted_attrib = torch.sort(torch.abs(attributions), dim=1)[0]
        
        # Compute the sparseness score
        spars_score = self.get_sparsity(sorted_attrib)
        
        # Store the sparseness score as a scalar value in the result dictionary
        result = spars_score.item()
        return result