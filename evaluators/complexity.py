import torch
import numpy as np
import pandas as pd
from .baseevaluator import BaseEvaluator

class ComplexityEvaluator(BaseEvaluator):
    """
    Class to evaluate the complexity of token-level attributions for text data.

    Computes the entropy of the fractional contribution of each token
    to measure the complexity of the attributions.

    References:
        - `Evaluating and Aggregating Feature-based Model Explanations
        <https://arxiv.org/abs/2005.00631>`
    """
    NAME = "Complexity ↓"

    def __init__(self, model, tokenizer, device="cpu"):
        """
        Initialize the evaluator.

        Args:
            model: The trained text classification model.
            tokenizer: Tokenizer used for the model.
            device: The device to use for computations ("cuda" or "cpu").
        """
        self.model = model
        self.tokenizer = tokenizer
        self._device = device
        self.requires_human_rationale = False  # Indicates if human rationale is required for evaluation

    def _total_contribution(self, attributions: torch.Tensor) -> torch.Tensor:
        """
        Compute the total contribution for each instance.

        Args:
            attributions (torch.Tensor): Saliency scores (batch_size, seq_len).

        Returns:
            torch.Tensor: Total contribution for each instance (batch_size,).
        """
        # Sum of absolute values of attributions across all tokens
        return torch.sum(torch.abs(attributions), dim=1)

    def _fractional_contribution(
        self, attributions: torch.Tensor, feature_index: int
    ) -> torch.Tensor:
        """
        Compute the fractional contribution of a specific feature.

        Args:
            attributions (torch.Tensor): Saliency scores (batch_size, seq_len).
            feature_index (int): Index of the feature.

        Returns:
            torch.Tensor: Fractional contribution of the feature (batch_size,).
        """
        # Compute total contribution for normalization
        total_contrib = self._total_contribution(attributions)
        # Fractional contribution of the specific feature
        return torch.abs(attributions[:, feature_index]) / (total_contrib + 1e-8)

    def compute_complexity(self, attributions: torch.Tensor) -> torch.Tensor:
        """
        Compute the complexity score for the given attributions.

        Args:
            attributions (torch.Tensor): Saliency scores (batch_size, seq_len).

        Returns:
            torch.Tensor: Complexity score for the batch.
        """
        # Ensure attributions are 2D for consistent processing
        if attributions.ndim == 1:
            attributions = attributions.unsqueeze(0)  # Reshape to (1, seq_len)

        batch_size, seq_len = attributions.shape
        complexity = torch.zeros(batch_size, device=self._device)  # Initialize complexity scores

        # Iterate over each feature (token) to compute entropy
        for feature_index in range(seq_len):
            frac_contrib = self._fractional_contribution(attributions, feature_index)
            # Accumulate entropy for each feature
            complexity += -frac_contrib * torch.log(frac_contrib + 1e-8)

        # Normalize complexity by sequence length
        complexity = complexity / seq_len
        # Return the mean complexity across the batch
        return torch.mean(complexity)

    def evaluate(self, explanations, split_type="test"):
        """
        Evaluate the complexity of attributions for the given dataset split.

        Args:
            explanations: List of explanations containing text and saliency scores.
            split_type (str): Dataset split to evaluate (e.g., 'test').

        Returns:
            pd.DataFrame: DataFrame with complexity scores for each instance.
        """
        results = []
        for idx, exp in enumerate(explanations):
            text = exp.text  # Extract the text for the current instance
            attributions = torch.tensor(exp.scores, device=self._device)  # Convert saliency scores to tensor

            # Compute complexity for the current instance
            complexity_score = self.compute_complexity(attributions).item()  # Convert to Python float
            results.append({"Instance": idx, "Text": text, "Complexity": complexity_score})

        # Return results as a DataFrame
        return pd.DataFrame(results)

    def compute(self, explanations):
        """
        Compute the average complexity across the dataset.

        Args:
            explanations: List of explanations containing text and saliency scores.

        Returns:
            float: The average complexity score across the dataset.
        """
        # Ensure explanations are in list format
        explanations = explanations if isinstance(explanations, list) else [explanations]
        # Perform the evaluation
        results_df = self.evaluate(explanations, split_type="test")
        # Calculate the average complexity score
        avg_complexity = results_df["Complexity"].mean()

        return avg_complexity