import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from scipy.special import softmax
import torch
from numpy import trapz
from explanation import Explanation
from tqdm import tqdm
from .baseevaluator import BaseEvaluator

class FADEvaluator(BaseEvaluator):
    """
    Class to evaluate the impact of feature (token) dropping on text data
    based on saliency scores (e.g., Gradient x Input).
    """
    NAME = "FAD ↓"

    def __init__(self, model, tokenizer, device="cpu", batch_size=32):
        """
        Initialize the evaluator.

        Args:
            model: The trained text classification model.
            tokenizer: Tokenizer used for the model.
            batch_size: Number of samples to process in a batch.
            device: Device to run computations on (e.g., "cuda:0" or "cpu").
        """
        # Store model, tokenizer, and other configurations
        self.model = model
        self.tokenizer = tokenizer
        self.requires_human_rationale = False
        self.batch_size = batch_size
        self._device = device

    def _replace_tokens_with_baseline(self, tokens, saliency_scores, percent_to_drop):
        """
        Replace the top `percent_to_drop` tokens with a baseline token (e.g., [MASK]).

        Args:
            tokens: List of tokens.
            saliency_scores: Saliency scores for each token.
            percent_to_drop: Percentage of tokens to drop.

        Returns:
            List of modified tokens.
        """
        # Copy tokens to avoid modifying the original list
        tokens_modified = tokens.copy()
        # Calculate the number of tokens to drop
        num_tokens_to_drop = int(len(tokens) * percent_to_drop / 100)
        # Sort tokens by saliency scores in descending order
        saliency_sorted_indices = np.argsort(-np.abs(saliency_scores))
        # Select the indices of tokens to replace
        tokens_to_replace = saliency_sorted_indices[:num_tokens_to_drop]

        # Replace selected tokens with the baseline token
        baseline_token = self.tokenizer.mask_token
        for idx in tokens_to_replace:
            tokens_modified[idx] = baseline_token
        return tokens_modified

    def evaluate(self, explanations, percent_dropped_features):
        """
        Evaluate the impact of dropping tokens on model accuracy.

        Args:
            explanations: List of explanation objects.
            percent_dropped_features: List of percentages of tokens to drop.

        Returns:
            pd.DataFrame: Results containing accuracy for each percentage of tokens dropped.
        """
        results = []
        for percent_to_drop in percent_dropped_features:
            predictions, labels = [], []

            # Process explanations in batches
            for i in range(0, len(explanations), self.batch_size):
                batch_explanations = explanations[i:i + self.batch_size]
                batch_texts = []
                batch_labels = []

                for exp in batch_explanations:
                    # Extract tokens, saliency scores, and target label
                    tokens = exp.tokens
                    saliency_scores = np.array(exp.scores)
                    label = exp.target_pos_idx

                    # Replace top tokens with baseline and convert back to text
                    modified_tokens = self._replace_tokens_with_baseline(tokens, saliency_scores, percent_to_drop)
                    modified_text = self.tokenizer.convert_tokens_to_string(modified_tokens)
                    batch_texts.append(modified_text)
                    batch_labels.append(label)

                # Tokenize the modified texts and move inputs to the specified device
                encoded_inputs = self.tokenizer(
                    batch_texts, return_tensors="pt", truncation=True, max_length=512, padding=True
                )
                encoded_inputs = {key: val.to(self._device) for key, val in encoded_inputs.items()}

                # Perform inference to get predictions
                with torch.no_grad():
                    logits = self.model(**encoded_inputs).logits
                batch_predictions = torch.argmax(logits, dim=1).cpu().numpy()

                # Collect predictions and labels
                predictions.extend(batch_predictions)
                labels.extend(batch_labels)

            # Calculate accuracy for the current percentage of tokens dropped
            accuracy = accuracy_score(labels, predictions)
            results.append({"Percent Dropped": percent_to_drop, "Accuracy": accuracy})

        return pd.DataFrame(results)

    def calculate_n_auc(self, results_df, percent_range=(0, 20)):
        """
        Calculate the normalized AUC for the given percentage range.

        Args:
            results_df: DataFrame containing the results.
            percent_range: Range of percentages to consider for AUC calculation.

        Returns:
            float: Normalized AUC.
        """
        # Filter results within the specified percentage range
        filtered_results = results_df[(results_df["Percent Dropped"] >= percent_range[0]) &
                                      (results_df["Percent Dropped"] <= percent_range[1])]
        x = filtered_results["Percent Dropped"].values
        y = filtered_results["Accuracy"].values
        # Compute the area under the curve (AUC)
        auc = trapz(y, x)
        # Normalize the AUC by dividing by the maximum possible AUC
        max_auc = (x[-1] - x[0]) * max(y)
        return auc / max_auc if max_auc > 0 else 0.0

    def plot_results(self, results_df):
        """
        Plot the results (accuracy vs. percentage of tokens dropped).

        Args:
            results_df: DataFrame containing the results.
        """
        import matplotlib.pyplot as plt
        # Create a line plot for accuracy vs. percentage of tokens dropped
        plt.figure(figsize=(8, 5))
        plt.plot(results_df["Percent Dropped"], results_df["Accuracy"], marker="o", label="Accuracy")
        plt.xlabel("Percent of Tokens Dropped")
        plt.ylabel("Accuracy")
        plt.title("Impact of Dropping Top Tokens on Accuracy")
        plt.grid()
        plt.legend()
        plt.show()

    def compute(self, explanations, percent_dropped_features=None, percent_range=(0, 20)):
        """
        Compute the normalized AUC for the given explanations.

        Args:
            explanations: List of explanation objects.
            percent_dropped_features: List of percentages of tokens to drop.
            percent_range: Range of percentages to consider for AUC calculation.

        Returns:
            float: Normalized AUC.
        """
        # Ensure explanations are in list format
        explanations = explanations if isinstance(explanations, list) else [explanations]
        # Default percentages to drop if not provided
        if percent_dropped_features is None:
            percent_dropped_features = list(range(0, 41, 10))

        # Evaluate the model and calculate normalized AUC
        results_df = self.evaluate(explanations, percent_dropped_features)
        final_n_auc = self.calculate_n_auc(results_df, percent_range)
        # self.plot_results(results_df)  # Uncomment to plot results
        return final_n_auc