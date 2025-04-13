import numpy as np
from sklearn.metrics import auc, precision_recall_curve
from explanation import Explanation
from .baseevaluator import BaseEvaluator

class AUPRCEvaluator(BaseEvaluator):
    """
    Class to evaluate saliency explanations using AUPRC (Area Under the Precision-Recall Curve).
    """
    NAME = "AUPRC ↑"  # Metric name displayed in evaluation results

    def __init__(self, model, tokenizer, device="cpu"):
        """
        Initialize the evaluator.

        Args:
            model: The model used for generating explanations.
            tokenizer: The tokenizer used for processing text.
            device (str): Device to use for computations ("cuda" or "cpu").
        """
        self.model = model  # Model used for generating saliency explanations
        self.tokenizer = tokenizer  # Tokenizer for text processing
        self._device = device  # Device for computation
        self.requires_human_rationale = True  # Indicates if human rationale is required

    @staticmethod
    def calculate_auprc(saliency_scores: list[float], human_rationale: list[int]) -> float:
        """
        Calculate AUPRC for a single instance using sklearn's precision_recall_curve.

        Args:
            saliency_scores (list of float): Saliency scores for a single instance.
            human_rationale (list of int): Ground-truth rationale mask for the instance.

        Returns:
            float: AUPRC score for the instance.
        """
        # Check for invalid or empty inputs
        if human_rationale is None or saliency_scores is None or len(saliency_scores) == 0 or len(human_rationale) == 0:
            return None  # Skip computation for invalid inputs
        
        # Remove [CLS] and [SEP] tokens from saliency scores
        saliency_scores = saliency_scores[1:-1]
        # Truncate human rationale to match the length of saliency scores
        human_rationale = human_rationale[:len(saliency_scores)]

        # Compute precision-recall curve and calculate AUC
        precision, recall, _ = precision_recall_curve(human_rationale, saliency_scores)
        return auc(recall, precision)  # Return the area under the precision-recall curve

    def evaluate_instance(self, explanation: Explanation, human_rationale: list[int]) -> float:
        """
        Evaluate AUPRC for a single instance.

        Args:
            explanation (Explanation): Explanation object containing saliency scores.
            human_rationale (list of int): Ground-truth rationale for the instance.

        Returns:
            float: AUPRC score for the instance.
        """
        # Extract saliency scores from the explanation object
        saliency_scores = explanation.scores
        # Calculate AUPRC for the instance
        return self.calculate_auprc(saliency_scores, human_rationale)

    def evaluate(self, explanations) -> float:
        """
        Evaluate AUPRC for all instances in the dataset or for a single instance.

        Args:
            explanations: List of explanations or a single explanation object.

        Returns:
            float: Average AUPRC score across all instances.
        """        
        # Single instance evaluation
        if len(explanations) == 1:
            human_rationales = explanations[0].rationale
            if human_rationales is None:
                raise ValueError("Human rationale is required to compute this metric.")
            return self.evaluate_instance(explanations[0], human_rationales)

        # Dataset evaluation
        auprc_scores = []  # List to store AUPRC scores for all instances
        for exp in explanations:
            human_rationales = exp.rationale
            # Evaluate each instance and append the score
            score = self.evaluate_instance(exp, exp.rationale)
            if score is not None:
                auprc_scores.append(score)

        # Return the average AUPRC score across all instances
        return np.mean(auprc_scores) if auprc_scores else 0.0

    def compute(self, explanations) -> float:
        """
        Evaluate AUPRC for the entire dataset or a single instance.

        Args:
            explanations: List of explanations or a single explanation object.

        Returns:
            float: Average AUPRC score across the dataset or for a single instance.
        """
        # Ensure explanations is a list
        explanations = explanations if isinstance(explanations, list) else [explanations]

        # Compute and return the average AUPRC score
        average_auprc = self.evaluate(explanations)
        return average_auprc