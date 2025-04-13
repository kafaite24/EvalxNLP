import numpy as np
import torch
import torch.nn.functional as F
from utils.saliency_utils import top_k_selection
from tqdm import tqdm
from .baseevaluator import BaseEvaluator

class SoftComprehensivenessEvaluator(BaseEvaluator):
    """
    Class to evaluate Soft Normalized Comprehensiveness using Explanation objects.
    """
    NAME = "Soft Comprehensiveness ↑"

    def __init__(self, model, tokenizer, device="cpu", max_len=512):
        """
        Initialize the evaluator.

        Args:
            model: The pre-trained model (e.g., BERT).
            tokenizer: Tokenizer used to tokenize text inputs.
            max_len: Maximum token length for padding/truncation.
            device: Device to run computations on (e.g., "cuda:0" or "cpu").
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_len = max_len
        self._device = device
        self.requires_human_rationale = False

    def soft_perturb(self, embeddings, importance_scores, attention_mask):
        """
        Apply soft perturbation to embeddings based on importance scores.

        Args:
            embeddings: Original embeddings of the input text.
            importance_scores: Saliency scores indicating token importance.
            attention_mask: Mask to indicate valid tokens.

        Returns:
            perturbed_embeddings: Embeddings after applying perturbation.
        """
        # Normalize importance scores to range [0, 1]
        importance_scores = importance_scores.unsqueeze(-1).to(self._device)
        attention_mask = attention_mask.unsqueeze(-1).float().to(self._device)
        normalized_importance_scores = (importance_scores - importance_scores.min()) / (importance_scores.max() - importance_scores.min())
        
        # Generate a mask using Bernoulli sampling and apply it to embeddings
        mask = torch.bernoulli(1 - normalized_importance_scores)
        mask = mask * attention_mask
        perturbed_embeddings = embeddings * mask
        return perturbed_embeddings

    def compute_comprehensiveness(self, original_input, perturbed_input):
        """
        Compute comprehensiveness score by comparing predictions on original and perturbed inputs.

        Args:
            original_input: Tokenized input for the original text.
            perturbed_input: Tokenized input for the perturbed text.

        Returns:
            comprehensiveness: Array of comprehensiveness scores for each input.
        """
        # Get model predictions for the original input
        original_output = self.model(**original_input)
        original_prediction = F.softmax(original_output.logits, dim=-1).detach().cpu().numpy()
        full_text_probs = original_prediction.max(-1)  # Maximum probability for the predicted class
        full_text_class = original_prediction.argmax(-1)  # Predicted class indices

        # Get model predictions for the perturbed input
        rows = np.arange(original_input["input_ids"].size(0))
        perturbed_output = self.model(**perturbed_input)
        reduced_probs = F.softmax(perturbed_output.logits, dim=-1).detach().cpu().numpy()[rows, full_text_class]

        # Compute comprehensiveness as the reduction in probability for the predicted class
        comprehensiveness = np.maximum(0, full_text_probs - reduced_probs)
        return comprehensiveness

    def compute_batch(self, explanations, batch_size=1):
        """
        Compute comprehensiveness scores for a batch of explanations.

        Args:
            explanations: List of Explanation objects containing text and saliency scores.
            batch_size: Number of explanations to process in a single batch.

        Returns:
            Mean comprehensiveness score across all explanations.
        """
        all_comprehensiveness = []
        
        for i in range(0, len(explanations), batch_size):
            # Process a batch of explanations
            batch_explanations = explanations[i:i + batch_size]

            # Extract original sentences and saliency scores
            original_sentences = [exp.text[0] if isinstance(exp.text, list) else exp.text for exp in batch_explanations]
            saliency_scores_batch = [torch.tensor(exp.scores).to(self._device) for exp in batch_explanations]

            # Tokenize the original sentences
            original_input = self.tokenizer(
                original_sentences, padding=True, truncation=True, 
                max_length=self.max_len, return_tensors="pt"
            )
            original_input = {key: val.to(self._device) for key, val in original_input.items()}
            
            # Get the original embeddings from the model
            with torch.no_grad():
                outputs = self.model(**original_input, output_hidden_states=True)
                original_embeddings = outputs.hidden_states[-1]

            # Apply soft perturbation to the embeddings
            perturbed_embeddings = self.soft_perturb(
                original_embeddings, torch.stack(saliency_scores_batch), original_input['attention_mask']
            )
            
            # Create perturbed input by replacing input_ids with perturbed embeddings
            perturbed_input = original_input.copy()
            perturbed_input['input_ids'] = perturbed_embeddings.argmax(dim=-1)

            # Compute comprehensiveness scores for the batch
            comp_scores = self.compute_comprehensiveness(original_input, perturbed_input)
            all_comprehensiveness.extend(comp_scores)

        # Return the mean comprehensiveness score
        return np.mean(all_comprehensiveness)

    def compute(self, explanations, batch_size=1):
        """
        Compute comprehensiveness scores for all explanations.

        Args:
            explanations: List of Explanation objects or a single Explanation object.
            batch_size: Number of explanations to process in a single batch.

        Returns:
            Mean comprehensiveness score across all explanations.
        """
        explanations = explanations if isinstance(explanations, list) else [explanations]
        return self.compute_batch(explanations, batch_size)