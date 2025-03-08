import pandas as pd
import shelve
import hashlib
from typing import Dict, List, Optional, Tuple

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from evaluators import SoftSufficiencyEvaluator, FADEvaluator, AUPRCEvaluator, SparsenessEvaluator, IOUF1Evaluator,TokenF1Evaluator, SoftComprehensivenessEvaluator, ComplexityEvaluator, AUCTPEvaluator
from explainers import InputXGradientExplainer, IntegratedGradientsExplainer,DeepLiftExplainer, LimeExplainer, SHAPExplainer, SHAPIQExplainer, GuidedBackpropExplainer
from explanation import Explanation

from utils.text_classifier_utils import TextClassifierEngine
from utils.saliency_utils import lp_normalize
from utils.generate_heatmap import generate_explanation_heatmap

class XAIFramework:
    def __init__(self, model,tokenizer,explainers: Optional[List] = None, evaluators: Optional[List] = None, cache_filename="xai_cache.db", device="cpu"):
        self.cache_filename = cache_filename
        self.model= model
        self.tokenizer = tokenizer
        self.explainers = explainers or self._initialize_default_explainers()
        self.evaluators = evaluators or self._initialize_default_evaluators()
        self.text_classifier_engine = TextClassifierEngine(self.model, self.tokenizer)
        self.device = device

    def _initialize_model_and_tokenizer(self) -> Tuple[Optional[AutoModelForSequenceClassification], Optional[AutoTokenizer]]:
        """
        Initialize the model and tokenizer from the pretrained model name.
        """
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            print(f"Successfully loaded model from: {self.model_name}")
            return model, tokenizer
        except Exception as e:
            print(f"Error loading model from {self.model_name}: {e}")
            return None, None

    def _initialize_default_explainers(self) -> List[object]:
        return [
            SHAPExplainer(self.model, self.tokenizer),
            LimeExplainer(self.model, self.tokenizer),
            InputXGradientExplainer(self.model, self.tokenizer, multiply_by_inputs=False),
            InputXGradientExplainer(self.model, self.tokenizer, multiply_by_inputs=True),
            IntegratedGradientsExplainer(self.model, self.tokenizer, multiply_by_inputs=False),
            DeepLiftExplainer(self.model, self.tokenizer),
            SHAPIQExplainer(self.model, self.tokenizer),
            GuidedBackpropExplainer(self.model, self.tokenizer)
        ]

    def _initialize_default_evaluators(self) -> List[object]:
        return [
            SoftComprehensivenessEvaluator(self.model, self.tokenizer),
            SoftSufficiencyEvaluator(self.model, self.tokenizer),
            FADEvaluator(self.model, self.tokenizer),
            AUCTPEvaluator(self.model, self.tokenizer),
            ComplexityEvaluator(self.model, self.tokenizer),
            SparsenessEvaluator(self.model, self.tokenizer),
            AUPRCEvaluator(self.model, self.tokenizer),
            IOUF1Evaluator(self.model, self.tokenizer),
            TokenF1Evaluator(self.model, self.tokenizer)
        ]

    def _generate_cache_key(self, model: str, explainer: str, input_text: str, extra_params: Optional[Dict] = None) -> str:
        """
        Generate a unique cache key based on model, explainer, dataset, input text, and extra parameters.
        """ 
        text_hash = hashlib.md5(input_text.encode('utf-8')).hexdigest()

        input_str = f"{model}_{explainer}_{text_hash}"
        if extra_params:
            input_str += f"_{str(extra_params)}"
        cache_key = hashlib.md5(input_str.encode('utf-8')).hexdigest()
        return cache_key
    
    def create_pivot_table(self,dataset_metrics: Dict) -> pd.DataFrame:
        """
        Create a pivot table from the results DataFrame.

        Args:
            results_df (pd.DataFrame): DataFrame containing the results.

        Returns:
            pd.DataFrame: Pivot table.
        """
        try:
            results = []
            for explainer_name, explainer_metrics in dataset_metrics.items():
                for evaluator_name, metric_value in explainer_metrics.items():
                    result = {
                        "Explainer": explainer_name,
                        "Evaluator": evaluator_name,
                        "Metric Value": metric_value
                    }
                    results.append(result)

            results_df = pd.DataFrame(results)
            pivot_table = results_df.pivot(index="Explainer", columns="Evaluator", values="Metric Value")
            return pivot_table
        
        except Exception as e:
            print(f"Error generating explainer table for dataset: {e}")
            return pd.DataFrame()
    
    def benchmark_dataset(self, input_texts: List[str], labels: List[str], rationales: Optional[List[str]] = None, extra_params: Optional[Dict] = None, split_type: str = "test") -> pd.DataFrame:
        """
        Generate a table with results for all explainers and metrics for the entire dataset.
        """
        try:
            dataset_metrics = self.get_evaluation_metrics(input_texts, labels, rationales, extra_params)
            pivot_table = self.create_pivot_table(dataset_metrics)
            return pivot_table
        except Exception as e:
            print(f"Error generating explainer table for dataset: {e}")

    def get_evaluation_metrics(self, input_texts: List[str], labels: List[str], rationales: Optional[List[str]], extra_params: Optional[Dict] = None) -> Dict[str, Dict[str, float]]:
        """
        Retrieve evaluation metrics for a model, explainer, and dataset, either from cache or by computation.
        Now supports a list of explainer names.
        """
        try:           
            explainer_results = self.get_feature_importance_for_dataset(input_texts, labels, rationales)
            metrics = self.compute_evaluation_metrics(explainer_results)
            return metrics
        except Exception as e:
            print(f"Error retrieving evaluation metrics: {e}")
            return {}

    def compute_evaluation_metrics(self, explainer_results: Dict[str, List[Explanation]]) -> Dict[str, Dict[str, float]]:
        """
        Compute evaluation metrics for the entire dataset batch.
        """
        try:
            evaluator_results = {}
            for explainer_name, explanations in explainer_results.items():
                print(f"\nExplainer: {explainer_name}")
                evaluator_results[explainer_name] = {}
                evaluators = self.evaluators
                for evaluator in evaluators:
                    print(f"Computing value for {evaluator.NAME}")
                    evaluator_name = evaluator.NAME
                    try:
                        metric_result = evaluator.compute(explanations)
                    except Exception as e:
                        print(f"Error computing metric for {evaluator_name}: {e}")
                        continue
                    evaluator_results[explainer_name][evaluator_name] = metric_result
            return evaluator_results
        except Exception as e:
            print(f"Error computing evaluation metrics: {e}")
            return {}

    def process_text_list(self,text_list):
        """
        Processes a list of text inputs. If the input is a list of tuples, combines each tuple into a single string using the separator token.
        If the input is a list of strings, leaves it as is.

        Args:
            text_list (List[Union[str, Tuple[str, str]]): List of text inputs, which can be strings or tuples of two strings.
            sep_token (str): Separator token to use when combining tuples. Default is "[SEP]".

        Returns:
            List[str]: Processed list of strings.
        """
        sep_token="[SEP]"
        processed_texts = []

        for text in text_list:
            if isinstance(text, str):
                # If it's a string, add it directly to the list
                processed_texts.append(text)
            elif isinstance(text, tuple) and len(text) == 2:
                # If it's a tuple of two strings, combine them with the separator token
                combined_text = f"{text[0]} {sep_token} {text[1]}"
                processed_texts.append(combined_text)
            else:
                # Handle unexpected types (optional: raise an error or skip)
                raise ValueError(f"Unsupported input type: {type(text)}. Expected str or tuple of two strings.")

        return processed_texts

    def get_feature_importance_for_dataset(self, input_texts: List[str], labels: List[str], rationales: Optional[List[str]]=None, extra_params: Optional[Dict] = None, split_type: str = "test") -> Dict[str, List[Explanation]]:
        """
        Compute feature importance scores for the entire dataset, for all explainer methods.
        Now using the output from xai_framework.load_fields_from_dataset.

        Args:
            input_texts (List[str]): List of input texts.
            labels (List[str]): List of labels.
            rationales (Optional[List[str]]): List of rationales.
            explainer_names (List[str]): List of explainer names to use.
            extra_params (Optional[Dict]): Additional parameters for explainers.
            split_type (str): Type of dataset split (e.g., "train", "test").

        Returns:
            Dict[str, List[Explanation]]: Dictionary of explanations for each explainer.
        """
        
        processed_texts = self.process_text_list(input_texts)
        try:
            all_explanations = {}
            indices = range(len(processed_texts))

            for instance_id in indices:
                print(f"Processing instance {instance_id}")
                text = processed_texts[instance_id]
                target_label = labels[instance_id]
                rationale = rationales[instance_id] if rationales is not None else None
            
                explainer_results = self.get_feature_importance_for_instance(text, target_label, rationale, instance_id)
                for explainer_name, explanation in explainer_results.items():
                    if explainer_name not in all_explanations:
                        all_explanations[explainer_name] = []
                    all_explanations[explainer_name].append(explanation)
            return all_explanations
        except Exception as e:
            print(f"Error computing feature importance for dataset: {e}")
            return {}

    def get_feature_importance_for_instance(self, text: str, target_label: int, rationale: Optional[str], extra_params: Optional[Dict] = None) -> Dict[str, Explanation]:
        """
        Retrieve or compute feature importance scores for a given model, explainer, and dataset instance.
        """
        print(text)
        print(target_label)
        try:
            all_results = {}
            explainers = self.explainers
           
            for explainer in explainers:
                cache_key = self._generate_cache_key(self.model, explainer.NAME, text, extra_params)
                with shelve.open(self.cache_filename) as cache:
                    if cache_key in cache:
                        print(f"Found cached explanation for {explainer.NAME}")
                        all_results[explainer.NAME] = cache[cache_key]
                    else:
                        print(f"Computing explanation for {explainer.NAME}")
                        explanation = explainer.compute_feature_importance(text, target=target_label)
                        explanation = lp_normalize([explanation])
                        explanation.rationale = rationale
                        cache[cache_key] = explanation
                        all_results[explainer.NAME] = explanation

            return all_results
        except Exception as e:
            print(f"Error computing feature importance for instance: {e}")
            return {}

    def explain(self, input_data: str, target_label: int = 1) -> pd.DataFrame:
        """
        Generate and visualize explanations for single or multiple input sentences.

        Args:
            input_data (str): Single input sentence or a list of input sentences.
            target_label (int): Target label for explanation.
            explainer_names (Optional[List[str]]): List of explainer names to use.

        Returns:
            pd.DataFrame: Combined heatmap data for all inputs.
        """
        try:
            explainers = self.explainers

            validated_input = self.text_classifier_engine.validate_input(input_data)
            explanation_outputs = []

            for explainer in explainers:
                result = explainer.compute_feature_importance(validated_input, target_label)
                explanation_outputs.append(result)

            explanation_outputs = lp_normalize(explanation_outputs)
            return explanation_outputs
        except Exception as e:
            print(f"Error generating explanations: {e}")
            return pd.DataFrame()

    def visualize(self, explanations: List[Explanation], skip_boundary_tokens: bool = True, visualization_style: Optional[str] = "cool_warm") -> pd.DataFrame:
        """
        Visualize explanations as heatmaps.

        Args:
            explanations (List[Explanation]): List of explanations to visualize.
            skip_boundary_tokens (bool): Whether to skip [CLS] and [SEP] tokens.
            visualization_style (Optional[str]): Heatmap visualization style.

        Returns:
            pd.DataFrame: Heatmap data.
        """
        try:
            generate_explanation_heatmap(explanations, skip_boundary_tokens, visualization_style)
            return
        except Exception as e:
            print(f"Error visualizing explanations: {e}")
            return pd.DataFrame()

    def classify_text(self, input_data: str) -> Dict[str, float]:
        """
        Classify input text and return probabilities for each class.

        Args:
            input_data (str): Input text to classify.

        Returns:
            Dict[str, float]: Dictionary of class probabilities.
        """
        try:
            _, prediction_logits = self.text_classifier_engine.run_inference(input_data)
            probabilities = prediction_logits[0].softmax(-1)
            return {self.model.config.id2label[idx]: prob.item() for idx, prob in enumerate(probabilities)}
        except Exception as e:
            print(f"Error classifying text: {e}")
            return {}

    def evaluate_single_sentence(self, input_data: str, target_label: int = 1, human_rationale: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Evaluate all explainers and evaluators with the option to use human rationale for plausibility metrics.

        Args:
            input_data (str): Input text or data for explanation.
            target_label (int): The target label for the explanation.
            explainer_names (Optional[List[str]]): List of explainer names to evaluate. Defaults to all registered explainers.
            evaluator_names (Optional[List[str]]): List of evaluator names to use. Defaults to all registered evaluators.
            human_rationale (Optional[List[int]]): Human-provided rationale for plausibility metrics. Defaults to None.

        Returns:
            pd.DataFrame: A pivot table summarizing evaluation results.
        """
        try:
            explainers= self.explainers
            evaluators= self.evaluators
            results = []

            for explainer in explainers:
                print(f"explainer {explainer.NAME}")
                explanation = explainer.compute_feature_importance(input_data, target_label)
                explanation = lp_normalize([explanation])
                explanation.rationale = human_rationale
                for evaluator in evaluators:
                    try:
                        if evaluator.requires_human_rationale and human_rationale is None:
                            metric_result = None
                            print(f"Skipping {evaluator.NAME} as human rationale is not provided.")
                            continue
                        else:
                            metric_result = evaluator.compute([explanation])
                        results.append({
                            "Explainer": explainer.NAME,
                            "Evaluator": evaluator.NAME,
                            "Value": metric_result
                        })
                    except:
                        print(f"Skipping {evaluator.NAME} due to an error.")
                        continue

            results_df = pd.DataFrame(results)
            pivot_df = results_df.pivot_table(index="Explainer", columns="Evaluator", values="Value")

            return pivot_df
        except Exception as e:
            print(f"Error evaluating single sentence: {e}")
            return pd.DataFrame()



