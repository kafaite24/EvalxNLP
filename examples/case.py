import sys, warnings, torch, openpyxl
sys.path.append('..')
warnings.filterwarnings("ignore")
from XAIbenchmark import XAIFramework
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from dataset_loaders.dataset_loader import LoadDatasetArgs, load_fields_from_dataset
from dataset_loaders.movie_rationales import MovieRationalesProcessor
from LLMExplanationGenerator import LLMExplanationGenerator

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
xai_framework = XAIFramework(model, tokenizer, device=device)

dataset_args = LoadDatasetArgs(dataset_name="eraser-benchmark/movie_rationales", text_field="review", label_field="label", rationale_field="evidences", dataset_split="test")
results = load_fields_from_dataset(dataset_args)
processed_rationales = MovieRationalesProcessor(tokenizer).process_dataset(results['text'], results['labels'], results['rationales'])

sample = slice(2)
exp_scores = xai_framework.get_feature_importance_for_dataset(results['text'][sample], results['labels'][sample], processed_rationales[sample])

explanations, saved_files = LLMExplanationGenerator(api_key="9b305c0291728a7f4ca861254b9bedd3e7785d5360100d93603dd1758931be6d").generate_and_save_explanations(exp_scores, output_format="html")
print(f"Saved files: {[str(p) for p in saved_files]}")

xai_framework.create_pivot_table(xai_framework.compute_evaluation_metrics(exp_scores))