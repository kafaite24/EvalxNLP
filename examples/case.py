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

xai_framework.benchmark_dataset(input_texts_sample, labels_sample,rationales=rationale_sample, save_path="../results/movie_rationales.xlsx")