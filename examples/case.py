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

xai_framework = XAIFramework(model, tokenizer,device=device)

dataset_args = LoadDatasetArgs(dataset_name="eraser-benchmark/movie_rationales", text_field="review", label_field="label", rationale_field="evidences", dataset_split="test")
results = load_fields_from_dataset(dataset_args)

processed_rationales = MovieRationalesProcessor(tokenizer).process_dataset(results['text'], results['labels'], results['rationales'])

sample = slice(2)
xai_framework.benchmark_dataset(results['text'][sample], results['labels'][sample],rationales=processed_rationales[sample], save_metrics_path="../results/metrics/movie_rationales.xlsx",save_scores_path="../results/scores/movie_rationales.json")