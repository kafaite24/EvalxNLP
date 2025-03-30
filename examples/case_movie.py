import sys
sys.path.append('..')
from XAIbenchmark import XAIFramework
import warnings
import openpyxl
import torch
warnings.filterwarnings("ignore")

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from explainers import InputXGradientExplainer, IntegratedGradientsExplainer, DeepLiftExplainer, LimeExplainer, SHAPExplainer, SHAPIQExplainer, GuidedBackpropExplainer
from evaluators import AUCTPEvaluator, SoftComprehensivenessEvaluator, SoftSufficiencyEvaluator, FADEvaluator, SparsenessEvaluator, ComplexityEvaluator, IOUF1Evaluator,TokenF1Evaluator, AUPRCEvaluator
from dataset_loaders.dataset_loader import LoadDatasetArgs,load_fields_from_dataset
from dataset_loaders.movie_rationales import MovieRationalesProcessor
from dataset_loaders.hatexplain import HateSpeechProcessor
from LLMExplanationGenerator import LLMExplanationGenerator

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

ig= IntegratedGradientsExplainer(model,tokenizer,device=device)
gb= GuidedBackpropExplainer(model,tokenizer,device=device)
dl= DeepLiftExplainer(model,tokenizer,device=device)
ixg= InputXGradientExplainer(model,tokenizer,multiply_by_inputs=True,device=device)
g= InputXGradientExplainer(model,tokenizer,multiply_by_inputs=False,device=device)
lime= LimeExplainer(model,tokenizer)
shap= SHAPExplainer(model,tokenizer)
shapiq= SHAPIQExplainer(model,tokenizer)

sc= SoftComprehensivenessEvaluator(model,tokenizer,device=device)
ss= SoftSufficiencyEvaluator(model,tokenizer,device=device)
fad= FADEvaluator(model,tokenizer,device=device)
sp= SparsenessEvaluator(model,tokenizer,device=device)
cx= ComplexityEvaluator(model,tokenizer,device=device)
auctp= AUCTPEvaluator(model,tokenizer,device=device)
iou_f1= IOUF1Evaluator(model,tokenizer,device=device)
token_f1= TokenF1Evaluator(model,tokenizer,device=device)
auprc= AUPRCEvaluator(model,tokenizer,device=device)

xai_framework = XAIFramework(model, tokenizer,explainers=[gb,dl,ixg,g,ig,lime,shap,shapiq], evaluators=[sc,ss,fad,sp,cx,auctp,iou_f1,token_f1,auprc], device=device)

dataset_args_ = LoadDatasetArgs(
    dataset_name="eraser-benchmark/movie_rationales",
    text_field="review",
    label_field="label",
    rationale_field="evidences",
    dataset_split="test",
)

# Load the dataset fields
results = load_fields_from_dataset(dataset_args_)
input_texts= results['text']
labels= results['labels']
rationales= results['rationales']

mv= MovieRationalesProcessor(tokenizer)
processed_rationales= mv.process_dataset(input_texts, labels, rationales)

#Select a sub-sample if you want
input_texts_sample=input_texts[:5]
labels_sample=labels[:5]
rationale_sample= processed_rationales[:5]

api_key = "9b305c0291728a7f4ca861254b9bedd3e7785d5360100d93603dd1758931be6d"
explainer = LLMExplanationGenerator(api_key=api_key)

# Generate and save explanations (returns both explanations and file paths)
explanations, saved_files = explainer.generate_and_save_explanations(
    exps=exps,
    output_format="both"  # or "json"/"html"
)

# Print saved locations
print(f"Saved files: {[str(p) for p in saved_files]}")

xai_framework.benchmark_dataset(input_texts_sample, labels_sample,rationales=rationale_sample, save_path="../results/movie_rationales.xlsx")