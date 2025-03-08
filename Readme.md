# EvalXAI Framework

## Table of Contents

- [Introduction](#introduction)
  - [Explainers](#explainers)
  - [Evaluation Metrics](#metrics)
- [Getting Started](#getting-started)
- [Installation](#installation)
- [Usage](#usage)
  - [Explain and Evaluate](#explain-and-evaluate)
  - [Loading the Dataset](#loading-the-dataset)
  - [Explain and Benchmark](#explain-and-benchmark)
- [Tutorials](#tutorials)

## Introduction

EvalXAI is an easy-to-use yet comprehensive toolbox that enables the quantitative evaluation, comparison, and benchmarking of post-hoc explainable AI methods, particularly feature attribution methods. It offers a comprehensive, end-to-end pipeline to assess the performance and reliability of these explanation methods.

### Explainers

This project supports several explanation methods to help understand the model’s predictions. Each method attributes importance to input features (tokens) in different ways:

-**Gradient** [https://arxiv.org/abs/1312.6034](Simonyan et al., 2014)
Uses plain gradients or gradients multiplied by input embeddings to measure feature influence on the model’s prediction.

-**Integrated Gradient** [https://arxiv.org/abs/1703.01365](Sundararajan et al., 2017)
Computes gradients from a baseline input to the actual input, aggregating them to measure each feature’s contribution.

-**Guided Backprop** [https://arxiv.org/abs/1412.6806](Springenberg et al.,2015)
Modifies backpropagation to mask negative gradients, highlighting features with positive contributions to the prediction.

-**DeepLift** [https://arxiv.org/abs/1704.02685](Shrikumar et al., 2019)
Compares model outputs relative to a baseline, propagating differences across the network to determine feature importance.

-**SHAP** [https://arxiv.org/abs/1705.07874](Lundberg & Lee, 2017)
Uses Shapley values from cooperative game theory to fairly distribute feature importance across all possible feature subsets.

-**SHAPIQ** [https://arxiv.org/abs/2410.01649](Muschalik et al., 2024)
A variant of SHAP designed for classification tasks, refining Shapley value approximations for better interpretation in classification problems.

-**LIME** [https://arxiv.org/abs/1602.04938](Ribeiro et al., 2016)
A model-agnostic method that explains predictions by approximating the model with a simpler surrogate model in the local neighborhood of the instance.

### Metrics

This library aims to aggregate and implement evaluation metrics from the XAI literature to automate XAI quantification. These metrics generally fall into one of four main categories: faithfulness, robustness, plausibility, complexity. Our library provides implementations of the following evaluation metrics:

#### Faithfulness:

Measures how well the explanations reflect the model’s predictive behavior.

**Soft Comprehensiveness** [https://aclanthology.org/2023.acl-long.261/](Zhao and Aletras, 2023)

**Soft Sufficiency** [https://aclanthology.org/2023.acl-long.261/](Zhao and Aletras, 2023)

**FAD and N-AUC** [https://aclanthology.org/2022.bionlp-1.33/](Ngai and Rudzicz, 2022)

**AUC-TP** [https://arxiv.org/abs/2009.13295] (Atanasova,2020)
Computes the area under the curve where the x-axis is the proportion of features removed and the y-axis is the performance change

#### Plausibility:

Assesses how intuitively the explanations align with human understanding.

**Area-Under-Precision-Recall-Curve** [https://aclanthology.org/2020.acl-main.408/](DeYoung et al., 2020)
Evaluates how well the explanation prioritizes truly important tokens based on precision-recall tradeoffs.

**Token F1** [https://aclanthology.org/2020.acl-main.408/](DeYoung et al., 2020)
Measures the overlap between predicted important tokens and human-annotated tokens.

**Token Intersection Over Union** [https://aclanthology.org/2020.acl-main.408/](DeYoung et al., 2020)
Assesses the similarity between explanation token sets and ground-truth annotations.

### Complexity

Evaluates the conciseness of explanations (fewer features imply higher clarity).

**Sparseness** [https://arxiv.org/abs/1810.06583](Chalasani et al., 2020): uses the Gini Index for measuring, if only highly attributed features are truly predictive of the model output

**Complexity** [https://arxiv.org/abs/2005.00631](Bhatt et al., 2020): computes the entropy of the fractional contribution of all features to the total magnitude of the attribution individually

## Getting Started

### Prerequisites

The project is entirely made in Python 3.9 and tested on Windows 11 64 bits.

### Installation

EvalXAI project can be installed from source using:

1. Clone the repo

```bash
git clone https://github.com/kafaite24/EvalXAI.git
```

2. Install the requirements

```bash
pip install -r requirements.txt
```

### Usage

#### Explain and Evaluate

Below is the example of how you can use the library in a very simple way to generate explanations of a given input sentence for any explainers you want.

Initialize the model and tokenizer

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
```

Initialize explainers and evaluators that you want

```python
from explainers import DeepLiftExplainer, GuidedBackpropExplainer
from evaluators import AUCTPEvaluator, SoftComprehensivenessEvaluator, FADEvaluator, SparsenessEvaluator

gb= GuidedBackpropExplainer(model,tokenizer)
dl= DeepLiftExplainer(model,tokenizer)

sc= SoftComprehensivenessEvaluator(model,tokenizer)
fad= FADEvaluator(model,tokenizer)
sp= SparsenessEvaluator(model,tokenizer)
auctp= AUCTPEvaluator(model,tokenizer)
```

Initialize the Benchmark class

```python
xai_framework = XAIFramework(model, tokenizer,explainers=[gb,dl], evaluators=[auprc,auctp,sc,fad,sp,cx,iou])
```

Explain and visualize the scores

```python
sentence = "I love the product, but the customer service was terrible."
exps= xai_framework.explain(input_data=sentence,target_label="negative")
xai_framework.visualize(exps)

```

You can evaluate these explanations for the defined evaluation metrics as follows.
You have the flexibility to define the explainers and evaluators you want the results for.

```python
xai_framework.evaluate_single_sentence(sentence, target_label="negative", human_rationale=[0, 1, 0, 1, 0,0,0,0,1,0,1,0])
```

#### Loading the Dataset

First, load your dataset using the [load_fields_from_dataset] function as follows. You can load a huggingface model or csv/excel file.

```python
from dataset_loaders.dataset_args import LoadDatasetArgs

dataset_args_ = LoadDatasetArgs( # Path to the dataset file
    # dataset_name= "csv",
    dataset_name="eraser-benchmark/movie_rationales",
    text_field="review",  # Assuming "text" column contains the text data
    label_field="label",  # Assuming "label" column contains the sentiment labels
    rationale_field="evidences",
    dataset_split="test",  # Assuming you want to load the test split
    # dataset_files=["healthFC_annotated.csv"]
)

# Load the dataset fields
results = load_fields_from_dataset(dataset_args_)
```

#### Explain and Benchmark

With a single line of code, you can explain and evaluate an entire dataset or even sub-samples of it.

```python
xai_framework.benchmark_dataset(input_texts_sample, labels_sample,rationales=rationale_sample)
```

## Tutorials

Further tutorials are available that demonstrate usage of the library for different usecases. Please see notebooks in the examples folder which include:

[Sentiment Analysis example](./examples/tutorial_sentiment_analysis.ipynb)

[Hate Speech Detection example](./examples/tutorial_hate_speech_detection.ipynb)

[Natural Language Inference example](./examples/tutorial_nli.ipynb)
