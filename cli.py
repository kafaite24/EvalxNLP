import argparse
from Benchmarking import benchmark, MODEL_NAME_MAPPING
from explainers import IntegratedGradientsExplainer, GuidedBackpropExplainer
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Benchmark NLP Explainability Methods")
    parser.add_argument("--model", required=True, help="Pretrained model name")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g., HateXplain)")
    parser.add_argument("--explainer", choices=["IntegratedGradients", "GuidedBackprop"], default="IntegratedGradients",
                        help="Explainer to use (default: IntegratedGradients)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing results")
    parser.add_argument("--output", help="File to save benchmark results (e.g., results.json)")

    args = parser.parse_args()

    # Validate model
    if args.model not in MODEL_NAME_MAPPING:
        print(f"Error: Model '{args.model}' not supported. Choose from: {list(MODEL_NAME_MAPPING.keys())}")
        return

    # Map explainer
    available_explainers = {
        "IntegratedGradients": IntegratedGradientsExplainer,
        "GuidedBackprop": GuidedBackpropExplainer,
    }

    if args.explainer not in available_explainers:
        print(f"Error: Explainer '{args.explainer}' not supported.")
        return
    explainer_class = available_explainers[args.explainer]

    # Run benchmark
    print(f"Running benchmark for model: {args.model}, dataset: {args.dataset}, explainer: {args.explainer}")
    results = benchmark(MODEL_NAME_MAPPING, [args.dataset], [explainer_class], overwrite=args.overwrite)

    # Output results
    if args.output:
        results.to_json(args.output, orient="records", indent=4)
        print(f"Results saved to {args.output}")
    else:
        print("\nBenchmark Results:")
        print(results)

if __name__ == "__main__":
    main()
