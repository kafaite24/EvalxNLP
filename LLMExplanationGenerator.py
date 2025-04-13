import json
from pathlib import Path
from IPython.display import display, HTML
from together import Together
import os

class LLMExplanationGenerator:
    def __init__(self, api_key, model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"):
        """
        Initialize the explanation generator with API key, model name, and output directory setup.
        """
        # Mapping of explanation methods to their full names
        self.explainer_names = {
            'DL': 'DeepLift',
            'IG': 'Integrated Gradients',
            'GBP': 'Guided Backprop',
            'SHAP': 'SHAP',
            'LIME': 'LIME',
            'SHAP-I': 'SHAP-I',
            'GxI': 'Gradient×Input',
            'Saliency': 'Saliency'
        }
        self.model_name = model_name  # Name of the model to use for generating explanations
        self.api_key = api_key  # API key for authentication
        self.client = Together(api_key=self.api_key)  # Client for interacting with the Together API
        self.output_dir = Path("../results/llm_explanations")  # Directory to save output files
        self.output_dir.mkdir(exist_ok=True)  # Create the directory if it doesn't exist

    def generate_and_save_explanations(self, exps, output_format="both", filename="scores_llm_explanations"):
        """
        Generate explanations for the given inputs and save them in the specified format(s).
        
        Args:
            exps: List or dictionary of explanation objects.
            output_format: Format to save the explanations ("json", "html", or "both").
            
        Returns:
            Tuple: List of generated explanations and paths to saved files.
        """
        explanations = self.generate_explanation(exps)  # Generate explanations
        saved_files = []

        # Strip any extensions from filename
        base_filename = os.path.splitext(filename)[0]

        # Save explanations in JSON format for programmatic use
        json_path = self.output_dir / f"{base_filename}.json"
        self._save_to_json(explanations, json_path)
        saved_files.append(json_path)

        # Optionally save explanations in HTML format for human-readable reports
        if output_format in ("html", "both"):
            html_path = self.output_dir /  f"{base_filename}.html"
            self._save_to_html(explanations, html_path)
            saved_files.append(html_path)
            print(f"LLM generated textual explanations for importance scores saved in {html_path} and {json_path}")
        return explanations, saved_files

    def _save_to_json(self, explanations, filepath):
        """
        Save the generated explanations to a JSON file with metadata.
        
        Args:
            explanations: List of explanation objects.
            filepath: Path to save the JSON file.
        """
        output_data = {
            "metadata": {
                "model": self.model_name,  # Include model name in metadata
                "num_explanations": len(explanations)  # Number of explanations generated
            },
            "explanations": explanations  # Actual explanations
        }
        # Write the data to a JSON file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

    def _save_to_html(self, explanations, filepath):
        """
        Save the generated explanations to an HTML file for easy visualization.
        
        Args:
            explanations: List of explanation objects.
            filepath: Path to save the HTML file.
        """
        explanation_divs = []
        for exp in explanations:
            # Get the full name of the explanation method
            full_name = self.explainer_names.get(exp['method'], exp['method'])
            # Create a styled HTML block for each explanation
            div = f"""
            <div class="explanation">
                <h3>{full_name}</h3>
                <span style='color:black'> Sentence: {exp['text']} </span><br>
                <span style='color:black'> Model Label: {exp['target']} </span>
                <pre>{exp['explanation'].replace('Interpretation of Saliency Scores', '')}</pre>
            </div>
            """
            explanation_divs.append(div)
        
        # Combine all explanation blocks into a complete HTML document
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Explanations Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; }}
                .explanation {{ 
                    background: #f8f9fa;
                    border-left: 4px solid #4e79a7;
                    padding: 15px;
                    margin-bottom: 20px;
                    border-radius: 4px;
                }}
                h3 {{ color: #2c3e50; margin-top: 0; }}
                .meta {{ color: #7f8c8d; font-size: 0.9em; }}
                pre {{ white-space: pre-wrap; }}
            </style>
        </head>
        <body>
            <h1>Model Explanations Report</h1>
            <div class="meta">
                Model: {self.model_name}<br>
            </div>
            <hr>
            {''.join(explanation_divs)}
        </body>
        </html>
        """
        # Write the HTML content to a file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)

    def display_explanations(self, explanations):
        """
        Display the generated explanations in a Jupyter notebook with formatting.
        
        Args:
            explanations: List of explanation objects.
        """
        for exp in explanations:
            # Get the full name of the explanation method
            full_name = self.explainer_names.get(exp['method'], exp['method'])
            # Clean up the explanation text
            clean_explanation = exp['explanation'].replace('Interpretation of Saliency Scores', '')
            # Display the explanation as an HTML block
            display(HTML(f"""
            <div style='
                padding: 12px;
                margin: 12px 0;
                border-radius: 4px;
                border-left: 4px solid #4e79a7;
                color:white;
            '>
                <h4 style='margin-top:0;color:white;'>
                    {full_name} <span style='color:white'></span>
                </h4>
                <span style='color:white'> Sentence: {exp['text']} </span><br>
                <span style='color:white'> Model Label: {exp['target']} </span>
                <pre style='white-space:pre-wrap;margin-bottom:0;'>{clean_explanation}</pre>
            </div>
            """))

    def generate_explanation(self, exps):
        """
        Generate textual explanations for a list or dictionary of explanation objects.
        
        Args:
            exps: List or dictionary of explanation objects.
            
        Returns:
            List of generated explanations with method, explanation text, target, and input text.
        """
        explanations = []

        # Handle case where exps is a dictionary of {method: [explanation_objects]}
        if isinstance(exps, dict):
            for method, exp_list in exps.items():
                for exp in exp_list:
                    prompt = self._create_prompt(exp)  # Create a method-specific prompt
                    try:
                        # Generate explanation using the LLM
                        response = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=500,
                            temperature=0.7,
                        )
                        explanation = response.choices[0].message.content.strip()
                        explanations.append({
                            "method": method,  # Use the dictionary key as method
                            "explanation": explanation,
                            "target": exp.target,  # Assuming exp has a target attribute
                            "text": exp.text
                        })
                    except Exception as e:
                        print(e)  # Log any errors during explanation generation
                        continue
        # Handle case where exps is a single explanation or list
        else:
            exp_list = [exps] if not isinstance(exps, list) else exps
            for exp in exp_list:
                prompt = self._create_prompt(exp)  # Create a method-specific prompt
                try:
                    # Generate explanation using the LLM
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=500,
                        temperature=0.7,
                    )
                    explanation = response.choices[0].message.content.strip()
                    explanations.append({
                        "method": exp.explainer,  # Use the explainer attribute
                        "explanation": explanation,
                        "target": exp.target,
                        "text": exp.text
                    })
                except Exception as e:
                    print(e)  # Log any errors during explanation generation
                    continue
        
        return explanations

    def _create_prompt(self, exp):
        """
        Create a prompt for generating method-specific explanations.
        
        Args:
            exp: Explanation object containing tokens, scores, and other attributes.
            
        Returns:
            String prompt for the LLM.
        """
        # Get the full name of the explanation method
        full_name = self.explainer_names.get(exp.explainer, exp.explainer)
        print(f"Generating LLM generated textual explanation for {full_name} scores...")
        # Construct the prompt with method-specific details
        prompt = f"""
        You are analyzing {full_name} explanations for a text classification model.
        The target class is '{exp.target}'.
        
        Given these token importance scores from {full_name}, explain:
        1. How are these scores calculated?
        2. How to interpret them with respect to the model's decision?

        Tokens: {exp.tokens}
        Scores: {exp.scores}

        Provide concise explanation specifically for {full_name} method:
        """
        return prompt

