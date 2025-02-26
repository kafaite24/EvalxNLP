from setuptools import setup, find_packages
import os

# Read the README file if it exists
current_directory = os.path.dirname(os.path.abspath(__file__))
readme_path = os.path.join(current_directory, "README.md")
try:
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "NLP Explainability Benchmarking Framework"

setup(
    name="EvalXAI",
    version="0.1.0",
    description="NLP Explainability Benchmarking Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Kafaite Zahra Hussain",
    author_email="kafait.e.zahra@gmail.com",
    url="https://github.com/kafaite24/EvalXAI",
    packages=find_packages(include=["evalxai", "evalxai.*"]),  # Explicitly include subpackages
    install_requires=[
        "torch",
        "transformers",
        "captum",
        "scikit-learn",
        "matplotlib",
        "datasets",
        "numpy",
        "pandas",
        "seaborn",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
