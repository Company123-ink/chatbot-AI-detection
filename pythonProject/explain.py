import shap
import numpy as np
from transformers import pipeline

# Load model once
explainer_model = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# Create SHAP explainer ONCE (faster)
explainer = shap.Explainer(
    explainer_model,
    algorithm="permutation"   # MUCH faster than default
)

def explain_text(text: str):
    """
    Generate word-level explanations using SHAP.
    """
    shap_values = explainer([text])

    # Extract tokens and importance values
    tokens = shap_values.data[0]
    values = shap_values.values[0]

    explanations = []
    for token, value in zip(tokens, values):
        scalar_value = float(np.array(value).mean())
        explanations.append((token, scalar_value))

    return explanations

