from transformers import pipeline

# Load sentiment analysis model (used as proxy for depression detection)
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

def predict_depression(text: str):
    """
    Predict depressive indicators from text.
    """
    result = classifier(text)[0]

    label = result["label"]
    score = result["score"]

    if label == "NEGATIVE":
        prediction = "Depressive Indicators Detected"
    else:
        prediction = "No Strong Depressive Indicators Detected"

    return prediction, round(score, 3)
