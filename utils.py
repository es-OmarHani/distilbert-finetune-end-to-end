from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

# Use a pipeline as a high-level helper with your model checkpoint
model_chpt = '0marr/distilbert-base-multilingual-cased-finetuned'
classifier = pipeline('text-classification', model=model_chpt)

# The labels and their mapping
labels_map = {
    'LABEL_0': 'none',
    'LABEL_1': 'anger',
    'LABEL_2': 'joy',
    'LABEL_3': 'sadness',
    'LABEL_4': 'love',
    'LABEL_5': 'sympathy',
    'LABEL_6': 'surprise',
    'LABEL_7': 'fear'
}

def map_labels_to_emotions(preds: List[Dict]) -> List[Dict]:
    """Map the prediction labels to their corresponding emotion names.

    Args:
        preds (List[Dict]): A list of prediction dictionaries.

    Returns:
        List[Dict]: A list of dictionaries with emotions and their scores.
    """
    mapped_preds = []
    for pred in preds:
        label = pred['label']
        emotion = labels_map.get(label)
        if emotion:
            mapped_preds.append({'emotion': emotion, 'score': round(pred['score'], 4)})
    return mapped_preds

def predict_emotion(text: str) -> Tuple[Dict, List[Dict]]:
    """Classify the emotion of the provided text.

    Args:
        text (str): The input text to classify.

    Returns:
        Tuple[Dict, List[Dict]]: The final prediction and all mapped predictions.
    """
    all_preds = classifier(text, top_k=None)

    # Map predictions to emotions
    mapped_preds = map_labels_to_emotions(preds=all_preds)

    # The final prediction (sorted)
    final_pred = mapped_preds[0]

    return final_pred, mapped_preds

def plot_preds(preds: List[Dict]) -> plt.Figure:
    """Create a bar plot of emotion scores from the predictions.

    Args:
        preds (List[Dict]): A list of predicted emotions and their scores.

    Returns:
        plt.Figure: A Matplotlib figure containing the bar plot.
    """
    # Plotting the class probabilities
    emotions = [item['emotion'] for item in preds]
    scores = [item['score'] for item in preds]

    # Create a Matplotlib figure
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(emotions, scores, color='skyblue')
    ax.set_xlabel('Emotions')
    ax.set_ylabel('Scores')
    ax.set_title('Emotion Scores')
    ax.set_xticklabels(emotions, rotation=45)
    plt.tight_layout()

    return fig  # Return the figure
