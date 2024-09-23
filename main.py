from fastapi import FastAPI, Form, HTTPException
from pydantic import BaseModel
from typing import Dict, List
from utils import predict_emotion  

# Initialize an app
app = FastAPI(debug=True)

# Pydantic Response model
class EmotionsResponse(BaseModel):
    final_pred: Dict
    all_preds: List[Dict]

@app.post('/predict_emotions', response_model=EmotionsResponse)
async def predict_emotions(text: str = Form(...)):
    """Predict the emotions from the input text.

    Args:
        text (str): The input text for emotion classification.

    Returns:
        EmotionsResponse: A response containing the final and all predictions.
    """
    try:
        # Call the function from utils.py
        final_pred, all_preds = predict_emotion(text=text)

        # The response
        response = EmotionsResponse(final_pred=final_pred, all_preds=all_preds)

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f'An error occurred: {str(e)}')
