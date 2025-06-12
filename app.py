from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from typing import List
import os

app = FastAPI(title="Credit Card Fraud Detection API")



class TransactionData(BaseModel):
    features: List[float]

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    iso_score: int

@app.get("/")
async def root():
    return {"message": "Credit Card Fraud Detection API is running"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(transaction: TransactionData):
    """
    Make prediction for a single transaction
    """
    xgb_model = joblib.load('xgb_model_smote_iso.pkl')
        
    iso_forest = joblib.load('iso_forest_model.pkl')
    if xgb_model is None or iso_forest is None:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    try:
        features = np.array(transaction.features).reshape(1, -1)
        
   
        iso_score = -iso_forest.decision_function(features) 
        
        features_with_iso = np.column_stack((features, iso_score))
        
        
        # Make prediction
        prediction = xgb_model.predict(features_with_iso)[0]
        probability = xgb_model.predict_proba(features_with_iso)[0][1]
        
        return PredictionResponse(
            prediction=int(prediction),
            probability=float(probability),
            iso_score=int(iso_score[0])
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error making prediction: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 