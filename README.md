# Credit Card Fraud Detection API

This project provides a FastAPI-based web service for credit card fraud detection using machine learning models. The service loads pre-trained XGBoost and Isolation Forest models and exposes endpoints for real-time inference.

## Setup Instructions


1. **Clone the repository and navigate to the project directory:**
   ```bash
   git clone <your-repo-url>
   cd <your-project-directory>
   ```

2. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure the following model files are present in the project directory:**
   - `xgb_model_smote.pkl` (XGBoost model trained with SMOTE and Isolation Forest score)
   - `iso_forest.pkl` (Isolation Forest model)

4. **Run the FastAPI application:**
   ```bash
   python app.py
   ```

5. **Access the API:**
   - The API will be available at: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
   - Interactive API docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) #swagger docs

6. **if you want to try notebook**
   -run download.py 
   -dataset will get downloaded with this script


## API Endpoints

### Health Check
- `GET /`
  - Returns a message confirming the API is running.

### Single Prediction
- `POST /predict`
  - Request body: `{ "features": [<list of feature values>] }`
  - Response: `{ "prediction": 0 or 1, "probability": float, "iso_score": float }`


---

# credit_fraud.ipynb Description

1. **Base Model: Random Forest**
   - A baseline model using Random Forest was trained for fraud detection.

2. **XGBoost with Isolation Forest Score**
   - An XGBoost model was trained using the original features plus an anomaly score from Isolation Forest as an additional feature.

3. **XGBoost with Isolation Forest + SMOTE**
   - The best results were achieved by combining XGBoost, Isolation Forest score, and SMOTE for class balancing.
   - This approach achieved a recall of **91%** and reduced false negatives by around **60%** compared to the baseline. 