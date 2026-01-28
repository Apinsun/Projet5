from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI()

# 1. Définition des données attendues (Pydantic)
class PredictionInput(BaseModel):
    feature_1: float
    feature_2: float

# 2. Chargement du modèle (Global)
# On essaie de charger le vrai modèle, sinon on met None (pour ne pas crasher au démarrage)
model = None
MODEL_PATH = "models/model.pkl"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)

# 3. Endpoint de prédiction
@app.post("/predict")
def predict(input_data: PredictionInput):
    global model
    
    # Sécurité : Si le modèle n'est pas là (et qu'on n'est pas en test mocké)
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non disponible (Fichier manquant)")

    # Conversion Pydantic -> DataFrame
    data_df = pd.DataFrame([input_data.model_dump()])
    
    # Prédiction
    try:
        prediction = model.predict(data_df)[0]
        return {"prediction": float(prediction)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))