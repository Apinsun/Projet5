import pandas as pd
import joblib
import os
from pathlib import Path
import sys
from sklearn.metrics import f1_score

# Chemins relatifs
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
MODEL_PATH = project_root / "models" / "model.pkl"
SAMPLE_PATH = project_root / "tests" / "test_sample.csv"
sys.path.append(str(project_root / "src"))
def test_model_performance_on_sample():
    """Vérifie que le modèle conserve de bonnes performances sur l'échantillon de test"""
    # 1. Vérifier que les fichiers existent
    assert os.path.exists(MODEL_PATH), "Le modèle model.pkl est introuvable."
    assert os.path.exists(SAMPLE_PATH), "L'échantillon test_sample.csv est introuvable."
    
    # 2. Charger le modèle et les données
    model = joblib.load(MODEL_PATH)
    df_sample = pd.read_csv(SAMPLE_PATH)
    
    # 3. Séparer X et y
    X_sample = df_sample.drop(columns=['a_quitte_l_entreprise'])
    y_true = df_sample['a_quitte_l_entreprise']
    
    # 4. Faire les prédictions
    y_pred = model.predict(X_sample)
    
    # 5. Calculer le F1-score
    score = f1_score(y_true, y_pred)
    
    # 6. L'assertion critique : Le score doit rester au-dessus d'un seuil acceptable (ex: 0.60)
    # Si quelqu'un casse le modèle, ce test échouera et bloquera le déploiement
    print(f"F1-Score sur l'échantillon : {score:.4f}")
    assert score > 0.45, f"Performance dégradée ! F1-score de {score} inférieur à 0.45"
