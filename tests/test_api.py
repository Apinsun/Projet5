from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from src.app import app

client = TestClient(app)

# --- FAUX MODÈLE POUR LES TESTS ---
class DummyModel:
    def predict(self, X):
        # Renvoie toujours 1, peu importe l'input
        return [1.0]

# --- LE TEST ---
def test_predict_endpoint_works():
    # C'est ici qu'on "injecte" le faux modèle dans l'application
    # On remplace la variable 'model' de src.app par notre DummyModel
    with patch("src.app.model", DummyModel()):
        
        # Données de test
        payload = {"feature_1": 10.5, "feature_2": 2.0}
        
        # On appelle l'API (qui utilise le faux modèle sans le savoir)
        response = client.post("/predict", json=payload)
        
        # Vérifications
        assert response.status_code == 200
        assert response.json() == {"prediction": 1.0}

def test_predict_validation_error():
    # Test sans mocker le modèle (car Pydantic bloque AVANT le modèle)
    # On envoie du texte au lieu d'un nombre
    payload = {"feature_1": "pas un nombre", "feature_2": 2.0}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422