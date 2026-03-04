from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import sys
import os

# Ajoute le dossier parent au path pour trouver 'src'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# On importe l'app ET la fonction get_db qu'on vient de créer
from src.app import app, get_db

client = TestClient(app)

# --- MOCK DE LA BASE DE DONNÉES ---
# On crée une fausse session BDD qui ne fait rien (MagicMock)
def override_get_db():
    db_mock = MagicMock()
    yield db_mock

# On dit à FastAPI d'utiliser notre faux get_db pendant les tests
app.dependency_overrides[get_db] = override_get_db


# --- FAUX MODÈLE POUR LES TESTS ---
class DummyModel:
    def __init__(self):
        self.threshold = 0.5

    def predict(self, X):
        return [0] 

    def predict_proba(self, X):
        return [[0.8, 0.2]]


# Payload valide de test
VALID_PAYLOAD = {
    "satisfaction_employee_environnement": 3,
    "note_evaluation_precedente": 3,
    "satisfaction_employee_nature_travail": 4,
    "satisfaction_employee_equipe": 3,
    "satisfaction_employee_equilibre_pro_perso": 2,
    "revenu_mensuel": 4500,
    "augementation_salaire_precedente": "12%",
    "nombre_participation_pee": 1,
    "annee_experience_totale": 10,
    "nombre_experiences_precedentes": 2,
    "annees_dans_l_entreprise": 5,
    "annees_dans_le_poste_actuel": 2,
    "annees_depuis_la_derniere_promotion": 1,
    "annes_sous_responsable_actuel": 2,
    "niveau_education": 3,
    "nb_formations_suivies": 2,
    "distance_domicile_travail": 15,
    "heure_supplementaires": "Yes",
    "genre": "M",
    "statut_marital": "Marié(e)",
    "departement": "Consulting",
    "poste": "Tech Lead",
    "domaine_etude": "Infra & Cloud",
    "frequence_deplacement": "Occasionnel"
}

# --- LES TESTS ---

def test_predict_endpoint_works():
    """Test 1 : Tout se passe bien (Prédiction + Mock BDD)"""
    with patch("src.app.model", DummyModel()):
        response = client.post("/predict", json=VALID_PAYLOAD)
        
        assert response.status_code == 200
        json_response = response.json()
        assert "prediction" in json_response
        assert json_response["prediction"] == 0

def test_predict_validation_error_types():
    """Test 2 : Rejet par Pydantic (Mauvais type de donnée)"""
    bad_payload = VALID_PAYLOAD.copy()
    bad_payload["revenu_mensuel"] = "cinq mille" # Erreur provoquée
    
    response = client.post("/predict", json=bad_payload)
    
    assert response.status_code == 422 # 422 Unprocessable Entity
    assert "revenu_mensuel" in str(response.json())

def test_predict_validation_error_enum():
    """Test 3 : Rejet par Pydantic (Valeur Enum non autorisée)"""
    bad_payload = VALID_PAYLOAD.copy()
    bad_payload["poste"] = "Boulanger" # Erreur provoquée
    
    response = client.post("/predict", json=bad_payload)
    
    assert response.status_code == 422

def test_predict_model_failure():
    """Test 4 : Le modèle plante de manière inattendue"""
    class BuggyModel:
        def predict(self, X):
            raise ValueError("Le modèle a explosé !")
            
    with patch("src.app.model", BuggyModel()):
        response = client.post("/predict", json=VALID_PAYLOAD)
        
        # On s'attend à notre erreur 500 configurée dans le bloc 'except'
        assert response.status_code == 500
        assert "Erreur interne du modèle" in response.json()["detail"]
