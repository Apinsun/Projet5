from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import pytest
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

# Profil "Fidèle" : Salaire énorme, très heureux, promu récemment
LOYAL_PAYLOAD = VALID_PAYLOAD.copy()
LOYAL_PAYLOAD.update({
    "revenu_mensuel": 15000,
    "satisfaction_employee_environnement": 4,
    "satisfaction_employee_nature_travail": 4,
    "annees_depuis_la_derniere_promotion": 0
})

# Cas Limite Extrême : Le nouvel embauché (0 expérience, 0 ancienneté)
NEW_HIRE_PAYLOAD = VALID_PAYLOAD.copy()
NEW_HIRE_PAYLOAD.update({
    "annee_experience_totale": 0,
    "nombre_experiences_precedentes": 0,
    "annees_dans_l_entreprise": 0,
    "annees_dans_le_poste_actuel": 0,
    "annees_depuis_la_derniere_promotion": 0,
    "annes_sous_responsable_actuel": 0
})

# --- LES TESTS ---

def test_read_root():
    """Test 0 : Vérifier que l'API est bien en ligne sur la racine"""
    response = client.get("/")
    
    # On vérifie que la requête réussit (Code 200)
    assert response.status_code == 200
    
    # On vérifie que le message de bienvenue est bien là
    json_response = response.json()
    assert "message" in json_response
    assert "Bienvenue sur l'API" in json_response["message"]

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

@pytest.mark.parametrize("payload", [VALID_PAYLOAD, LOYAL_PAYLOAD, NEW_HIRE_PAYLOAD])
def test_functional_real_model_prediction(payload):
    """Test 5 (Fonctionnel) : Tester le vrai pipeline ML avec différents profils"""
    # 1. On s'assure que le modèle est bien chargé par l'application
    from src.app import model
    
    # Si le fichier model.pkl n'est pas là, on saute le test
    if model is None:
        pytest.skip("Modèle non trouvé en local, test fonctionnel ignoré.")

    # 2. On lance la requête avec le profil passé en paramètre
    response = client.post("/predict", json=payload)
    
    # 3. Vérifications
    assert response.status_code == 200
    json_response = response.json()
    
    assert "prediction" in json_response
    assert json_response["prediction"] in [0, 1] 
    assert 0.0 <= json_response["probabilite_depart"] <= 1.0
