from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import sys
import os

# --- 1. Gestion des chemins pour les imports (Indispensable pour le CI/CD) ---
# Ajoute le dossier parent au path pour trouver 'src'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.app import app

client = TestClient(app)

# --- FAUX MODÈLE POUR LES TESTS ---
# --- Le Mock du Modèle (Doit imiter ModelWithThreshold) ---
class DummyModel:
    def __init__(self):
        # L'API lit cet attribut dans le return, il doit exister !
        self.threshold = 0.5

    def predict(self, X):
        # L'API fait [0], donc on renvoie une liste
        return [0] 

    def predict_proba(self, X):
        # L'API fait [0][1], donc on renvoie une liste de listes
        # [Proba_classe_0, Proba_classe_1]
        return [[0.8, 0.2]]

# --- LE TEST ---
def test_predict_endpoint_works():
    # On remplace le vrai modèle par notre DummyModel pour le test
    # Cela évite d'avoir besoin du fichier .pkl ou des classes scikit-learn
    with patch("src.app.model", DummyModel()):
        
        # Payload valide (copié de ta class Config)
        # Il doit passer la validation Pydantic STRICTEMENT
        payload = {
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
        
        # Appel de l'API
        response = client.post("/predict", json=payload)
        
        # Debug : Affiche l'erreur si ce n'est pas 200
        if response.status_code != 200:
            print("Erreur API:", response.json())

        # Vérifications
        assert response.status_code == 200
        
        json_response = response.json()
        
        # On vérifie que les clés attendues sont là
        assert "prediction" in json_response
        assert "probabilite_depart" in json_response
        assert "seuil_utilise" in json_response
        
        # On vérifie les valeurs renvoyées par le DummyModel
        assert json_response["prediction"] == 0
        assert json_response["seuil_utilise"] == 0.5

def test_predict_validation_error_types():
    """Test qui échoue car le type de donnée est mauvais (str au lieu de int)"""
    # Pas besoin de 'with patch' ici !
    
    payload = {
        # ERREUR ICI : On envoie du texte alors qu'un int est attendu
        "revenu_mensuel": "cinq mille", 
        
        # Même si le reste est bon, ça doit planter
        "satisfaction_employee_environnement": 3,
        "genre": "M"
        # ... (on ne met pas tout, car Pydantic va de toute façon
        # raller qu'il manque des champs OU que le type est mauvais)
    }
    
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 422
    # On peut vérifier que l'erreur concerne bien le revenu ou le type
    assert "revenu_mensuel" in str(response.json()) or "missing" in str(response.json())

def test_predict_validation_error_enum():
    """Test qui échoue car la valeur n'est pas dans la liste autorisée (Enum)"""
    
    payload = {
        # On remplit un peu pour la forme
        "revenu_mensuel": 4000,
        "satisfaction_employee_environnement": 3,
        
        # ERREUR ICI : "Boulanger" n'est pas dans ton PosteEnum
        "poste": "Boulanger" 
    }
    
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 422
    # Le message d'erreur contiendra "Input should be 'Cadre Commercial', 'Assistant de Direction'..."
    response_text = str(response.json())
    assert "poste" in response_text