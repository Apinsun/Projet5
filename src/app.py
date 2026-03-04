from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
import joblib
import pandas as pd
import os
from pathlib import Path
import sys  # <--- Ajoute cet import

from sqlalchemy import create_engine, Column, Integer, Float, DateTime, JSON
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from fastapi import Depends
from dotenv import load_dotenv
from datetime import datetime, timezone


# --- AJOUT CRUCIAL POUR LE CI/CD ---
# Cela dit à Python : "Regarde aussi dans le dossier où se trouve ce fichier (app.py)"
# Peu importe d'où on lance la commande (root, tests, etc.), il trouvera les voisins.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
app = FastAPI()

# --- CONFIGURATION BDD ---
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

# On prépare SQLAlchemy seulement si on a une URL (évite de faire planter les tests sans .env)
if DATABASE_URL:
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()

    # Définition de notre table de logs
    class PredictionLog(Base):
        __tablename__ = "prediction_logs"
        id = Column(Integer, primary_key=True, index=True)
        timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
        input_data = Column(JSON)
        prediction_result = Column(Integer)
        prediction_probability = Column(Float)

# Fonction pour fournir une session BDD à chaque requête (Injection de dépendance)
def get_db():
    if not DATABASE_URL:
        yield None
        return
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

try:
    # Maintenant ça marchera même depuis les tests GitHub
    from model_With_Threshold import ModelWithThreshold
    from preprocessing import DataCleaningTransformer, FeatureEngineeringTransformer, ColumnDropperTransformer, SalaryFeatureEngineering
except ImportError as e:
    print(f"ERREUR D'IMPORT : {e}")

from enum import Enum
from typing import Literal

# Hérite de str et Enum pour que Pydantic comprenne que ce sont des chaînes
class PosteEnum(str, Enum):
    CADRE_COMMERCIAL = "Cadre Commercial"
    ASSISTANT_DIRECTION = "Assistant de Direction"
    CONSULTANT = "Consultant"
    TECH_LEAD = "Tech Lead"
    MANAGER = "Manager"
    SENIOR_MANAGER = "Senior Manager"
    REPRESENTANT_COMMERCIAL = "Représentant Commercial"
    DIRECTEUR_TECHNIQUE = "Directeur Technique"
    RH = "Ressources Humaines"

# --- Définition du format attendu (Schéma) ---
class EmployeeInput(BaseModel):
    # Features numériques (Int)
    satisfaction_employee_environnement: int = Field(..., ge=0, le=4, description="Note de 0 à 4")
    note_evaluation_precedente: int = Field(..., ge=0, le=4, description="Note de 0 à 4")
    satisfaction_employee_nature_travail: int = Field(..., ge=0, le=4, description="Note de 0 à 4")
    satisfaction_employee_equipe: int = Field(..., ge=0, le=4, description="Note de 0 à 4")
    satisfaction_employee_equilibre_pro_perso: int = Field(..., ge=0, le=4, description="Note de 0 à 4")
    revenu_mensuel: int = Field(..., gt=0, description="Le revenu doit être positif")
    nombre_experiences_precedentes: int = Field(..., ge=0, description="Nombre d'expériences précédentes")
    annee_experience_totale: int = Field(..., ge=0, description="Années d'expérience totale")
    annees_dans_l_entreprise: int = Field(..., ge=0, description="Années dans l'entreprise")
    annees_dans_le_poste_actuel: int = Field(..., ge=0, description="Années dans le poste actuel")
    nombre_participation_pee: int = Field(..., ge=0, description="Nombre de participations au PEE")
    nb_formations_suivies: int = Field(..., gt=0, description="Le nombre de formations doit être positif")
    distance_domicile_travail: int = Field(..., gt=0, description="La distance doit être positive")
    niveau_education: int = Field(..., ge=0, description="Niveau d'éducation entre 0 et 5")
    annees_depuis_la_derniere_promotion: int = Field(..., ge=0, description="Années depuis la dernière promotion")
    annes_sous_responsable_actuel: int = Field(..., ge=0, description="Années sous responsable actuel")

    # Features catégorielles ou brutes (Str)
    heure_supplementaires: Literal["Yes", "No"]

    # ^     : Début de la chaine
    # \d+   : Un ou plusieurs chiffres (0-9)
    # %     : Le caractère % littéral
    # $     : Fin de la chaine
    augementation_salaire_precedente: str = Field(
        ..., 
        pattern=r"^\d+%$", 
        description="Pourcentage (ex: '15%')"
    )

    genre: Literal['M','F']
    statut_marital: Literal['Marié(e)','Célibataire','Divorcé(e)']
    departement: Literal['Consulting','Commercial','Ressources Humaines']
    poste: PosteEnum
    domaine_etude: Literal['Infra & Cloud','Transformation Digitale','Marketing','Entrepreunariat','Autre','Ressources Humaines']
    frequence_deplacement: Literal['Occasionnel','Frequent','Aucun']

# Configuration pour la documentation automatique (Swagger UI)
    model_config = ConfigDict(
        json_schema_extra = {
            "example": {
                # --- Notes (doivent être <= 4) ---
                "satisfaction_employee_environnement": 3,
                "note_evaluation_precedente": 3,
                "satisfaction_employee_nature_travail": 4,
                "satisfaction_employee_equipe": 3,
                "satisfaction_employee_equilibre_pro_perso": 2,

                # --- Données Financières & Carrière ---
                "revenu_mensuel": 4500,
                "augementation_salaire_precedente": "12%", # Respecte le regex (chiffres + %)
                "nombre_participation_pee": 1,
                
                # --- Expérience (Cohérence temporelle) ---
                "annee_experience_totale": 10,
                "nombre_experiences_precedentes": 2,
                "annees_dans_l_entreprise": 5,
                "annees_dans_le_poste_actuel": 2,
                "annees_depuis_la_derniere_promotion": 1,
                "annes_sous_responsable_actuel": 2,

                # --- Profil ---
                "niveau_education": 3,
                "nb_formations_suivies": 2,
                "distance_domicile_travail": 15,
                "heure_supplementaires": "Yes",

                # --- Catégories (Doivent matcher EXACTEMENT tes Literals/Enums) ---
                "genre": "M",
                "statut_marital": "Marié(e)",
                "departement": "Consulting",
                "poste": "Tech Lead",
                "domaine_etude": "Infra & Cloud",
                "frequence_deplacement": "Occasionnel"
            }
        }
    )

# 2. Chargement du modèle (Global)
# On essaie de charger le vrai modèle, sinon on met None (pour ne pas crasher au démarrage)
model = None
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
models_dir = project_root / "models"
MODEL_PATH = models_dir / "model.pkl"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)

@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API de prédiction de turnover ! 🚀 L'API est en ligne."}

# 3. Endpoint de prédiction
@app.post("/predict")
def predict(input_data: EmployeeInput, db: Session = Depends(get_db)):
    global model
    
    # Sécurité 1 : Le modèle est-il chargé ?
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non disponible (Fichier manquant)")

    # Sécurité 2 : Faire la prédiction (Si une donnée échappe à Pydantic et fait planter le modèle)
    data_df = pd.DataFrame([input_data.model_dump()])
    try:
        prediction = int(model.predict(data_df)[0])
        probabilite = float(model.predict_proba(data_df)[0][1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne du modèle lors de la prédiction : {str(e)}")

    # Sécurité 3 : Sauvegarde dans la BDD
    # Le projet exige une traçabilité totale, donc si la BDD plante, on le signale en erreur 500
    if db is not None:
        try:
            log_entry = PredictionLog(
                input_data=input_data.model_dump(),
                prediction_result=prediction,
                prediction_probability=probabilite
            )
            db.add(log_entry)
            db.commit() # On valide l'écriture
        except Exception as e:
            db.rollback() # En cas d'erreur, on annule pour ne pas corrompre la BDD
            print(f"Erreur BDD: {str(e)}")
            raise HTTPException(status_code=500, detail="La prédiction a réussi mais l'enregistrement en BDD a échoué.")

    # 4. Retour du résultat final
    return {
        "prediction": prediction,
        "probabilite_depart": probabilite,
        "seuil_utilise": float(model.threshold) if hasattr(model, 'threshold') else 0.5
    }
