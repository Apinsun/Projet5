from pathlib import Path
import pandas as pd
from datetime import datetime, timezone
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.orm import declarative_base, sessionmaker

# --- 1. CONFIGURATION DE LA CONNEXION ---
# On charge les variables du fichier .env
load_dotenv()

# On récupère l'URL de manière sécurisée
DATABASE_URL = os.getenv("DATABASE_URL")

# Petite sécurité : si on a oublié le fichier .env, le script s'arrête en l'expliquant
if not DATABASE_URL:
    raise ValueError("⚠️ ERREUR : La variable DATABASE_URL est introuvable. As-tu bien créé ton fichier .env ?")

# Initialisation du moteur SQLAlchemy
engine = create_engine(DATABASE_URL)
Base = declarative_base()

# --- 2. DÉFINITION DU SCHÉMA (LES TABLES) ---

class TrainingData(Base):
    __tablename__ = "training_dataset"
    
    # On laisse PostgreSQL créer un ID unique automatique
    id = Column(Integer, primary_key=True, index=True)
    
    # Colonnes numériques
    satisfaction_employee_environnement = Column(Integer)
    note_evaluation_precedente = Column(Integer)
    satisfaction_employee_nature_travail = Column(Integer)
    satisfaction_employee_equipe = Column(Integer)
    satisfaction_employee_equilibre_pro_perso = Column(Integer)
    revenu_mensuel = Column(Integer)
    nombre_experiences_precedentes = Column(Integer)
    annee_experience_totale = Column(Integer)
    annees_dans_l_entreprise = Column(Integer)
    annees_dans_le_poste_actuel = Column(Integer)
    nombre_participation_pee = Column(Integer)
    nb_formations_suivies = Column(Integer)
    distance_domicile_travail = Column(Integer)
    niveau_education = Column(Integer)
    annees_depuis_la_derniere_promotion = Column(Integer)
    annes_sous_responsable_actuel = Column(Integer)
    
    # Colonnes de type texte
    heure_supplementaires = Column(String)
    augementation_salaire_precedente = Column(String)
    genre = Column(String)
    statut_marital = Column(String)
    departement = Column(String)
    poste = Column(String)
    domaine_etude = Column(String)
    frequence_deplacement = Column(String)
    
    # Colonne cible (Target)
    a_quitte_l_entreprise = Column(String)

class PredictionLog(Base):
    __tablename__ = "prediction_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    # On stocke tout le dictionnaire Pydantic ici !
    input_data = Column(JSON) 
    prediction_result = Column(Integer)      # 0 ou 1
    prediction_probability = Column(Float)   # ex: 0.85 (85% de certitude)


# --- 3. EXÉCUTION DU SCRIPT ---
if __name__ == "__main__":

    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    models_dir = project_root / "models"
    artifacts_dir = project_root / "artifacts"


    print("⏳ Création des tables dans la base de données...")
    # Cette ligne crée les tables si elles n'existent pas encore
    Base.metadata.create_all(bind=engine)
    print("✅ Tables créées avec succès !")

    print("⏳ Lecture du fichier CSV...")
    # On lit ton fichier nettoyé
    df = pd.read_csv(artifacts_dir / "df_final.csv")
    
    print(f"⏳ Insertion de {len(df)} lignes dans la table 'training_dataset'...")
    # L'astuce magique de Pandas + SQLAlchemy : on envoie tout d'un coup !
    # 'append' permet d'ajouter à la table existante, 'if_exists' gère les cas où la table existe déjà.
    # index=False pour ne pas enregistrer l'index 0, 1, 2 du dataframe comme colonne
    df.to_sql(name="training_dataset", con=engine, if_exists="append", index=False)
    
    print("🚀 TERMINÉ ! Le dataset complet est sécurisé dans le Cloud PostgreSQL !")