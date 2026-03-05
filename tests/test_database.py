import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import sys
import os
from datetime import datetime, timezone

# Ajoute le dossier parent au path pour trouver 'src'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# On importe l'objet Base (qui contient la structure des tables) et ta classe PredictionLog
from src.app import Base, PredictionLog

# --- CONFIGURATION DE LA BDD DE TEST (EN MÉMOIRE) ---
# SQLite en mémoire : ultra rapide et n'écrit rien sur le disque dur
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture()
def db_session():
    """Fixture Pytest : Crée les tables, fournit une session propre, puis nettoie."""
    # Création des tables dans la base de données SQLite en mémoire
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)

def test_database_insertion(db_session):
    """Test vérifiant la bonne insertion d'un log dans le SGDB."""
    
    # 1. On prépare une fausse ligne de log correspondant exactement à ton modèle app.py
    fake_log = PredictionLog(
        input_data={"revenu_mensuel": 4500, "poste": "Tech Lead"}, # Simule le JSON de Pydantic
        prediction_result=1,
        prediction_probability=0.85,
        timestamp=datetime.now(timezone.utc)
    )
    
    # 2. On exécute la requête d'insertion (Test de la fonctionnalité d'ajout)
    db_session.add(fake_log)
    db_session.commit()
    
    # 3. On interroge la base pour récupérer le premier (et unique) enregistrement
    inserted_log = db_session.query(PredictionLog).first()
    
    # 4. Assertions (Vérification de l'intégrité des données)
    assert inserted_log is not None, "L'enregistrement n'a pas été inséré dans la base."
    assert inserted_log.prediction_result == 1
    assert inserted_log.prediction_probability == 0.85
    assert inserted_log.input_data["poste"] == "Tech Lead"

def test_database_deletion(db_session):
    """Test vérifiant la suppression d'un log (Optionnel mais apprécié par les jurys)."""
    # Insertion
    fake_log = PredictionLog(prediction_result=0, prediction_probability=0.10)
    db_session.add(fake_log)
    db_session.commit()
    
    # Suppression
    log_to_delete = db_session.query(PredictionLog).first()
    db_session.delete(log_to_delete)
    db_session.commit()
    
    # Vérification
    empty_result = db_session.query(PredictionLog).first()
    assert empty_result is None, "La donnée n'a pas été supprimée correctement."
