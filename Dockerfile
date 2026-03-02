# 1. Utiliser une image Python officielle, légère et correspondante à notre projet
FROM python:3.12-slim

# 2. Variables d'environnement pour optimiser Python et Poetry
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1

# 3. Spécifique à Hugging Face : Créer un utilisateur non-root (UID 1000)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# 4. Définir le répertoire de travail dans le conteneur
WORKDIR /app

# 5. Installer Poetry (uniquement pour l'utilisateur courant)
RUN pip install --user --no-cache-dir poetry

# 6. Copier les fichiers de dépendances EN PREMIER
# (Cela permet à Docker de mettre en cache cette étape si les dépendances ne changent pas)
COPY --chown=user:user pyproject.toml poetry.lock ./

# 7. Installer UNIQUEMENT les dépendances de production (API)
# On exclut Ydata-profiling, pytest, etc., pour avoir une image toute légère !
RUN poetry install --without dev,test --no-root

# 8. Copier le code source et le modèle entraîné
COPY --chown=user:user src/ ./src/
COPY --chown=user:user models/ ./models/

# 9. Exposer le port requis par Hugging Face
EXPOSE 7860

# 10. Commande pour démarrer l'API
# On suppose que ton fichier s'appelle src/api.py et que l'instance FastAPI s'appelle 'app'
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "7860"]
