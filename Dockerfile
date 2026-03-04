# 1. Utiliser une image Python officielle légère
FROM python:3.12-slim

# 2. Installation des dépendances système nécessaires pour psycopg2 (PostgreSQL)
# Sans 'gcc' et 'libpq-dev', l'installation de SQLAlchemy/psycopg2 échouera
USER root
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 3. Configuration de l'utilisateur Hugging Face (UID 1000)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /app

# 4. Installation de Poetry en tant qu'utilisateur
RUN pip install --user --no-cache-dir poetry

# 5. Copier les fichiers de dépendances
COPY --chown=user:user pyproject.toml poetry.lock* ./

# 6. CONFIGURATION CRUCIALE POUR LES PERMISSIONS
# On dit à Poetry de ne pas créer de venv et on force l'installation 
# dans le dossier utilisateur avec 'pip install' piloté par Poetry
RUN poetry config virtualenvs.create false \
    && poetry install --no-cache-dir --without dev,test --no-root

# 7. Copier le reste du code (src, models, etc.)
COPY --chown=user:user . .

# 8. Port exposé par Hugging Face
EXPOSE 7860

# 9. Lancement de l'application
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "7860"]
