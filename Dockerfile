# 1. Image de base
FROM python:3.12-slim

# 2. INSTALLATION DES DÉPENDANCES SYSTÈME (En root, au début)
# On installe gcc et les libs postgres pour que psycopg2 et greenlet puissent compiler
USER root
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 3. CONFIGURATION UTILISATEUR HUGGING FACE
RUN useradd -m -u 1000 user
WORKDIR /app

# 4. INSTALLATION DE POETRY
# On l'installe globalement pour simplifier
RUN pip install --no-cache-dir poetry

# 5. COPIE DES FICHIERS DE DÉPENDANCES
COPY pyproject.toml poetry.lock* ./

# 6. INSTALLATION DES LIBS (Sans venv pour éviter les conflits de dossiers)
RUN poetry config virtualenvs.create false \
    && poetry install --without dev,test --no-root

# 7. PASSAGE À L'UTILISATEUR NON-ROOT (Sécurité HF)
RUN chown -R user:user /app
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# 8. COPIE DU RESTE DU CODE
COPY --chown=user:user . .

# 9. PORT ET LANCEMENT
EXPOSE 7860
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "7860"]
