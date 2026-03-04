FROM python:3.12-slim

# 1. On installe juste le strict nécessaire système
RUN apt-get update && apt-get install -y gcc libpq-dev && rm -rf /var/lib/apt/lists/*

# 2. On crée l'utilisateur et on bascule DÉJÀ en utilisateur (plus de droits root !)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# 3. On travaille dans son dossier
WORKDIR /app

# 4. On installe Poetry et on installe les libs DIRECTEMENT dans le dossier user
# Comme on est déjà l'utilisateur 'user', tout va s'installer dans /home/user/.local/
RUN pip install --user --no-cache-dir poetry \
    && poetry config virtualenvs.create false \
    && poetry install --without dev,test --no-root

# 5. On copie le code
COPY --chown=user:user . .

# 6. On lance
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "7860"]
