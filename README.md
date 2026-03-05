---
title: Mon API ML Projet 5
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# 🚀 API de Prédiction de Turnover (Machine Learning)

Ce projet déploie un modèle de Machine Learning via une API RESTful robuste, permettant de prédire le risque de départ (turnover) des employés. L'API est conteneurisée avec Docker, intègre une base de données distante pour la journalisation, et est déployée de manière continue via un pipeline CI/CD.

## 🏗️ 1. Architecture et Choix Techniques

Afin de garantir la fiabilité et l'évolutivité du projet, les choix technologiques suivants ont été faits :
* **API :** `FastAPI` (pour sa rapidité et la génération automatique de la documentation Swagger).
* **Validation des données :** `Pydantic` (garantit que le modèle ML ne reçoit que des données conformes, ex: notes de 0 à 4).
* **Modélisation ML :** `scikit-learn` avec une pipeline complète (Preprocessing, Encodage, RandomForestClassifier).
* **Conteneurisation :** `Docker` (assure que l'application tourne de la même manière en local et en production).
* **Base de données (Logs) :** `Supabase` (PostgreSQL). **Choix technique spécifique :** Utilisation du *Connection Pooler* (Port 6543) pour forcer le trafic en IPv4 afin de contourner les limitations réseau des espaces Docker Hugging Face.
* **CI/CD :** `GitHub Actions` avec séparation stricte des environnements (Test sur toutes les branches, Déploiement conditionné à la branche `main`).

## ⚙️ 2. Installation et Configuration en Local

### Prérequis
* Python 3.12
* Poetry (Gestionnaire de dépendances)
* Docker (Optionnel, pour tester le conteneur en local)

### Lancement via Poetry
1. Cloner le dépôt : `git clone https://github.com/Apinsun/Projet5.git`
2. Installer les dépendances : `poetry install`
3. Créer un fichier `.env` à la racine et y ajouter la clé de la BDD :
   `DATABASE_URL="postgresql://postgres.[id]:[mdp]@aws-0-[region].pooler.supabase.com:6543/postgres"`
4. Lancer l'API : `poetry run uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload`

### Lancement via Docker
```bash
docker build -t api-ml .
docker run -p 8000:8000 --env-file .env api-ml
