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

🌟 Documentation Interactive (Swagger) : https://apinsun-projet5.hf.space/docs

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
```

```python

import requests
# L'URL publique de ton API sur Hugging Face
API_URL = "https://apinsun-projet5.hf.space/predict"

# Les données de l'employé à tester
donnees_employe = {
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

# Envoi de la requête POST
reponse = requests.post(API_URL, json=donnees_employe)

# Affichage du résultat
if reponse.status_code == 200:
    resultat = reponse.json()
    print(f"Prédiction (1 = Départ, 0 = Reste) : {resultat['prediction']}")
    print(f"Probabilité de départ : {resultat['probabilite_depart'] * 100:.1f} %")
else:
    print(f"Erreur {reponse.status_code} : {reponse.text}")'
```

## 🧠 3. Modèle ML : Performances et Maintenance

### Performances du Modèle
Le modèle principal est un `RandomForestClassifier` optimisé avec une gestion du déséquilibre des classes (`class_weight='balanced_subsample'`).
* **Métrique d'optimisation :** F1-Score (choisi pour équilibrer la précision et le rappel sur une cible RH déséquilibrée).
* **F1-Score (Test) :** 0.57
* **Seuil de décision optimal :** 0.32 (Abaissé par rapport au 0.5 par défaut pour limiter les "oublis" et maximiser la détection des employés sur le départ).

### Maintenance et Protocole de Mise à Jour (Data Drift)
Pour garantir la fiabilité du modèle dans le temps face à l'évolution de l'entreprise :
1. **Monitoring :** Chaque prédiction est journalisée dans la base de données Supabase.
2. **Évaluation trimestrielle :** Les prédictions sont croisées avec les départs réels.
3. **Ré-entraînement :** Si le F1-Score chute de plus de 10 % en production, un ré-entraînement automatique est déclenché sur les données récentes.
4. **Tests CI/CD :** Le nouveau modèle doit valider le test de non-régression (`tests/test_model.py`) sur un échantillon de référence avant tout déploiement.

## 📚 4. Documentation Technique Complète

Pour une plongée approfondie dans l'architecture, la construction du modèle et le pipeline de déploiement, consultez notre documentation technique complète générée avec **MkDocs** :

👉 **https://github.com/Apinsun/Projet5/tree/main/docs**
