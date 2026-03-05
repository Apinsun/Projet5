# 🚀 Bienvenue dans la documentation du Projet Turnover

Ce projet vise à fournir un outil d'aide à la décision pour les services RH. Grâce à un modèle de Machine Learning, nous prédisons la probabilité de départ d'un employé en fonction de critères socio-professionnels.

## 🏗️ Architecture et Choix Techniques

Afin de garantir la fiabilité et l'évolutivité du projet, les choix technologiques suivants ont été faits :

* **API :** `FastAPI` (pour sa rapidité et la génération automatique de la documentation Swagger).

* **Validation des données :** `Pydantic` (garantit que le modèle ML ne reçoit que des données conformes, ex: notes de 0 à 4).

* **Modélisation ML :** `scikit-learn` avec une pipeline complète (Preprocessing, Encodage, RandomForestClassifier).

* **Conteneurisation :** `Docker` (assure que l'application tourne de la même manière en local et en production).

* **Base de données (Logs) :** `Supabase` (PostgreSQL). **Choix technique spécifique :** Utilisation du *Connection Pooler* (Port 6543) pour forcer le trafic en IPv4 afin de contourner les limitations réseau des espaces Docker Hugging Face.

* **CI/CD :** `GitHub Actions` avec séparation stricte des environnements (Test sur toutes les branches, Déploiement conditionné à la branche `main`).
