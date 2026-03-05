# 🛠️ Maintenance et Cycle de Vie du Modèle

Une fois déployé, un modèle de Machine Learning peut perdre en efficacité à cause du changement des comportements humains ou des politiques de l'entreprise. Cette page détaille notre stratégie pour garantir la fiabilité du système sur le long terme.

## 📡 1. Monitoring et Traçabilité (Logging)

Pour surveiller le comportement du modèle, nous avons mis en place une journalisation systématique :

* **Stockage des prédictions** : Chaque appel à l'endpoint `/predict` déclenche une écriture dans la base de données **Supabase** via SQLAlchemy.
* **Données enregistrées** : Nous stockons les caractéristiques d'entrée, la probabilité de départ calculée, la prédiction finale et l'horodatage.
* **Utilité** : Ces logs constituent notre futur jeu de données d'entraînement pour réévaluer le modèle par rapport aux départs réels observés par le département RH.

## 📉 2. Gestion du Data Drift (Dérive des données)

Le "Data Drift" survient lorsque les caractéristiques des employés changent (ex: une inflation galopante qui rendrait obsolète notre ancien seuil de `revenu_mensuel`).

**Protocole d'alerte :**

* **Évaluation trimestrielle** : Comparaison du F1-Score prédit vs réel (basé sur les départs effectifs constatés).
* **Analyse de distribution** : Vérifier si la répartition des probabilités de départ reste stable dans le temps.
* **Seuil d'alerte** : Si le F1-Score chute de plus de **10%** par rapport à notre étalon de test (`0.57`), un ré-entraînement est obligatoire.

## 🔄 3. Processus de Mise à Jour (Retraining)

Le cycle de mise à jour suit une procédure stricte de "Continuous Deployment" :

* **Ré-entraînement** : Utilisation du script `src/train.py` sur les nouvelles données collectées et étiquetées.
* **Tests de non-régression** : Avant tout remplacement en production, le nouveau modèle doit impérativement passer le test `tests/test_model.py`.

## 🚀 4. Pipeline CI/CD et Robustesse

Le déploiement est protégé par **GitHub Actions**. Aucune modification ne peut atteindre la production si elle ne satisfait pas les conditions suivantes :

* **Qualité du code** : Passage des tests unitaires (FastAPI, Pydantic).
* **Intégrité mathématique** : Succès du test de performance sur l'échantillon fixe (`test_sample.csv`).
* **Conteneurisation** : Build réussi de l'image Docker.
