# 🧠 Documentation Technique du Modèle

Cette page détaille la conception, l'évaluation et les choix de modélisation retenus pour la prédiction du turnover.

## 🛠️ 1. Pipeline de Préparation (Preprocessing)

Le modèle ne reçoit pas les données brutes. Il s'appuie sur une `Pipeline` Scikit-Learn garantissant que chaque transformation appliquée à l'entraînement est reproduite à l'identique lors de l'inférence via l'API.

**Étapes clés :**

* **Nettoyage (`DataCleaningTransformer`)** : Gestion des valeurs manquantes et formatage des types de colonnes.
* **Ingénierie des caractéristiques (`SalaryFeatureEngineering`)** : Création d'une métrique de positionnement salarial par rapport à la médiane du poste et du niveau d'éducation.
* **Encodage** : Application du `OneHotEncoder` pour les variables catégorielles (ex: poste, département) et `KBinsDiscretizer` pour transformer des variables continues en groupes logiques.

## 📊 2. Performances du Modèle

Le modèle retenu est un **Random Forest Classifier** optimisé via une recherche d'hyperparamètres (`GridSearchCV`).

### Métriques d'Évaluation
Comme le jeu de données est déséquilibré (peu de départs par rapport aux effectifs restants), nous avons privilégié le **F1-Score** plutôt que l'Accuracy.

* **F1-Score (Test)** : `0.57`
* **Précision** : Indique la fiabilité d'une alerte de départ.
* **Rappel (Recall)** : Indique notre capacité à ne pas "rater" un employé qui va réellement partir.

## 🎯 3. Optimisation du Seuil de Décision

Par défaut, un classifieur utilise un seuil de **0.5**. Cependant, pour ce projet RH, le coût d'un "oubli" (un employé qui part sans qu'on l'ait prédit) est plus élevé qu'une fausse alerte.

* **Seuil optimal retenu** : `0.32`
* **Justification** : Ce seuil a été calculé pour maximiser le F1-Score sur l'échantillon de test, permettant un meilleur équilibre entre la détection des départs et la précision des alertes.

## 🔍 4. Interprétabilité (Feature Importance)

Le modèle s'appuie principalement sur les variables suivantes pour prendre ses décisions :

* **Satisfaction au travail** (Environnement et Nature du travail).
* **Distance domicile-travail**.
* **Ancienneté sous le responsable actuel**.
* **Positionnement salarial**.
