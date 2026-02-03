import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from ydata_profiling import ProfileReport
from sklearn.calibration import cross_val_predict
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import KBinsDiscretizer
import joblib
from model_With_Threshold import ModelWithThreshold
from preprocessing import DataCleaningTransformer, FeatureEngineeringTransformer, ColumnDropperTransformer, SalaryFeatureEngineering
from pathlib import Path
import json
### Mes fonctions ###

# Crée une colonne indiquant si la personne a des études en adéquation avec son département
from sklearn.metrics import classification_report, f1_score, precision_recall_curve


def find_optimal_threshold(y_true, y_probas):
    """
    Trouve et affiche le meilleur seuil à partir des probabilités déjà calculées.
    """
    # 1. Calculer P, R et Thresholds
    precision, recall, thresholds = precision_recall_curve(y_true, y_probas)
    
    # 2. Calculer F1 pour chaque seuil
    numerator = 2 * recall * precision
    denominator = recall + precision
    
    # Gestion de la division par zéro
    f1_scores = np.divide(numerator, denominator, out=np.zeros_like(denominator), where=denominator!=0)
    
    # On aligne les tailles (P et R ont un élément de plus que thresholds)
    f1_scores = f1_scores[:-1] 
    precision = precision[:-1]
    recall = recall[:-1]
    
    # 3. Trouver l'optimum
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    print(f"Seuil optimal trouvé : {best_thresh:.4f} avec F1-score = {best_f1:.4f}, Précision = {precision[best_idx]:.4f}, Rappel = {recall[best_idx]:.4f}")
    
    return best_thresh



###################################################################
###         """ Partie principale du script """                 ###
###################################################################

# on récupère les différents répertoires
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
models_dir = project_root / "models"
artifacts_dir = project_root / "artifacts"

### Chargement des données ###
    
df_eval = pd.read_csv(project_root / "data/extrait_eval.csv")
df_sirh = pd.read_csv(project_root / "data/extrait_sirh.csv")
df_sondage = pd.read_csv(project_root / "data/extrait_sondage.csv")

### Fusion et nettoyage des datasets ###

df_eval['eval_number'] = df_eval['eval_number'].str.lower().str.replace('e_', '').astype(int)

df_final = (
    df_eval
    .merge(df_sirh, left_on='eval_number', right_on='id_employee')  # Jointure 1
    .drop(columns=['id_employee'])                     # Nettoyage 1
    .merge(df_sondage, left_on='eval_number', right_on='code_sondage')   # Jointure 2 (on réutilise 'id' qui vient de df1)
    .drop(columns=['code_sondage'])                      # Nettoyage 2
)

# ont uniquement la valeur 1, partout donc on enlève
df_clean = df_final.drop(columns=["nombre_employee_sous_responsabilite","ayant_enfants"])
# n'a que la valeur 80 donc on enlève également
df_clean = df_clean.drop(columns="nombre_heures_travailless")
# on enlève également les id
df_clean = df_clean.drop(columns="eval_number")


# on supprime les colonnes trop corrélées d'après le profiling généré auparavant
df_clean = df_clean.drop(columns=['age', 'note_evaluation_actuelle','niveau_hierarchique_poste'])


profile = ProfileReport(df_clean, title="Rapport d'Exploration du data frame final")
profile.to_file(artifacts_dir / "rapport_final.html")

# Affichage des spécifications des colonnes pour notre API



print(f"Spécifications des colonnes du dataset final après nettoyage, adapté pour l'API :")

# Création d'un dictionnaire propre pour ton API
schema_api = {}

for col, dtype in df_clean.dtypes.items():
    dtype_str = str(dtype)
    
    # Mapping basique Pandas -> Python/API
    if 'int' in dtype_str:
        api_type = 'int'
    elif 'float' in dtype_str:
        api_type = 'float'
    elif 'object' in dtype_str or 'category' in dtype_str:
        api_type = 'str' # Souvent 'object' en pandas = string
    elif 'bool' in dtype_str:
        api_type = 'bool'
    else:
        api_type = 'autre (' + dtype_str + ')'
        
    schema_api[col] = api_type

# Affichage propre

print(json.dumps(schema_api, indent=4))

####################################
### 1. Préparation du train test ###
####################################

X = df_clean.drop(columns=['a_quitte_l_entreprise'])
y = df_clean['a_quitte_l_entreprise'].map({'Oui': 1, 'Non': 0})

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


#colonnes à one-hot encoder (pour post il faut le faire avant le salary feature engineering)
cols_a_encoder = ['frequence_deplacement', 'statut_marital','poste']
# cols_target = ['poste']

# handle_unknown='ignore' évite les crashs si une nouvelle catégorie apparait
encodage = ColumnTransformer(
    transformers=[
        ('one_hot', OneHotEncoder(drop='first', handle_unknown='ignore'), cols_a_encoder),


        # La discrétisation pour le ratio (Binning)
        # strategy='quantile' crée des groupes de taille égale
        # n_bins=5 crée 5 niveaux (Très faible, Faible, Moyen, Élevé, Très élevé)
        ('binning_ratio', KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile'),
          ['ratio_stagnation_poste'])
    ],
    remainder='passthrough', # IMPORTANT : on garde les autres colonnes (numériques) telles quelles
    verbose_feature_names_out=False # pour éviter d'avoir "remainder_" devant le nom de nos colonnes
)

salaryFE = SalaryFeatureEngineering(
        group_cols=['poste','niveau_education'],
        target_col='revenu_mensuel',
        drop_original=True,  # On supprime la colonne salaire_mensuel originale car corrélée
        drop_reference=True, # On ne garde pas la colonne de référence médiane
        threshold_std=1      # Seuil pour la discrétisation
    )

rf_cw = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced_subsample',  # Gestion dynamique du déséquilibre
    min_samples_leaf=5,                 # Empêche l'overfitting sur les poids élevés
    max_depth=10                        # (Optionnel) Limite la complexité
)

# Création de la pipeline de feature ingénierie
pipeline_preprocessing = Pipeline([
    ('cleaning', DataCleaningTransformer()),
    ('feature_engineering', FeatureEngineeringTransformer(grosse_distance_threshold=17)),
    ('dropping', ColumnDropperTransformer())
])

full_pipeline = Pipeline([
    # Avant de one-hot encoder les colonnes catégorielles
    ('preprocessing', pipeline_preprocessing),

    ('salary_ref', salaryFE),
    
    # one-hot encoding
    ('encoder', encodage),

    # notre modèle
    ('model', rf_cw) 
])

### Recherche d'hyperparamètres pour le modèle sans imblearn ###

# 1. Définir la grille des paramètres à tester
param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [5, 10, 20, None],
    
    # Important avec class_weight : on évite 1 pour réduire l'overfitting sur les cas isolés
    'model__min_samples_leaf': [2, 4, 6],  
    
    # C'est ici qu'on teste les deux stratégies de gestion du déséquilibre
    'model__class_weight': ['balanced', 'balanced_subsample'] 
}

# 2. Initialiser le GridSearch

scoring_method = 'average_precision' 

# utilisation de la validation croisée stratifiée car la target est déséquilibrée
# Définition de la validation croisée stratifiée
cv_strat = StratifiedKFold(
    n_splits=5,        # Nombre de plis (folds)
    shuffle=True,      # Mélanger les données avant de diviser
    random_state=42    # Pour la reproductibilité
)

grid_search = GridSearchCV(
    estimator=full_pipeline,
    param_grid=param_grid,
    
    # Choix de la métrique
    scoring=scoring_method, 
    
    cv=cv_strat,      
    n_jobs=-1,
    verbose=1
)

# 3. Lancer la recherche
print("Recherche des meilleurs hyperparamètres (Mode Class Weight)...")
grid_search.fit(X_train, y_train)

# 4. Résultats
print(f"\nMeilleurs paramètres trouvés : {grid_search.best_params_}")
print(f"Meilleur score (CV) : {grid_search.best_score_:.4f}")

# On récupère le champion
best_model = grid_search.best_estimator_

y_probas_train_cv = cross_val_predict(best_model, X_train, y_train, cv=cv_strat, method='predict_proba')[:, 1]

# 2. Trouver le seuil optimal sur ces probabilités d'entraînement
best_threshold = find_optimal_threshold(y_train, y_probas_train_cv)

print(f"Le seuil à utiliser en production est : {best_threshold}")

# --- À ce stade, le modèle est calibré. On passe au TEST ---

# 3. Calculer les probabilités sur le TEST
y_probas_test = best_model.predict_proba(X_test)[:, 1]

# 4. Appliquer le seuil trouvé précédemment pour créer les prédictions finales
y_pred_test_optimized = (y_probas_test >= best_threshold).astype(int)

f1_final = f1_score(y_test, y_pred_test_optimized)
print(f"🎯 F1 Score Final (Test) : {f1_final:.4f}")

# 2. Afficher le rapport complet (Précision, Rappel, Support)
# C'est très utile pour voir si ton modèle privilégie trop une classe
print("\n📊 Rapport de Classification :")
print(classification_report(y_test, y_pred_test_optimized))

# 2. Créer l'objet final
final_model = ModelWithThreshold(best_model, best_threshold)

#On sauvegarde le modèle
save_path = models_dir / "model.pkl"
joblib.dump(final_model, save_path)

print(f"Modèle sauvegardé avec succès dans : {save_path}")