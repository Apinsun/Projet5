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
import joblib
from model_With_Threshold import ModelWithThreshold
from pathlib import Path
### Mes fonctions ###

# Crée une colonne indiquant si la personne a des études en adéquation avec son département
from sklearn.metrics import precision_recall_curve


def definir_match_etudes_poste(row):
    """
    Retourne 1 si les études correspondent au département, 0 sinon.
    """
    etudes = row['domaine_etude']
    dept = row['departement']
    
    # 1. Dictionnaire de correspondance (Etudes -> Départements acceptables)
    mapping_logique = {
        # Les Techs vont au Consulting
        'Infra & Cloud': ['Consulting'],
        'Transformation Digitale': ['Consulting'],
        
        # Les Business vont au Commercial
        'Marketing': ['Commercial'],
        'Entrepreunariat': ['Commercial'],
        
        # Les RH vont aux RH
        'Ressources Humaines': ['Ressources Humaines']
    }
    
    # 2. Gestion du cas "Autre" ( On considère que c'est pas OK)
    if etudes == 'Autre':
        return 0

    # 3. Vérification standard
    # On regarde si le département actuel est dans la liste autorisée pour ces études
    if etudes in mapping_logique:
        if dept in mapping_logique[etudes]:
            return 1 # C'est un Match
        else:
            return 0 # C'est un Mismatch (ex: Marketing -> Consulting)
            
    # Sécurité pour les cas imprévus
    return 0

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

from sklearn.base import BaseEstimator, TransformerMixin

class SalaryFeatureEngineering(BaseEstimator, TransformerMixin):
    """
    Transformateur personnalisé pour évaluer la compétitivité du salaire.
    
    Objectif :
    Comparer le salaire d'un employé par rapport à ses pairs (groupe défini par défaut 
    sur 'poste' et 'niveau_education') pour déterminer s'il est sous-payé ou sur-payé.
    
    Logique :
    1. Calcule la moyenne et l'écart-type du salaire pour chaque groupe.
    2. Calcule le Z-Score : (Salaire - Moyenne du groupe) / Ecart-type du groupe.
    3. Discrétise en 'score_competitivite' : 
       -1 (sous-payé), 0 (normal), 1 (bien payé) selon le seuil (threshold_std).
       
    Nettoyage spécifique (Hardcoded) :
    - Supprime 'niveau_education' après le calcul (feature selection intégrée).
    - Conserve 'poste'.
    """

    def __init__(self, group_cols=['poste', 'niveau_education'], 
                 target_col='revenu_mensuel', 
                 drop_original=True,
                 drop_reference=True,
                 threshold_std=1.0): 
        
        self.group_cols = group_cols
        self.target_col = target_col
        self.drop_original = drop_original
        self.drop_reference = drop_reference
        self.threshold_std = threshold_std
        
        self.group_stats_ = None
        self.global_stats_ = {}
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        X = X.copy()
        if hasattr(X, 'columns'):
            self.feature_names_in_ = list(X.columns)
        else:
            self.feature_names_in_ = [f"x{i}" for i in range(X.shape[1])]

        # 1. Stats globales
        self.global_stats_['mean'] = X[self.target_col].mean()
        self.global_stats_['std'] = X[self.target_col].std()
        self.global_stats_['median'] = X[self.target_col].median()

        # 2. Stats par groupe
        self.group_stats_ = X.groupby(self.group_cols)[self.target_col].agg(['mean', 'std', 'median'])
        self.group_stats_['std'] = self.group_stats_['std'].fillna(1.0).replace(0, 1.0)
        
        return self

    def transform(self, X):
        X = X.copy()
        
        # # 1. Join
        # X = X.join(self.group_stats_, on=self.group_cols)

        X = pd.merge(
            X, 
            self.group_stats_, 
            left_on=self.group_cols, 
            right_index=True, 
            how='left'
        )
        
        # 2. Remplissage
        X['mean'] = X['mean'].fillna(self.global_stats_['mean'])
        X['std'] = X['std'].fillna(self.global_stats_['std'])
        X['median'] = X['median'].fillna(self.global_stats_['median'])
        
        X['salaire_median_ref'] = X['median']
        
        # 3. Calcul Z-Score
        X['z_score_salaire'] = (X[self.target_col] - X['mean']) / X['std']
        
        # 4. Discrétisation
        X['score_competitivite'] = 0
        X.loc[X['z_score_salaire'] < -self.threshold_std, 'score_competitivite'] = -1
        X.loc[X['z_score_salaire'] > self.threshold_std, 'score_competitivite'] = 1
        
        # 5. NETTOYAGE
        cols_to_drop = ['mean', 'std', 'median', 'z_score_salaire']
        
        if self.drop_original:
            cols_to_drop.append(self.target_col)
        if self.drop_reference:
            cols_to_drop.append('salaire_median_ref')
            
        # --- C'EST ICI QUE VOUS FAITES VOTRE BRICOLAGE PROPREMENT ---
        # On supprime TOUJOURS 'niveau_education' car on a décidé qu'il était inutile
        if 'niveau_education' in X.columns:
            cols_to_drop.append('niveau_education')
        # if 'poste' in X.columns:
        #     cols_to_drop.append('poste')  
        # Par contre, ON NE TOUCHE PAS à 'poste', on le laisse passer !
            
        X = X.drop(columns=cols_to_drop, errors='ignore')
            
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = self.feature_names_in_
        feature_names = list(input_features)
        
        # 1. Suppression target
        if self.drop_original and self.target_col in feature_names:
            feature_names.remove(self.target_col)
            
        # 2. Suppression explicite de niveau_education (SYNCHRONISATION !)
        if 'niveau_education' in feature_names:
            feature_names.remove('niveau_education')

        # 3. Suppression médiane (si demandé)
        if not self.drop_reference:
            feature_names.append('salaire_median_ref')
            
        # 4. Ajout du score
        feature_names.append('score_competitivite')
        
        return np.array(feature_names, dtype=object)
    

### Classes permettant de réaliser le feature engineering dans des pipelines ###

class DataCleaningTransformer(BaseEstimator, TransformerMixin):
    """
    S'occupe du nettoyage initial et du typage.
    """
    def fit(self, X, y=None):
        return self # Rien à apprendre ici

    def transform(self, X):
        X = X.copy() # Toujours copier pour ne pas toucher aux données originales
        
        # Correction augmentation salaire
        if 'augementation_salaire_precedente' in X.columns:
            # On vérifie si c'est string avant de replace, au cas où des données propres arrivent
            if X['augementation_salaire_precedente'].dtype == 'object':
                X['augementation_salaire_precedente'] = X['augementation_salaire_precedente'].str.replace('%', '').astype(int)

        # Mappings binaires
        mappings = {'Oui': 1, 'Non': 0}
        
        # Heures supp
        if 'heure_supplementaires' in X.columns:
            X['heure_supplementaires'] = X['heure_supplementaires'].map(mappings)
            # Gestion des NaN éventuels si une valeur inconnue arrive
            X['heure_supplementaires'] = X['heure_supplementaires'].fillna(0) 

        # Genre
        if 'genre' in X.columns:
            X['genre'] = X['genre'].map({'M': 0, 'F': 1})
            
        return X

class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """
    Crée les nouvelles variables métier.
    """
    def __init__(self, grosse_distance_threshold=17):
        # On met le seuil en paramètre : cela permet de le changer 
        # ou de le "GridSearcher" plus tard !
        self.grosse_distance_threshold = grosse_distance_threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        
        # 1. Ratio Stagnation Entreprise
        # On sécurise pour éviter la division par zéro
        X['Ratio_Stagnation_entreprise'] = np.where(
            X['annees_dans_l_entreprise'] == 0, 
            0, 
            X['annees_depuis_la_derniere_promotion'] / X['annees_dans_l_entreprise']
        )

        # 2. Match Etudes (Assure-toi que la fonction est accessible)
        X['match_etudes_poste'] = X.apply(definir_match_etudes_poste, axis=1)
        # NOTE : Si cette fonction est lente, il faudra l'optimiser, mais gardons-la pour l'instant.

        # 3. Durée moyenne par poste
        X['duree_moyenne_par_poste'] = X['annee_experience_totale'] / (X['nombre_experiences_precedentes'] + 1)

        # 4. Habite loin (utilise le paramètre de la classe)
        X['habite_loin'] = (X['distance_domicile_travail'] > self.grosse_distance_threshold).astype(int)

        # 5. Ratio Stagnation Poste
        X['ratio_stagnation_poste'] = X['annees_dans_le_poste_actuel'] / (X['annees_dans_l_entreprise'] + 1)

        # 6. Nouveau Responsable
        X['nouveau_responsable'] = (X['annes_sous_responsable_actuel'] < 2).astype(int)

        # 7. Ratio Gain Effort
        # Attention : s'assurer que heure_supplementaires est bien numérique (fait dans l'étape d'avant)
        X['Ratio_Gain_Effort'] = X['revenu_mensuel'] / (1 + X['heure_supplementaires'])

        return X

class ColumnDropperTransformer(BaseEstimator, TransformerMixin):
    """
    Supprime les colonnes inutiles après le feature engineering.
    """
    def __init__(self, columns_to_drop=None):
        if columns_to_drop is None:
            self.columns_to_drop = [
                'domaine_etude',
                'annees_depuis_la_derniere_promotion',
                'nombre_experiences_precedentes',
                'annees_dans_l_entreprise',
                'annees_dans_le_poste_actuel',
                'distance_domicile_travail',
                'annes_sous_responsable_actuel',
                'departement',
                'annee_experience_totale'
            ]
        else:
            self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # On ne drop que les colonnes qui existent réellement dans X
        # (Pour éviter les erreurs si on relance le script deux fois)
        cols_existing = [c for c in self.columns_to_drop if c in X.columns]
        return X.drop(columns=cols_existing)

###################################################################
###         """ Partie principale du script """                 ###
###################################################################

# on récupère les différents répertoires
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
models_dir = project_root / "models"

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
profile.to_file("../artifacts/rapport_final.html")

# 1. Préparation du train test
#from category_encoders import TargetEncoder
from sklearn.preprocessing import KBinsDiscretizer


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

# 2. Créer l'objet final
final_model = ModelWithThreshold(best_model, best_threshold)

#On sauvegarde le modèle
save_path = models_dir / "model.pkl"
joblib.dump(final_model, save_path)

print(f"Modèle sauvegardé avec succès dans : {save_path}")