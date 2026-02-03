from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

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