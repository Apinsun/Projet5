from sklearn.base import BaseEstimator, ClassifierMixin


class ModelWithThreshold(BaseEstimator, ClassifierMixin):
    def __init__(self, model, threshold):
        self.model = model
        self.threshold = threshold

    def predict(self, X):
        # Récupère les probabilités de la classe positive (index 1)
        probas = self.model.predict_proba(X)[:, 1]
        # Applique le seuil personnalisé
        return (probas >= self.threshold).astype(int)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
