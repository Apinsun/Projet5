# 📚 Documentation de l'API

L'API est construite avec FastAPI, ce qui permet une validation automatique via Pydantic et une documentation interactive native.

## 🔗 Accès à la documentation interactive
Vous pouvez tester l'API directement via l'interface Swagger UI :
👉 [Lien vers Swagger UI](https://apinsun-projet5.hf.space/docs)

## 🚀 Utilisation avec Python
Voici un script exemple pour interroger l'API de prédiction :

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
