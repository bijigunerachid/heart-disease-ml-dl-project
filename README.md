# Heart Disease Prediction Project

Projet de prédiction de maladies cardiaques utilisant le dataset UCI Heart Disease.  
Approche complète : EDA → Preprocessing → ML → DL → Évaluation.

---

## Dataset

- **Source** : [UCI Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)
- **Fichier** : `data/raw/heart_disease_uci.csv`
- **Taille** : 920 observations × 16 features
- **Cible** : `num` (0 = pas de maladie, 1-4 = maladie)

### Features principales

| Feature    | Description                          | Type       |
|------------|--------------------------------------|------------|
| `age`      | Âge du patient                       | int        |
| `sex`      | Sexe (Male / Female)                 | catégoriel |
| `cp`       | Type de douleur thoracique           | catégoriel |
| `trestbps` | Pression artérielle au repos (mmHg)  | int        |
| `chol`     | Cholestérol (mg/dl)                  | int        |
| `fbs`      | Glycémie à jeun > 120 mg/dl          | bool       |
| `restecg`  | Résultat ECG au repos                | catégoriel |
| `thalch`   | Fréquence cardiaque maximale         | int        |
| `exang`    | Angine induite par l'effort          | bool       |
| `oldpeak`  | Dépression ST à l'effort             | float      |
| `slope`    | Pente du segment ST                  | catégoriel |
| `ca`       | Nb artères colorées par fluoroscopie | int        |
| `thal`     | Type de thalassémie                  | catégoriel |
| `dataset`  | Centre de collecte                   | catégoriel |
| `num`      | **Cible** (0–4)                      | int        |

---

## Structure du projet

```
disease_prediction_project/
├── .gitignore
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                    # Données brutes (ne pas modifier)
│   │   └── heart_disease_uci.csv
│   └── processed/              # Données après preprocessing
├── notebooks/
│   ├── 01_EDA.ipynb            # Analyse exploratoire
│   ├── 02_preprocessing.ipynb  # Nettoyage & feature engineering
│   ├── 03_ml_models.ipynb      # Modèles ML classiques
│   └── 04_dl_model.ipynb       # Modèle Deep Learning
├── src/
│   ├── preprocess.py           # Pipeline de prétraitement
│   ├── train_ml.py             # Entraînement ML
│   ├── train_dl.py             # Entraînement DL
│   └── evaluate.py             # Évaluation & métriques
├── models/
│   ├── ml/                     # Modèles ML sauvegardés (.pkl / .joblib)
│   └── dl/                     # Modèles DL sauvegardés (.pt / .h5)
└── reports/
    └── figures/                # Graphiques exportés
```

---

## Lancement rapide

```bash
# 1. Installer les dépendances
pip install -r requirements.txt

# 2. Prétraitement
python src/preprocess.py

# 3. Entraîner les modèles ML
python src/train_ml.py

# 4. Entraîner le modèle DL
python src/train_dl.py

# 5. Évaluation
python src/evaluate.py
```

---

## Résultats attendus

| Modèle              | Accuracy (approx.) |
|---------------------|--------------------|
| Logistic Regression | ~82%               |
| Random Forest       | ~85%               |
| XGBoost             | ~87%               |
| Neural Network (DL) | ~86%               |

---

## Auteur

Projet réalisé dans le cadre du cursus **Bachelor en Technologie – Big Data**  
EST Fkih Ben Salah, Université Sultan Moulay Slimane
