import pandas as pd
import numpy as  np 
import joblib 
from pathlib import Path 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler , LabelEncoder , OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# ── Chemins ───────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
RAW_PATH    = ROOT / "data" / "raw" / "heart_disease_uci.csv"
PROC_PATH   = ROOT / "data" / "processed"
ML_PATH     = ROOT / "models" / "ml"
PROC_PATH.mkdir(parents=True, exist_ok=True)
ML_PATH.mkdir(parents=True, exist_ok=True)

TARGET   = "num"
DROP     = ["id"]          # identifiant inutile

## Etape 1 - Chargement 

def load(path):
    df = pd.read_csv(path)
    print(f"[load]  {df.shape[0]} lignes × {df.shape[1]} colonnes")
    return df

## Etape 2 - Nettoyage des valeurs aberrantes

def clean_outliers(df):
    df = df.copy()

    # chol = 0 → impossible médicalement (172 cas) → NaN
    n_chol = (df["chol"] == 0).sum()
    df.loc[df["chol"] == 0, "chol"] = np.nan
    print(f"[clean] chol=0 → NaN : {n_chol} valeurs")

    # trestbps = 0 → impossible (1 cas) → NaN
    n_bps = (df["trestbps"] == 0).sum()
    df.loc[df["trestbps"] == 0, "trestbps"] = np.nan
    print(f"[clean] trestbps=0 → NaN : {n_bps} valeur")

    # oldpeak < 0 → valeurs négatives n'ont pas de sens (12 cas) → 0
    n_old = (df["oldpeak"] < 0).sum()
    df.loc[df["oldpeak"] < 0, "oldpeak"] = 0.0
    print(f"[clean] oldpeak<0 → 0 : {n_old} valeurs")

    return df

## Etape 3 — Cible binaire + typage

def prepare_target(df):
    df = df.copy()
    df[TARGET] = (df[TARGET] > 0).astype(int)

    # fbs / exang : True/False string → 0/1 (garder NaN)
    for col in ["fbs", "exang"]:
        df[col] = df[col].map({True: 1, False: 0, "True": 1, "False": 0})

    print(f"[target] Distribution : {dict(df[TARGET].value_counts().sort_index())}")
    return df

## Etape 4 — Feature engineering

def feature_engineering(df):
    df = df.copy()

    # Flags manque (avant imputation)
    df["ca_missing"]   = df["ca"].isna().astype(int)
    df["thal_missing"] = df["thal"].isna().astype(int)

    # Features dérivées
    df["chol_per_age"]  = df["chol"] / df["age"]
    df["thalch_ratio"]  = df["thalch"] / (220 - df["age"]).clip(lower=1)

    print(f"[feat]  Features ajoutées : ca_missing, thal_missing, chol_per_age, thalch_ratio")
    return df

## Etape 5 — Définition des colonnes & pipeline sklearn

def build_preprocessor():
    # Colonnes numériques (imputation médiane + RobustScaler)
    num_cols = [
        "age", "trestbps", "chol", "thalch", "oldpeak", "ca",
        "chol_per_age", "thalch_ratio"
    ]
    # Colonnes binaires/ordinales déjà numériques (0/1 ou entier)
    bin_cols = ["fbs", "exang", "ca_missing", "thal_missing"]

    # Catégorielles → OHE  (cp, thal, dataset)
    ohe_cols = ["cp", "thal", "dataset"]

    # Catégorielles → Label (sex, restecg, slope — ordinal ou binaire)
    lbl_cols = ["sex", "restecg", "slope"]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  RobustScaler()),
    ])
    bin_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])
    ohe_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe",     OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    lbl_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("scaler",  RobustScaler()),          # après label encoding manuel ci-dessous
    ])

    return num_cols, bin_cols, ohe_cols, lbl_cols, num_pipe, bin_pipe, ohe_pipe, lbl_pipe

## Etape 6 — Encodage manuel des colonnes label (pour garder l'ordre ordinal)

def label_encode(df, lbl_cols, fit=True, encoders=None):
    df = df.copy()
    if fit:
        encoders = {}
    for col in lbl_cols:
        df[col] = df[col].astype(str)
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
        else:
            le = encoders[col]
            # Gestion des labels inconnus (test set)
            known = set(le.classes_)
            df[col] = df[col].apply(lambda x: x if x in known else le.classes_[0])
            df[col] = le.transform(df[col])
    return df, encoders

## Main

def main():
    print("\n" + "="*60)
    print("  PIPELINE DE PRÉTRAITEMENT — Heart Disease UCI")
    print("="*60 + "\n")

    # 1. Chargement
    df = load(RAW_PATH)
    df = df.drop(columns=DROP)

    # 2. Nettoyage aberrants
    df = clean_outliers(df)

    # 3. Cible + typage
    df = prepare_target(df)

    # 4. Feature engineering
    df = feature_engineering(df)

    # 5. Split AVANT toute imputation/normalisation (éviter data leakage)
    X = df.drop(columns=[TARGET])
    y = df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n[split] Train={len(X_train)}  Test={len(X_test)}  "
          f"(pos_rate train={y_train.mean():.3f}  test={y_test.mean():.3f})")

    # 6. Encodage label (fit sur train seulement)
    num_cols, bin_cols, ohe_cols, lbl_cols, num_pipe, bin_pipe, ohe_pipe, _ = build_preprocessor()
    X_train, le_encoders = label_encode(X_train, lbl_cols, fit=True)
    X_test,  _           = label_encode(X_test,  lbl_cols, fit=False, encoders=le_encoders)

    # Ajouter lbl_cols aux numériques après encoding
    all_num = num_cols + lbl_cols + bin_cols

    # 7. ColumnTransformer (fit sur train, transform sur test)
    preprocessor = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("bin", bin_pipe, bin_cols + lbl_cols),
        ("ohe", ohe_pipe, ohe_cols),
    ], remainder="drop")

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc  = preprocessor.transform(X_test)

    # Récupérer les noms de features
    ohe_feature_names = preprocessor.named_transformers_["ohe"]["ohe"].get_feature_names_out(ohe_cols)
    feature_names = num_cols + bin_cols + lbl_cols + list(ohe_feature_names)

    print(f"\n[preproc] X_train : {X_train_proc.shape}  X_test : {X_test_proc.shape}")
    print(f"[preproc] Features finales ({len(feature_names)}) : {feature_names}")

    # 8. SMOTE sur le train (déséquilibre de classes)
    try:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42)
        X_train_bal, y_train_bal = smote.fit_resample(X_train_proc, y_train)
        print(f"\n[smote] Avant : {dict(zip(*np.unique(y_train, return_counts=True)))}")
        print(f"[smote] Après : {dict(zip(*np.unique(y_train_bal, return_counts=True)))}")
    except ImportError:
        print("\n[smote] imbalanced-learn non installé → pas de SMOTE (pip install imbalanced-learn)")
        X_train_bal, y_train_bal = X_train_proc, y_train

    # 9. Sauvegarde
    np.save(PROC_PATH / "X_train.npy",     X_train_bal)
    np.save(PROC_PATH / "X_test.npy",      X_test_proc)
    np.save(PROC_PATH / "y_train.npy",     y_train_bal)
    np.save(PROC_PATH / "y_test.npy",      y_test)
    np.save(PROC_PATH / "feature_names.npy", np.array(feature_names))

    # CSV lisible
    pd.DataFrame(X_train_bal, columns=feature_names).assign(num=y_train_bal)\
      .to_csv(PROC_PATH / "train.csv", index=False)
    pd.DataFrame(X_test_proc, columns=feature_names).assign(num=y_test)\
      .to_csv(PROC_PATH / "test.csv",  index=False)

    joblib.dump(preprocessor,  ML_PATH / "preprocessor.joblib")
    joblib.dump(le_encoders,   ML_PATH / "label_encoders.joblib")

    print(f"\n[done]  Fichiers sauvegardés dans {PROC_PATH}")
    print(f"        Preprocessor sauvegardé dans {ML_PATH}")


if __name__ == "__main__":
    main()
