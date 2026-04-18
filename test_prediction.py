#!/usr/bin/env python
"""
🧪 TEST DU PIPELINE DE PRÉDICTION
Vérifie que l'app fonctionne sans Streamlit UI
"""
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

# Config
ROOT      = Path(__file__).resolve().parent
ML_PATH   = ROOT / "models" / "ml"
PROC_PATH = ROOT / "data" / "processed"

print("\n" + "="*60)
print("  🧪 TEST DU PIPELINE DE PRÉDICTION")
print("="*60 + "\n")

# 1. Charger les artefacts
print("[1/4] Chargement des artefacts ML...")
try:
    model        = joblib.load(ML_PATH / "best_model_tuned.joblib")
    preprocessor = joblib.load(ML_PATH / "preprocessor.joblib")
    le_encoders  = joblib.load(ML_PATH / "label_encoders.joblib")
    print(f"  ✅ Modèle chargé : {type(model).__name__}")
    print(f"  ✅ Preprocessor chargé : {type(preprocessor).__name__}")
    print(f"  ✅ Label encoders chargés : {len(le_encoders)} colonnes")
except Exception as e:
    print(f"  ❌ Erreur : {e}")
    sys.exit(1)

# 2. Créer une donnée de test
print("\n[2/4] Création des données de test...")
try:
    test_inputs = {
        "age": 55,
        "sex": "Male",
        "cp": "typical angina",
        "trestbps": 140,
        "chol": 250,
        "fbs": 0,
        "restecg": "normal",
        "thalch": 142,
        "exang": 0,
        "oldpeak": 1.5,
        "slope": "upsloping",
        "ca": 0.0,
        "thal": "normal",
        "dataset": "cleveland",
    }
    
    # Build raw DF (même logique que app.py)
    age = float(test_inputs.get("age", 1) or 1)
    df = pd.DataFrame([test_inputs])
    
    # Convertir colonnes OHE en strings IMMÉDIATEMENT
    ohe_cols = ["cp", "thal", "dataset"]
    for col in ohe_cols:
        if col in df.columns:
            df[col] = df[col].fillna("missing").astype(str)
    
    # Traiter les valeurs zéro
    if pd.notna(df.at[0, "chol"]) and float(df.at[0, "chol"]) == 0:
        df.at[0, "chol"] = np.nan
    if pd.notna(df.at[0, "trestbps"]) and float(df.at[0, "trestbps"]) == 0:
        df.at[0, "trestbps"] = np.nan
    
    # 🔥 FEATURE ENGINEERING (IMPORTANT!)
    df["oldpeak"]      = pd.to_numeric(df["oldpeak"], errors="coerce").clip(lower=0.0)
    df["ca_missing"]   = df["ca"].isna().astype(np.int8)
    df["thal_missing"] = df["thal"].isna().astype(np.int8)
    df["chol_per_age"] = pd.to_numeric(df["chol"], errors="coerce") / age
    df["thalch_ratio"] = pd.to_numeric(df["thalch"], errors="coerce") / max(220.0 - age, 1.0)
    
    print(f"  ✅ DataFrame créé : {df.shape}")
    print(f"  ✅ Colonnes OHE en string :")
    for col in ohe_cols:
        print(f"     - {col}: {df[col].dtype}")
except Exception as e:
    print(f"  ❌ Erreur : {e}")
    sys.exit(1)

# 3. Appliquer le preprocessing
print("\n[3/4] Application du preprocessing...")
try:
    _LBL_COLS = ["sex", "restecg", "slope"]
    
    # Encoder les colonnes label
    for col in _LBL_COLS:
        if col not in le_encoders:
            raise KeyError(f"LabelEncoder manquant : '{col}'")
        le  = le_encoders[col]
        raw = df.at[0, col]
        val = str(raw) if pd.notna(raw) else ""
        if val not in set(le.classes_):
            val = le.classes_[0]
        df.at[0, col] = le.transform([val])[0]
    
    # Convertir en numérique
    for col in _LBL_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # S'assurer que les colonnes OHE restent strings
    ohe_cols = ["cp", "thal", "dataset"]
    for col in ohe_cols:
        if col in df.columns and df[col].dtype != object:
            df[col] = df[col].fillna("missing").astype(str)
    
    # Transform
    X = preprocessor.transform(df)
    print(f"  ✅ Preprocessing réussi : X.shape = {X.shape}")
except Exception as e:
    print(f"  ❌ Erreur : {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 4. Lancer la prédiction
print("\n[4/4] Lancement de la prédiction...")
try:
    est = getattr(model, "best_estimator_", model)
    if not hasattr(est, "predict_proba"):
        raise AttributeError(f"{type(est).__name__} ne supporte pas predict_proba")
    
    pm   = est.predict_proba(X)
    prob = float(np.clip(pm[0, 1], 0.0, 1.0))
    label = int(prob >= 0.5)
    
    print(f"  ✅ Prédiction réussie !")
    print(f"     Probabilité risque : {prob:.4f}")
    print(f"     Diagnostic : {'⚠️  RISQUE' if label else '✅ NORMAL'}")
except Exception as e:
    print(f"  ❌ Erreur : {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("  ✅ TOUS LES TESTS RÉUSSIS !")
print("  L'app est prête pour Streamlit Cloud")
print("="*60 + "\n")
