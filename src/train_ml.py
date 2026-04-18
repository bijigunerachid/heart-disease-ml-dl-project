"""
train_ml.py
Entraîne plusieurs modèles ML et sauvegarde les meilleurs.
Modèles : Logistic Regression, Random Forest, XGBoost, SVM, KNN
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, roc_auc_score, confusion_matrix
)
from sklearn.model_selection import cross_val_score
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# ── Chemins ─────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
PROC_PATH = ROOT / "data" / "processed"
ML_PATH   = ROOT / "models" / "ml"
ML_PATH.mkdir(parents=True, exist_ok=True)

# ── Modèles ──────────────────────────────────────────────────────────────────
MODELS = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
    "random_forest":       RandomForestClassifier(n_estimators=200, random_state=42),
    "gradient_boosting":   GradientBoostingClassifier(n_estimators=200, random_state=42),
    "svm":                 SVC(probability=True, random_state=42),
    "knn":                 KNeighborsClassifier(n_neighbors=7),
}
if HAS_XGB:
    MODELS["xgboost"] = XGBClassifier(
        n_estimators=200, use_label_encoder=False,
        eval_metric="logloss", random_state=42
    )


def load_data():
    X_train = np.load(PROC_PATH / "X_train.npy")
    X_test  = np.load(PROC_PATH / "X_test.npy")
    y_train = np.load(PROC_PATH / "y_train.npy")
    y_test  = np.load(PROC_PATH / "y_test.npy")
    return X_train, X_test, y_train, y_test


def train_and_evaluate(name, model, X_train, X_test, y_train, y_test):
    print(f"\n{'='*50}")
    print(f"  Modèle : {name}")
    print(f"{'='*50}")

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    print(f"  CV Accuracy : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Entraînement final
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

    print(f"  Test Accuracy : {acc:.4f}")
    if auc:
        print(f"  ROC-AUC       : {auc:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Sain', 'Malade'])}")

    # Sauvegarde
    joblib.dump(model, ML_PATH / f"{name}.joblib")

    return {"model": name, "cv_acc": cv_scores.mean(), "test_acc": acc, "roc_auc": auc}


def main():
    print("Chargement des données...")
    X_train, X_test, y_train, y_test = load_data()
    print(f"Train : {X_train.shape}  |  Test : {X_test.shape}\n")

    results = []
    for name, model in MODELS.items():
        res = train_and_evaluate(name, model, X_train, X_test, y_train, y_test)
        results.append(res)

    # Tableau récapitulatif
    df_results = pd.DataFrame(results).sort_values("test_acc", ascending=False)
    print("\n" + "="*50)
    print("  RÉCAPITULATIF")
    print("="*50)
    print(df_results.to_string(index=False))
    df_results.to_csv(ML_PATH / "results_summary.csv", index=False)

    best = df_results.iloc[0]["model"]
    print(f"\n  Meilleur modèle : {best}")
    print(f"  Modèles sauvegardés dans : {ML_PATH}")


if __name__ == "__main__":
    main()
