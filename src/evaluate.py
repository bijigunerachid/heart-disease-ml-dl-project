"""
evaluate.py
Évaluation complète de tous les modèles ML sauvegardés + génération de figures.
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve,
    confusion_matrix, classification_report, ConfusionMatrixDisplay
)

# ── Chemins ─────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
PROC_PATH = ROOT / "data" / "processed"
ML_PATH   = ROOT / "models" / "ml"
FIGS_PATH = ROOT / "reports" / "figures"
FIGS_PATH.mkdir(parents=True, exist_ok=True)


def load_data():
    X_test  = np.load(PROC_PATH / "X_test.npy")
    y_test  = np.load(PROC_PATH / "y_test.npy")
    return X_test, y_test


def load_models():
    models = {}
    for f in ML_PATH.glob("*.joblib"):
        if "imputer" not in f.stem and "scaler" not in f.stem and "encoder" not in f.stem:
            models[f.stem] = joblib.load(f)
    return models


def plot_roc_curves(models, X_test, y_test):
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, proba)
            auc = roc_auc_score(y_test, proba)
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Comparaison des modèles ML")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIGS_PATH / "roc_curves.png", dpi=150)
    plt.close()
    print(f"  [fig] roc_curves.png sauvegardé")


def plot_confusion_matrices(models, X_test, y_test):
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (name, model) in zip(axes, models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=["Sain", "Malade"])
        disp.plot(ax=ax, colorbar=False)
        ax.set_title(name.replace("_", " ").title())
    plt.tight_layout()
    plt.savefig(FIGS_PATH / "confusion_matrices.png", dpi=150)
    plt.close()
    print(f"  [fig] confusion_matrices.png sauvegardé")


def plot_accuracy_comparison(results_df):
    fig, ax = plt.subplots(figsize=(8, 5))
    results_sorted = results_df.sort_values("test_acc", ascending=True)
    bars = ax.barh(results_sorted["model"], results_sorted["test_acc"], color="steelblue")
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Test Accuracy")
    ax.set_title("Comparaison des modèles — Test Accuracy")
    plt.tight_layout()
    plt.savefig(FIGS_PATH / "accuracy_comparison.png", dpi=150)
    plt.close()
    print(f"  [fig] accuracy_comparison.png sauvegardé")


def feature_importance(models, X_test):
    """Affiche l'importance des features pour les modèles tree-based."""
    feature_cols = pd.read_csv(PROC_PATH / "train.csv").drop(columns=["num"]).columns.tolist()
    for name, model in models.items():
        if hasattr(model, "feature_importances_"):
            fi = pd.Series(model.feature_importances_, index=feature_cols)
            fi_sorted = fi.sort_values(ascending=True)
            fig, ax = plt.subplots(figsize=(7, 5))
            fi_sorted.plot(kind="barh", ax=ax, color="coral")
            ax.set_title(f"Feature Importances — {name.replace('_', ' ').title()}")
            plt.tight_layout()
            plt.savefig(FIGS_PATH / f"feature_importance_{name}.png", dpi=150)
            plt.close()
            print(f"  [fig] feature_importance_{name}.png sauvegardé")


def main():
    print("Chargement des données et modèles...\n")
    X_test, y_test = load_data()
    models = load_models()

    if not models:
        print("Aucun modèle trouvé. Lancez d'abord train_ml.py")
        return

    # Tableau de résultats
    rows = []
    for name, model in models.items():
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        rows.append({
            "model":    name,
            "test_acc": accuracy_score(y_test, y_pred),
            "roc_auc":  roc_auc_score(y_test, y_proba) if y_proba is not None else None,
        })
        print(f"  {name}:")
        print(classification_report(y_test, y_pred, target_names=["Sain", "Malade"]))

    df = pd.DataFrame(rows).sort_values("test_acc", ascending=False)
    print("RÉCAPITULATIF :\n", df.to_string(index=False))

    # Figures
    print("\nGénération des figures...")
    plot_roc_curves(models, X_test, y_test)
    plot_confusion_matrices(models, X_test, y_test)
    plot_accuracy_comparison(df)
    feature_importance(models, X_test)

    print(f"\nFigures disponibles dans : {FIGS_PATH}")


if __name__ == "__main__":
    main()
