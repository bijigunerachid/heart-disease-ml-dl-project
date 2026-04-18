"""
CardioRisk AI — Outil d'aide à la décision clinique (prototype avancé)
══════════════════════════════════════════════════════════════════════
Modèle     : XGBoost (GridSearchCV) entraîné sur Heart Disease UCI (920 patients)
Version    : 2.0.0
Python     : 3.10+
Dépendances: streamlit, pandas, numpy, joblib, plotly, scikit-learn
Usage      : streamlit run app.py
"""

from __future__ import annotations

import csv
import logging
import platform
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import os

# ══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════════
_LOG_LEVEL = logging.DEBUG if os.getenv("CARDIORISK_DEBUG") == "1" else logging.INFO
logging.basicConfig(
    level=_LOG_LEVEL,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("cardiorisk")

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES
# ══════════════════════════════════════════════════════════════════════════════
APP_VERSION = "2.0.0"
ROOT      = Path(__file__).resolve().parent
ML_PATH   = ROOT / "models" / "ml"
PROC_PATH = ROOT / "data" / "processed"
LOG_PATH  = ROOT / "logs"
LOG_FILE  = LOG_PATH / "predictions_audit.csv"

_LBL_COLS: list[str] = ["sex", "restecg", "slope"]
_CLINICAL_BOUNDS: dict[str, tuple[float, float]] = {
    "age": (18,120), "trestbps": (60,220),
    "chol": (0,600), "thalch": (60,250), "oldpeak": (0.0,10.0),
}
_RGPD_EXCLUDED_FIELDS: frozenset[str] = frozenset()

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG STREAMLIT — thème géré nativement par Streamlit (menu ⋮ → Settings)
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CardioRisk AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
def _init_session_state() -> None:
    defaults: dict[str, Any] = {
        "last_prediction": None,
        "artifacts_loaded": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_session_state()

# ══════════════════════════════════════════════════════════════════════════════
# CSS — design professionnel, compatible thème clair ET sombre natif Streamlit
# On utilise les variables CSS natives de Streamlit (--background-color, etc.)
# + quelques overrides pour la typographie et les composants custom.
# ══════════════════════════════════════════════════════════════════════════════
def _inject_css() -> None:
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ══ FONT ═══════════════════════════════════════════════════════════════ */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    -webkit-font-smoothing: antialiased !important;
}

/* ══ CACHE LE MENU PRINCIPAL mais pas le header (pour le toggle thème) ═ */
#MainMenu { visibility: hidden !important; }
[data-testid="stDecoration"] { display: none !important; }
footer { visibility: hidden !important; }

/* ══ SCROLLBAR ══════════════════════════════════════════════════════════ */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(220,53,89,.35); border-radius: 2px; }

/* ══ SIDEBAR ════════════════════════════════════════════════════════════ */
section[data-testid="stSidebar"] {
    border-right: 1px solid rgba(128,128,128,0.15) !important;
    min-width: 256px !important;
    max-width: 256px !important;
}
[data-testid="stSidebar"] > div:first-child,
[data-testid="stSidebarContent"] {
    padding: 0 !important;
    display: flex !important;
    flex-direction: column !important;
    height: 100vh !important;
    overflow: hidden !important;
}

/* ── Radio nav ── */
[data-testid="stSidebar"] .stRadio > label { display: none !important; }
[data-testid="stSidebar"] .stRadio [role="radiogroup"] {
    gap: 2px !important;
    display: flex !important;
    flex-direction: column !important;
    padding: 4px 10px !important;
}
[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"] {
    background: transparent !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 9px 14px !important;
    cursor: pointer !important;
    transition: background .15s ease !important;
    width: 100% !important;
}
[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"]:hover {
    background: rgba(220,53,89,0.08) !important;
}
[data-testid="stSidebar"] .stRadio [aria-checked="true"] {
    background: rgba(220,53,89,0.12) !important;
    border-left: 2px solid #dc3559 !important;
    border-radius: 0 8px 8px 0 !important;
    padding-left: 12px !important;
}
[data-testid="stSidebar"] .stRadio [aria-checked="true"] p {
    color: #dc3559 !important;
    font-weight: 600 !important;
}
[data-testid="stSidebar"] .stRadio [data-baseweb="radio"] > div:first-child {
    display: none !important;
}
[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p {
    font-size: 13.5px !important;
    font-weight: 500 !important;
    line-height: 1 !important;
    letter-spacing: -.1px !important;
}

/* ── Sidebar HTML components ── */
.sb-logo {
    padding: 20px 16px 15px;
    border-bottom: 1px solid rgba(128,128,128,0.12);
    display: flex; align-items: center; gap: 10px;
}
.sb-logo-icon {
    width: 34px; height: 34px; border-radius: 9px;
    background: linear-gradient(135deg, #dc3559, #b91c3c);
    display: flex; align-items: center; justify-content: center;
    font-size: 17px; flex-shrink: 0;
    box-shadow: 0 3px 10px rgba(220,53,89,.30);
}
.sb-logo-name {
    font-size: 14.5px; font-weight: 700;
    letter-spacing: -.3px; display: block; line-height: 1.2;
}
.sb-logo-sub {
    font-size: 10.5px; opacity: .55;
    display: block; margin-top: 1px;
}
.sb-section {
    font-size: 9.5px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 1.2px;
    opacity: .45;
    padding: 16px 18px 5px; display: block;
}
.sb-spacer { flex: 1; min-height: 16px; }
.sb-footer {
    padding: 12px 16px 18px;
    border-top: 1px solid rgba(128,128,128,0.12);
}
.sb-stats {
    display: grid; grid-template-columns: 1fr 1fr;
    gap: 7px; margin-bottom: 10px;
}
.sb-stat {
    border: 1px solid rgba(128,128,128,0.12);
    border-radius: 8px; padding: 7px 9px;
    background: rgba(128,128,128,0.04);
}
.sb-stat-label {
    font-size: 9px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.8px;
    opacity: .45; display: block; margin-bottom: 2px;
}
.sb-stat-val {
    font-size: 14px; font-weight: 700;
    display: block; letter-spacing: -.3px;
}
.sb-tag {
    display: inline-flex; align-items: center; gap: 5px;
    background: rgba(220,53,89,.08);
    border: 1px solid rgba(220,53,89,.18);
    color: #dc3559 !important;
    border-radius: 6px; font-size: 10px; font-weight: 600;
    padding: 4px 9px; margin-top: 4px; letter-spacing: .2px;
}

/* ══ MAIN CONTENT ═══════════════════════════════════════════════════════ */
[data-testid="stMain"] { background: transparent !important; }
[data-testid="block-container"] {
    padding: 2.2rem 2.8rem 4rem !important;
    max-width: 1180px !important;
}

/* ══ TYPOGRAPHIE ════════════════════════════════════════════════════════ */
h1, h2, h3, h4 { font-weight: 700 !important; letter-spacing: -.3px !important; }
h2 { font-size: 1.45rem !important; }
h3 { font-size: 1.1rem !important; font-weight: 600 !important; }

/* ══ PAGE HEADER ════════════════════════════════════════════════════════ */
.page-header {
    padding: 2.4rem 0 1.8rem;
    border-bottom: 1px solid rgba(128,128,128,0.12);
    margin-bottom: 2rem;
}
.page-header-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(220,53,89,.08);
    border: 1px solid rgba(220,53,89,.16);
    color: #dc3559 !important;
    border-radius: 20px; font-size: 11px; font-weight: 600;
    padding: 4px 12px; margin-bottom: 14px; letter-spacing: .3px;
}
.page-header h1 {
    font-size: 2.1rem !important; font-weight: 800 !important;
    margin: 0 0 8px !important; letter-spacing: -.6px !important;
    line-height: 1.15 !important;
}
.page-header h1 em { font-style: normal; color: #dc3559; }
.page-header-sub {
    font-size: .98rem; opacity: .65;
    margin: 0 !important; font-weight: 400 !important; max-width: 560px;
}

/* ══ SECTION DIVIDER ════════════════════════════════════════════════════ */
.section-divider {
    display: flex; align-items: center; gap: 12px;
    margin: 2rem 0 1.2rem;
}
.section-divider-label {
    font-size: 9.5px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 1.2px; opacity: .45; white-space: nowrap;
}
.section-divider-line {
    flex: 1; height: 1px;
    background: rgba(128,128,128,0.15);
}

/* ══ DISCLAIMER ═════════════════════════════════════════════════════════ */
.disclaimer {
    display: flex; gap: 12px; align-items: flex-start;
    background: rgba(245,158,11,.07);
    border: 1px solid rgba(245,158,11,.25);
    border-radius: 10px; padding: 12px 16px; margin: 0 0 1.5rem;
}
.disclaimer-icon { font-size: 14px; flex-shrink: 0; margin-top: 1px; }
.disclaimer-text {
    font-size: 12.5px; color: #92400e !important;
    line-height: 1.6; margin: 0 !important;
}
.disclaimer-text strong { color: #78350f !important; font-weight: 600 !important; }

/* ══ STAT CARDS ═════════════════════════════════════════════════════════ */
.stat-card {
    border: 1px solid rgba(128,128,128,0.12);
    border-radius: 12px; padding: 1.3rem 1.1rem;
    position: relative; overflow: hidden;
    background: rgba(128,128,128,0.03);
    transition: transform .22s ease, box-shadow .22s ease;
}
.stat-card:hover { transform: translateY(-2px); }
.stat-card-accent {
    position: absolute; top: 0; left: 0;
    width: 3px; height: 100%;
    background: linear-gradient(180deg, #dc3559, transparent);
    border-radius: 12px 0 0 12px;
}
.stat-label {
    font-size: 10px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 1px; opacity: .5;
    margin-bottom: 6px; display: block;
}
.stat-value {
    font-size: 1.75rem; font-weight: 800;
    letter-spacing: -.5px; line-height: 1;
}
.stat-desc { font-size: 11.5px; opacity: .45; margin-top: 4px; display: block; }

/* ══ RÉSULTAT ═══════════════════════════════════════════════════════════ */
.result-card {
    border-radius: 14px; padding: 2rem 1.6rem;
    text-align: center;
}
.result-high {
    background: rgba(220,53,89,.07);
    border: 1px solid rgba(220,53,89,.20);
}
.result-low {
    background: rgba(16,185,129,.07);
    border: 1px solid rgba(16,185,129,.20);
}
.result-status {
    font-size: 10.5px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 1.2px; margin-bottom: 8px; display: block;
}
.result-high .result-status { color: #f87171 !important; }
.result-low  .result-status { color: #34d399 !important; }
.result-prob {
    font-size: 3rem; font-weight: 800;
    line-height: 1; letter-spacing: -2px; margin: 4px 0 10px;
}
.result-high .result-prob { color: #f87171 !important; }
.result-low  .result-prob { color: #34d399 !important; }
.result-label { font-size: 12px; opacity: .55; }

/* ══ FORMULAIRE ═════════════════════════════════════════════════════════ */
[data-testid="stForm"] {
    border-radius: 14px !important;
    padding: 1.8rem !important;
}
.stNumberInput input,
.stSelectbox > div > div,
.stTextInput > div > div > input {
    border-radius: 8px !important;
    font-size: 13.5px !important;
    transition: border-color .15s ease, box-shadow .15s ease !important;
}

/* ══ BOUTON SUBMIT ══════════════════════════════════════════════════════ */
[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(135deg, #dc3559 0%, #b91c3c 100%) !important;
    color: #ffffff !important; border: none !important;
    border-radius: 10px !important; font-weight: 600 !important;
    font-size: 14px !important; letter-spacing: .2px !important;
    box-shadow: 0 4px 18px rgba(220,53,89,.30) !important;
    height: 46px !important;
    transition: transform .18s ease, box-shadow .18s ease !important;
}
[data-testid="stFormSubmitButton"] button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 24px rgba(220,53,89,.40) !important;
}

/* ══ PROGRESS BAR ═══════════════════════════════════════════════════════ */
[data-testid="stProgress"] > div { border-radius: 4px !important; height: 5px !important; }
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #dc3559, #ff6b8a) !important;
    border-radius: 4px !important;
}

/* ══ TABLES MARKDOWN ════════════════════════════════════════════════════ */
.stMarkdown table {
    width: 100%; border-collapse: collapse;
    border-radius: 10px; overflow: hidden;
    font-size: 13.5px;
    border: 1px solid rgba(128,128,128,0.12) !important;
}
.stMarkdown th {
    font-weight: 600 !important; padding: 11px 16px !important;
    text-align: left !important;
    border-bottom: 1px solid rgba(128,128,128,0.15) !important;
    font-size: 11.5px !important; text-transform: uppercase !important;
    letter-spacing: .4px !important; opacity: .7;
}
.stMarkdown td {
    padding: 10px 16px !important;
    border-bottom: 1px solid rgba(128,128,128,0.08) !important;
}
.stMarkdown tr:last-child td { border-bottom: none !important; }

/* ══ MÉTRIQUES ══════════════════════════════════════════════════════════ */
[data-testid="stMetric"] {
    border: 1px solid rgba(128,128,128,0.12) !important;
    border-radius: 10px !important; padding: .85rem 1rem !important;
    transition: transform .2s ease !important;
}
[data-testid="stMetric"]:hover { transform: translateY(-2px) !important; }
[data-testid="stMetric"] label {
    font-size: 10px !important; text-transform: uppercase !important;
    letter-spacing: .8px !important; font-weight: 700 !important; opacity: .5 !important;
}
[data-testid="stMetricValue"] {
    font-size: 20px !important; font-weight: 700 !important;
    letter-spacing: -.5px !important;
}

/* ══ ALERTES ════════════════════════════════════════════════════════════ */
.stAlert { border-radius: 10px !important; border-left-width: 3px !important; }

/* ══ CAPTION ════════════════════════════════════════════════════════════ */
[data-testid="stCaptionContainer"] p {
    font-size: 11.5px !important; font-style: italic !important; opacity: .5 !important;
}

/* ══ SMOOTH GLOBAL ══════════════════════════════════════════════════════ */
* { box-sizing: border-box; }
[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"] {
    transition: background .15s ease, padding-left .15s ease !important;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ML
# ══════════════════════════════════════════════════════════════════════════════
# WRAPPER POUR LE PREPROCESSOR — Force les types avant transform
# ══════════════════════════════════════════════════════════════════════════════
class SafePreprocessorWrapper:
    """Wrapper qui force les types des colonnes OHE avant le transform"""
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform avec conversion de type stricte pour OHE colonnes"""
        X = X.copy()
        
        # Forcer les types pour les colonnes OneHotEncoded
        ohe_cols = ["cp", "thal", "dataset"]
        for col in ohe_cols:
            if col in X.columns:
                # Convertir en string, gérer les NaN
                X[col] = X[col].astype(str)
                X.loc[X[col].isin(['nan', 'None', '<NA>', 'NaN']), col] = 'missing'
        
        # Appeler le preprocessor original
        return self.preprocessor.transform(X)
    
    def fit_transform(self, X: pd.DataFrame, y=None) -> np.ndarray:
        """Fit + transform (si nécessaire)"""
        return self.preprocessor.fit_transform(X, y)
    
    def get_feature_names_out(self, input_features=None):
        """Passer à travers la méthode get_feature_names_out"""
        return self.preprocessor.get_feature_names_out(input_features)


@st.cache_resource(show_spinner=False)
def load_artifacts() -> tuple:
    required = {
        "model":        ML_PATH   / "best_model_tuned.joblib",
        "preprocessor": ML_PATH   / "preprocessor.joblib",
        "le_encoders":  ML_PATH   / "label_encoders.joblib",
        "feat_names":   PROC_PATH / "feature_names.npy",
    }
    missing = [str(p) for p in required.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Artefacts ML introuvables :\n" + "\n".join(f"  · {m}" for m in missing))
    try:
        model        = joblib.load(required["model"])
        preprocessor = joblib.load(required["preprocessor"])
        le_encoders  = joblib.load(required["le_encoders"])
        feat_names   = np.load(required["feat_names"], allow_pickle=True)
    except Exception as exc:
        raise RuntimeError(f"Échec désérialisation : {exc}") from exc
    if not isinstance(le_encoders, dict):
        raise TypeError(f"label_encoders doit être un dict : {type(le_encoders)}")
    if feat_names.ndim != 1 or len(feat_names) == 0:
        raise ValueError("feature_names doit être un tableau 1D non vide.")
    
    # WRAPPER le preprocessor pour forcer les types
    preprocessor = SafePreprocessorWrapper(preprocessor)
    
    return model, preprocessor, le_encoders, feat_names


@st.cache_data(show_spinner=False)
def get_feature_importance_cached(_model: Any, _feat_names: np.ndarray, top_n: int = 15) -> pd.DataFrame:
    est = getattr(_model, "best_estimator_", _model)
    if not hasattr(est, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])
    imp = est.feature_importances_
    n   = min(len(imp), len(_feat_names))
    df  = (pd.DataFrame({"feature": _feat_names[:n], "importance": imp[:n]})
           .sort_values("importance", ascending=False).head(top_n).reset_index(drop=True))
    df["importance"] = df["importance"].astype(float)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def _validate_clinical_inputs(inputs: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field, (lo, hi) in _CLINICAL_BOUNDS.items():
        val = inputs.get(field)
        if val is None: continue
        try:
            if np.isnan(float(val)): continue
        except (TypeError, ValueError):
            errors.append(f"Valeur non numérique pour '{field}' : {val!r}"); continue
        if not (lo <= float(val) <= hi):
            if field == "chol" and float(val) == 0: continue
            errors.append(f"'{field}' hors plage [{lo}–{hi}] : {val}")
    return errors


def build_raw_df(inputs: dict[str, Any]) -> pd.DataFrame:
    age = float(inputs.get("age", 1) or 1)
    if age <= 0: raise ValueError(f"Âge invalide : {age}")
    
    # Forcer les types AVANT de créer le DataFrame
    # (évite que pandas inère les types automatiquement)
    inputs_safe = dict(inputs)
    
    # S'ASSURER que les colonnes OneHotEncoded sont des STRINGS **IMMÉDIATEMENT**
    ohe_cols = ["cp", "thal", "dataset"]
    for col in ohe_cols:
        if col in inputs_safe:
            val = inputs_safe[col]
            # Forcer en string
            if val is None or (isinstance(val, float) and np.isnan(val)):
                inputs_safe[col] = "missing"
            else:
                inputs_safe[col] = str(val).strip()
    
    # Créer DataFrame avec types explicites
    df = pd.DataFrame([inputs_safe])
    
    # FORCER les types des colonnes OHE à 'object' (string)
    for col in ohe_cols:
        if col in df.columns:
            df[col] = df[col].astype('object').astype(str)
    
    logger.debug(f"[build_raw_df] OHE cols dtypes: {[(col, df[col].dtype) for col in ohe_cols]}")
    
    # Traiter les valeurs zéro
    if pd.notna(df.at[0, "chol"]) and float(df.at[0, "chol"]) == 0:
        df.at[0, "chol"] = np.nan
    if pd.notna(df.at[0, "trestbps"]) and float(df.at[0, "trestbps"]) == 0:
        df.at[0, "trestbps"] = np.nan
    
    # Feature engineering
    df["oldpeak"]      = pd.to_numeric(df["oldpeak"], errors="coerce").clip(lower=0.0)
    df["ca_missing"]   = df["ca"].isna().astype(np.int8)
    df["thal_missing"] = df["thal"].isna().astype(np.int8)
    df["chol_per_age"] = pd.to_numeric(df["chol"], errors="coerce") / age
    df["thalch_ratio"] = pd.to_numeric(df["thalch"], errors="coerce") / max(220.0 - age, 1.0)
    
    logger.debug(f"[build_raw_df] Final df shape: {df.shape}, dtypes: {df.dtypes.to_dict()}")
    return df


def apply_preprocessing(raw_df: pd.DataFrame, le_encoders: dict, preprocessor: Any) -> np.ndarray:
    df = raw_df.copy()
    
    # 1. Encoder les colonnes label (sex, restecg, slope)
    for col in _LBL_COLS:
        if col not in le_encoders: 
            raise KeyError(f"LabelEncoder manquant : '{col}'")
        le  = le_encoders[col]
        raw = df.at[0, col]
        val = str(raw) if pd.notna(raw) else ""
        if val not in set(le.classes_):
            val = le.classes_[0]
        df.at[0, col] = le.transform([val])[0]
    
    # 2. Convertir les colonnes label en numérique
    for col in _LBL_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # 3. Le wrapper SafePreprocessorWrapper gère les conversions de type OHE
    #    → pas besoin de faire la conversion ici
    
    logger.debug(f"[apply_preprocessing] Input df dtypes: {df.dtypes.to_dict()}")
    
    # Transform (avec wrapper qui force les types)
    X = preprocessor.transform(df)
    
    logger.debug(f"[apply_preprocessing] Output shape: {X.shape}")
    return X


def run_prediction(X: np.ndarray, model: Any) -> tuple[float, int]:
    est = getattr(model, "best_estimator_", model)
    if not hasattr(est, "predict_proba"):
        raise AttributeError(f"{type(est).__name__} ne supporte pas predict_proba")
    pm = est.predict_proba(X)
    if pm.shape[0] != 1 or pm.shape[1] < 2:
        raise ValueError(f"predict_proba shape inattendue : {pm.shape}")
    prob = float(np.clip(pm[0, 1], 0.0, 1.0))
    return prob, int(prob >= 0.5)


# ══════════════════════════════════════════════════════════════════════════════
# AUDIT LOG
# ══════════════════════════════════════════════════════════════════════════════
_CSV_LOCK = threading.Lock()
if platform.system() == "Windows":
    try:
        import msvcrt as _msvcrt; _HAS_MSVCRT = True
    except ImportError:
        _HAS_MSVCRT = False
    _HAS_FCNTL = False
else:
    try:
        import fcntl as _fcntl; _HAS_FCNTL = True
    except ImportError:
        _HAS_FCNTL = False
    _HAS_MSVCRT = False

def _lock_file(fh) -> None:
    if _HAS_FCNTL: _fcntl.flock(fh, _fcntl.LOCK_EX)
def _unlock_file(fh) -> None:
    if _HAS_FCNTL: _fcntl.flock(fh, _fcntl.LOCK_UN)

def log_prediction(inputs: dict[str, Any], prob: float, label: int) -> None:
    LOG_PATH.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp":   datetime.now().isoformat(timespec="seconds"),
        "prediction":  "Malade" if label == 1 else "Sain",
        "probability": round(float(prob), 4),
        **{k: v for k, v in inputs.items() if k not in _RGPD_EXCLUDED_FIELDS},
    }
    write_header = not LOG_FILE.exists()
    try:
        with _CSV_LOCK:
            with open(LOG_FILE, "a", newline="", encoding="utf-8") as fh:
                _lock_file(fh)
                try:
                    w = csv.DictWriter(fh, fieldnames=list(record.keys()))
                    if write_header: w.writeheader()
                    w.writerow(record); fh.flush()
                finally:
                    _unlock_file(fh)
    except OSError as exc:
        logger.error("Erreur log : %s", exc)


# ══════════════════════════════════════════════════════════════════════════════
# COMPOSANTS UI
# ══════════════════════════════════════════════════════════════════════════════
def render_disclaimer() -> None:
    st.markdown("""
    <div class="disclaimer">
        <span class="disclaimer-icon">⚠️</span>
        <p class="disclaimer-text">
            <strong>Avertissement médical</strong> — Outil d'aide à la décision basé sur l'IA.
            Ne remplace pas un diagnostic médical professionnel. Les résultats sont indicatifs
            et ne constituent pas le seul fondement d'une décision thérapeutique.
        </p>
    </div>""", unsafe_allow_html=True)


def _sec(label: str) -> None:
    st.markdown(f"""
    <div class="section-divider">
        <span class="section-divider-label">{label}</span>
        <span class="section-divider-line"></span>
    </div>""", unsafe_allow_html=True)


def _plotly_base(fig: go.Figure, height: int = 300) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=48, b=16),
        showlegend=False,
        font_family="Inter, -apple-system, sans-serif",
    )
    fig.update_xaxes(gridcolor="rgba(128,128,128,0.12)", zeroline=False,
                     tickfont=dict(size=11))
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.12)", zeroline=False,
                     tickfont=dict(size=11))
    return fig


def render_gauge(prob: float) -> go.Figure:
    prob = float(np.clip(prob, 0.0, 1.0))
    col  = "#f87171" if prob >= 0.5 else "#34d399"
    fig  = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = prob * 100,
        title = {"text": "Probabilité de maladie cardiaque",
                 "font": {"size": 12, "family": "Inter"}},
        number= {"suffix": "%", "valueformat": ".1f",
                 "font": {"size": 40, "family": "Inter"}},
        gauge = {
            "axis": {"range":[0,100], "tickwidth":1,
                     "tickfont":{"size":10}},
            "bar":  {"color": col, "thickness": 0.2},
            "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
            "steps": [
                {"range":[0,  35],  "color":"rgba(52,211,153,.06)"},
                {"range":[35, 50],  "color":"rgba(251,191,36,.06)"},
                {"range":[50, 70],  "color":"rgba(251,146,60,.07)"},
                {"range":[70, 100], "color":"rgba(248,113,113,.09)"},
            ],
            "threshold": {"line":{"width":1.5}, "thickness":0.75, "value":50},
        },
    ))
    return _plotly_base(fig, height=260)


def render_importance_chart(df_imp: pd.DataFrame,
                            title: str = "Variables les plus prédictives") -> go.Figure:
    if df_imp.empty:
        fig = go.Figure()
        fig.add_annotation(text="Importances non disponibles.", showarrow=False)
        return _plotly_base(fig, 300)
    mx   = df_imp["importance"].max() or 1.0
    df_s = df_imp.sort_values("importance")
    cols = [f"rgba(220,53,89,{0.28 + 0.72 * v / mx:.2f})" for v in df_s["importance"]]
    fig  = px.bar(df_s, x="importance", y="feature", orientation="h",
                  labels={"importance": "Score d'importance (gain)", "feature": ""},
                  title=title)
    fig.update_traces(marker_color=cols, marker_line_width=0)
    fig.update_layout(title_font=dict(size=13, family="Inter"),
                      yaxis=dict(tickfont=dict(size=12)))
    return _plotly_base(fig, 460)


# ══════════════════════════════════════════════════════════════════════════════
# PAGES
# ══════════════════════════════════════════════════════════════════════════════
def page_accueil() -> None:
    st.markdown(f"""
    <div class="page-header">
        <span class="page-header-badge">Prototype de recherche · v{APP_VERSION}</span>
        <h1>CardioRisk <em>AI</em></h1>
        <p class="page-header-sub">
            Système clinique d'aide à la décision pour la prédiction du risque
            cardiovasculaire basé sur l'apprentissage automatique.
        </p>
    </div>""", unsafe_allow_html=True)

    render_disclaimer()

    _sec("Performances du modèle")
    c1, c2, c3, c4 = st.columns(4)
    for col, (label, val, desc) in zip([c1,c2,c3,c4], [
        ("Accuracy",    "86.4 %",  "Test set UCI — 184 patients"),
        ("ROC-AUC",     "92.97 %", "Capacité discriminante"),
        ("Sensibilité", "91 %",    "Détection vrais malades"),
        ("Spécificité", "80 %",    "Identification vrais sains"),
    ]):
        with col:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-card-accent"></div>
                <span class="stat-label">{label}</span>
                <span class="stat-value">{val}</span>
                <span class="stat-desc">{desc}</span>
            </div>""", unsafe_allow_html=True)

    _sec("Fonctionnement du système")
    c1, c2, c3 = st.columns(3)
    for col, (num, title, desc) in zip([c1,c2,c3], [
        ("01", "Saisie des données cliniques",
         "Renseignez les paramètres du patient : données démographiques, bilan lipidique, ECG de repos et d'effort."),
        ("02", "Analyse par le modèle XGBoost",
         "Le modèle analyse 26 variables cliniques et calcule une probabilité de risque cardiaque calibrée."),
        ("03", "Résultat et interprétation",
         "Un score de risque, une jauge visuelle et les variables influentes guident la décision clinique."),
    ]):
        with col:
            st.info(f"**{num} — {title}**\n\n{desc}")

    _sec("Dataset d'entraînement")
    st.markdown("""
| Paramètre | Valeur |
|---|---|
| **Dataset** | Heart Disease UCI Repository |
| **Patients** | 920 — Cleveland · Hungary · Switzerland · VA Long Beach |
| **Variables originales** | 14 variables cliniques |
| **Variables finales** | 26 (feature engineering + OneHotEncoding) |
| **Algorithme** | XGBoost — GridSearchCV (5-fold cross-validation) |
| **Déséquilibre** | SMOTE appliqué sur le jeu d'entraînement uniquement |
""")


def page_prediction(model: Any, preprocessor: Any,
                    le_encoders: dict, feat_names: np.ndarray) -> None:
    st.markdown(f"""
    <div class="page-header">
        <span class="page-header-badge">Analyse clinique</span>
        <h1>Prédiction du <em>risque cardiaque</em></h1>
        <p class="page-header-sub">
            Renseignez les paramètres cliniques du patient pour obtenir une estimation
            de la probabilité de maladie cardiaque.
        </p>
    </div>""", unsafe_allow_html=True)

    render_disclaimer()

    with st.form("patient_form", clear_on_submit=False):
        _sec("Données démographiques")
        c1, c2, c3 = st.columns(3)
        with c1:
            age     = st.number_input("Âge (ans)", min_value=18, max_value=120, value=55, step=1)
        with c2:
            sex     = st.selectbox("Sexe biologique", ["Male", "Female"])
        with c3:
            dataset = st.selectbox("Site / Établissement",
                                   ["Cleveland", "Hungary", "Switzerland", "VA Long Beach"])

        _sec("Paramètres hémodynamiques")
        c1, c2, c3 = st.columns(3)
        with c1:
            trestbps = st.number_input("Pression artérielle repos (mmHg)",
                                       min_value=60, max_value=220, value=130, step=1)
        with c2:
            chol     = st.number_input("Cholestérol sérique (mg/dL)",
                                       min_value=0, max_value=600, value=245, step=1,
                                       help="Saisir 0 si inconnue")
        with c3:
            thalch   = st.number_input("Fréquence cardiaque maximale (bpm)",
                                       min_value=60, max_value=250, value=150, step=1)

        _sec("Symptômes et ECG")
        c1, c2 = st.columns(2)
        with c1:
            cp      = st.selectbox("Type de douleur thoracique",
                                   ["asymptomatic","typical angina","atypical angina","non-anginal"])
            restecg = st.selectbox("ECG au repos",
                                   ["normal","lv hypertrophy","st-t abnormality"])
            fbs     = st.radio("Glycémie à jeun > 120 mg/dL", options=[0,1],
                               format_func=lambda x: "Oui" if x==1 else "Non", horizontal=True)
        with c2:
            exang   = st.radio("Angine induite par l'effort", options=[0,1],
                               format_func=lambda x: "Oui" if x==1 else "Non", horizontal=True)
            oldpeak = st.number_input("Dépression ST à l'effort (mm)",
                                      min_value=0.0, max_value=10.0,
                                      value=1.0, step=0.1, format="%.1f")
            slope   = st.selectbox("Pente du segment ST",
                                   ["flat","upsloping","downsloping"])

        _sec("Examens complémentaires (optionnels)")
        c1, c2 = st.columns(2)
        with c1:
            ca_raw   = st.selectbox("Vaisseaux colorés — fluoroscopie",
                                    ["Non renseigné","0","1","2","3"])
        with c2:
            thal_raw = st.selectbox("Scintigraphie myocardique",
                                    ["Non renseigné","normal","fixed defect","reversable defect"])

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Lancer l'analyse du risque cardiaque",
                                         use_container_width=True, type="primary")

    if not submitted:
        return

    ca: Optional[float]
    if ca_raw != "Non renseigné":
        try:
            ca = float(ca_raw)
            if ca not in {0.0,1.0,2.0,3.0}:
                st.error(f"Valeur 'ca' invalide : {ca_raw}"); return
        except ValueError:
            st.error(f"Impossible de convertir 'ca' : {ca_raw!r}"); return
    else:
        ca = np.nan

    thal: Optional[str] = thal_raw if thal_raw != "Non renseigné" else None
    inputs: dict[str, Any] = {
        "age":int(age),"sex":sex,"cp":cp,"trestbps":int(trestbps),
        "chol":int(chol),"fbs":int(fbs),"restecg":restecg,"thalch":int(thalch),
        "exang":int(exang),"oldpeak":float(oldpeak),"slope":slope,
        "ca":ca,"thal":thal,"dataset":dataset,
    }
    errors = _validate_clinical_inputs(inputs)
    if errors:
        for msg in errors: st.error(f"Valeur hors plage clinique — {msg}")
        return

    with st.spinner("Analyse du profil cardiovasculaire en cours…"):
        try:
            raw_df      = build_raw_df(inputs)
            X           = apply_preprocessing(raw_df, le_encoders, preprocessor)
            prob, label = run_prediction(X, model)
        except (ValueError,TypeError,KeyError) as exc:
            st.error(f"Erreur de données — {exc}"); return
        except AttributeError as exc:
            st.error(f"Erreur de modèle — {exc}"); return
        except Exception as exc:
            st.error("Erreur inattendue lors de l'inférence.")
            logger.exception("Inférence : %s", exc); return

    log_prediction(inputs, prob, label)
    st.session_state.last_prediction = {"prob":prob,"label":label,"inputs":inputs}

    st.markdown("---")
    _sec("Résultats de l'analyse")

    col_res, col_gauge = st.columns([1,1], gap="large")
    with col_res:
        card_cls = "result-high" if label==1 else "result-low"
        status   = "Risque élevé détecté" if label==1 else "Risque faible"
        st.markdown(f"""
        <div class="result-card {card_cls}">
            <span class="result-status">{status}</span>
            <div class="result-prob">{prob*100:.1f}%</div>
            <span class="result-label">Probabilité de maladie cardiaque</span>
        </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("Probabilité",       f"{prob*100:.1f} %")
        m2.metric("Seuil décisionnel", "50 %")
        m3.metric("Résultat",          "Malade" if label==1 else "Sain")
    with col_gauge:
        st.plotly_chart(render_gauge(prob), use_container_width=True)

    _sec("Interprétation clinique")
    if prob >= 0.75:
        st.error("**Risque très élevé (≥ 75 %)** — Consultation cardiologique urgente recommandée.")
    elif prob >= 0.50:
        st.warning("**Risque modéré à élevé (50–75 %)** — Examens complémentaires conseillés.")
    elif prob >= 0.30:
        st.warning("**Risque intermédiaire (30–50 %)** — Suivi cardiologique recommandé à 6 mois.")
    else:
        st.success("**Risque faible (< 30 %)** — Profil cardiovasculaire favorable.")

    _sec("Variables prédictives")
    df_imp = get_feature_importance_cached(model, feat_names, top_n=15)
    st.plotly_chart(render_importance_chart(df_imp), use_container_width=True)
    st.caption("Importances globales XGBoost (gain moyen). "
               "Pour des explications individuelles, utiliser les SHAP values.")


def page_analyse(model: Any, feat_names: np.ndarray) -> None:
    st.markdown(f"""
    <div class="page-header">
        <span class="page-header-badge">Analyse exploratoire</span>
        <h1>Variables <em>cliniques</em></h1>
        <p class="page-header-sub">
            Vue globale des variables utilisées par le modèle,
            de leur importance prédictive et de leur interprétation clinique.
        </p>
    </div>""", unsafe_allow_html=True)

    df_imp = get_feature_importance_cached(model, feat_names, top_n=15)
    col_chart, col_rank = st.columns([3,1], gap="large")
    with col_chart:
        st.plotly_chart(render_importance_chart(
            df_imp, "Top 15 variables — score d'importance XGBoost (gain)"),
            use_container_width=True)
    with col_rank:
        _sec("Classement")
        if not df_imp.empty:
            mx = df_imp["importance"].max()
            if mx > 0:
                bars_html = ""
                for i, row in df_imp.iterrows():
                    pct     = float(row["importance"] / mx) * 100
                    name    = row["feature"]
                    rank    = i + 1
                    opacity = max(0.4, 1.0 - (rank - 1) * 0.05)
                    bars_html += f"""
<div style="margin-bottom:11px;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
    <span style="font-size:11.5px;font-weight:500;white-space:nowrap;
                 overflow:hidden;text-overflow:ellipsis;max-width:130px;"
          title="{name}">{name}</span>
    <span style="font-size:10px;font-weight:700;opacity:.4;
                 flex-shrink:0;margin-left:5px;">#{rank}</span>
  </div>
  <div style="height:4px;border-radius:3px;
              background:rgba(128,128,128,0.12);overflow:hidden;">
    <div style="height:100%;width:{pct:.1f}%;border-radius:3px;
                background:linear-gradient(90deg,#dc3559,#ff6b8a);
                opacity:{opacity:.2f};">
    </div>
  </div>
</div>"""
                st.markdown(bars_html, unsafe_allow_html=True)

    st.markdown("---")
    _sec("Dictionnaire des variables")
    clinical_dict = pd.DataFrame({
        "Variable":    ["age","sex","cp","trestbps","chol","fbs","restecg","thalch","exang",
                        "oldpeak","slope","ca","thal","thalch_ratio","chol_per_age","ca_missing","thal_missing"],
        "Nom clinique":["Âge","Sexe","Douleur thoracique","PA repos","Cholestérol","Glycémie à jeun",
                        "ECG repos","FC max","Angine effort","Dépression ST","Pente ST","Vaisseaux",
                        "Scintigraphie","% FC max théorique","Chol / âge","Manque ca","Manque thal"],
        "Type":        ["Numérique","Catégorielle","Cat. (4)","Numérique","Numérique","Binaire",
                        "Cat. (3)","Numérique","Binaire","Numérique","Cat. (3)","Num. 0–3","Cat. (3)",
                        "Dérivée","Dérivée","Flag","Flag"],
        "Description": ["Âge en années","Male / Female","typical / atypical / non-anginal / asymptomatic",
                        "Pression systolique repos (mmHg)","Cholestérol sérique (mg/dL)",
                        "Glycémie > 120 mg/dL","Normal / LV hypertrophy / ST-T abnormality",
                        "FC max à l'effort (bpm)","Angine à l'effort : 1=Oui","Dépression ST (mm)",
                        "upsloping / flat / downsloping","Vaisseaux opacifiés (0–3)",
                        "normal / fixed / reversable defect","FC max / (220 – âge)",
                        "Cholestérol / âge",">66 % manquant UCI",">53 % manquant UCI"],
    })
    st.dataframe(clinical_dict, use_container_width=True, hide_index=True)


def page_apropos() -> None:
    st.markdown(f"""
    <div class="page-header">
        <span class="page-header-badge">Documentation</span>
        <h1>À propos de <em>CardioRisk AI</em></h1>
        <p class="page-header-sub">
            Architecture technique, performances, conformité réglementaire et éthique.
        </p>
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown(f"""
### Modèle et méthodologie
- **Algorithme** : XGBoost (eXtreme Gradient Boosting)
- **Optimisation** : GridSearchCV — 5-fold cross-validation
- **Hyperparamètres** : `learning_rate=0.05`, `max_depth=3`, `n_estimators=100`, `subsample=0.8`
- **Preprocessing** : `RobustScaler` + `SimpleImputer` + `OneHotEncoder`
- **Déséquilibre** : SMOTE sur le train uniquement
- **Modèles comparés** : LR, RF, GBM, SVM, KNN, XGBoost

### Dataset d'entraînement
- **Source** : UCI Heart Disease Repository
- **Effectif** : 920 patients — 4 centres
- **Target** : variable `num` binarisée (0 = sain, 1 = malade)
- **Variables originales** : 14 — **Variables finales** : 26
- **Version** : {APP_VERSION}
""")
    with c2:
        st.markdown("""
### Performances — jeu de test (n=184)
| Métrique | Valeur |
|---|---|
| Accuracy | **86.41 %** |
| ROC-AUC | **92.97 %** |
| F1-Score (malade) | **88 %** |
| Sensibilité | **91 %** |
| Spécificité | **80 %** |

### Limitations
- Population spécifique (4 centres US/Europe)
- Données manquantes : `ca` (66 %), `thal` (53 %)
- Non validé sur cohorte externe indépendante
- Seuil décisionnel fixé à 50 %
""")
    st.markdown("---")
    st.markdown("""
### Conformité et éthique

**RGPD — Article 25 : Privacy by Design**
- Aucune donnée identifiante collectée
- Logs d'audit : paramètres anonymisés, stockage local uniquement

**Statut réglementaire**
- Prototype de recherche (TRL 4–5) — Non certifié MDR (EU 2017/745)
- Usage clinique direct sans validation réglementaire interdit
""")
    st.caption(f"CardioRisk AI v{APP_VERSION} — "
               f"Python {sys.version_info.major}.{sys.version_info.minor} · "
               "Streamlit · XGBoost · scikit-learn · Plotly")


def page_audit() -> None:
    st.markdown(f"""
    <div class="page-header">
        <span class="page-header-badge">RGPD — Art. 25</span>
        <h1>Journal <em>d'audit</em></h1>
        <p class="page-header-sub">
            Historique anonymisé des prédictions. Aucun identifiant patient.
        </p>
    </div>""", unsafe_allow_html=True)

    if not LOG_FILE.exists():
        st.info("Aucune prédiction enregistrée pour le moment."); return
    try:
        df_log = pd.read_csv(LOG_FILE, encoding="utf-8")
    except pd.errors.EmptyDataError:
        st.info("Le journal est vide."); return
    except Exception as exc:
        st.error(f"Impossible de lire le journal : {exc}"); return

    if df_log.empty:
        st.info("Aucune entrée."); return

    total  = len(df_log)
    n_sick = int((df_log.get("prediction", pd.Series()) == "Malade").sum())
    _sec("Résumé statistique")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total prédictions", total)
    c2.metric("Profils à risque",  n_sick)
    c3.metric("Profils sains",     total - n_sick)
    c4.metric("Taux de risque",    f"{n_sick/total*100:.0f} %" if total > 0 else "—")
    st.markdown("---")
    _sec("Entrées du journal")
    if "timestamp" in df_log.columns:
        df_log = df_log.sort_values("timestamp", ascending=False)
    st.dataframe(df_log, use_container_width=True, hide_index=True)
    st.download_button(
        "Exporter le journal CSV",
        data=df_log.to_csv(index=False).encode("utf-8"),
        file_name=f"audit_cardiorisk_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    _inject_css()

    with st.sidebar:
        st.markdown("""
        <div class="sb-logo">
            <div class="sb-logo-icon">🫀</div>
            <div class="sb-logo-text">
                <span class="sb-logo-name">CardioRisk AI</span>
                <span class="sb-logo-sub">Aide à la décision clinique</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<span class="sb-section">Navigation</span>', unsafe_allow_html=True)
        NAV = ["Accueil", "Prediction", "Analyse des variables",
               "Journal d'audit", "A propos"]
        page = st.radio("nav", options=NAV, label_visibility="collapsed")

        st.markdown('<div class="sb-spacer"></div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="sb-footer">
            <div class="sb-stats">
                <div class="sb-stat">
                    <span class="sb-stat-label">Accuracy</span>
                    <span class="sb-stat-val">86.4%</span>
                </div>
                <div class="sb-stat">
                    <span class="sb-stat-label">ROC-AUC</span>
                    <span class="sb-stat-val">93.0%</span>
                </div>
                <div class="sb-stat">
                    <span class="sb-stat-label">Modèle</span>
                    <span class="sb-stat-val">XGBoost</span>
                </div>
                <div class="sb-stat">
                    <span class="sb-stat-label">Version</span>
                    <span class="sb-stat-val">{APP_VERSION}</span>
                </div>
            </div>
            <span class="sb-tag">⚠ Prototype — Non certifié MDR</span>
        </div>
        """, unsafe_allow_html=True)

    try:
        model, preprocessor, le_encoders, feat_names = load_artifacts()
    except FileNotFoundError as exc:
        st.error(f"**Artefacts ML introuvables**\n\n{exc}"); st.stop()
    except RuntimeError as exc:
        st.error(f"**Erreur de chargement** : {exc}"); st.stop()

    routes = {
        "Accueil":               page_accueil,
        "Prediction":            lambda: page_prediction(model, preprocessor, le_encoders, feat_names),
        "Analyse des variables": lambda: page_analyse(model, feat_names),
        "Journal d'audit":       page_audit,
        "A propos":              page_apropos,
    }
    handler = routes.get(page)
    if handler:
        handler()
    else:
        st.error(f"Page inconnue : '{page}'")


if __name__ == "__main__":
    main()