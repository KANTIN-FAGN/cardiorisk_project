"""
Objectif :
- Faire des prédictions à partir de nouvelles données (Streamlit)
"""

import json
from pathlib import Path

import joblib
import pandas as pd

MODELS_DIR = Path("models")


def load_bundle(dataset_name):
    """Charge le modèle, le scaler, les features et les métriques d'un dataset entraîné."""
    d = MODELS_DIR / dataset_name
    with open(d / "metrics.json") as f:
        metrics = json.load(f)
    best_name = metrics["best_model"]
    return {
        "model": joblib.load(d / "best_model.pkl"),
        "scaler": joblib.load(d / "scaler.pkl"),
        "feature_names": joblib.load(d / "feature_names.pkl"),
        "model_name": best_name,
        "roc_auc": metrics["models"][best_name]["roc_auc"],
    }


def predict_risk(bundle, input_dict):
    """Retourne (probabilité de risque, classe prédite 0/1) pour un profil donné."""
    row = pd.DataFrame(
        [[input_dict[f] for f in bundle["feature_names"]]],
        columns=bundle["feature_names"],
    )
    row_scaled = bundle["scaler"].transform(row)
    proba = float(bundle["model"].predict_proba(row_scaled)[0, 1])
    return proba, int(proba >= 0.5)


def pick_example_profiles(df, target_col, bundle):
    """
    Sélectionne 3 profils réels représentatifs du dataset, notés par le modèle :
    - Sain    : risque prédit le plus bas
    - Modéré  : risque prédit le plus proche de 50%
    - Malade  : risque prédit le plus haut
    """
    X = df[bundle["feature_names"]]
    X_scaled = bundle["scaler"].transform(X)
    proba = bundle["model"].predict_proba(X_scaled)[:, 1]

    scored = df.copy()
    scored["_risk"] = proba

    picks = {
        "Sain": scored["_risk"].idxmin(),
        "Modéré": (scored["_risk"] - 0.5).abs().idxmin(),
        "Malade": scored["_risk"].idxmax(),
    }

    profiles = {}
    for level, idx in picks.items():
        row = scored.loc[idx]
        profiles[level] = {
            "risk": float(row["_risk"]),
            "data": row.drop(labels=["_risk"]).to_dict(),
        }
    return profiles
