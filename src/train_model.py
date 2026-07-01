"""
Objectif :
- Entraîner plusieurs modèles ML sur chaque dataset patients
- Comparer leurs performances
- Sauvegarder le meilleur modèle (+ scaler, features, métriques) par dataset
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.evaluate_model import compute_confusion_matrix, compute_metrics, compute_roc_curve
from src.feature_engineering import add_bmi_category, add_bp_category, add_unhealthy_days

MODELS_DIR = Path("models")


def engineer_cardio(df):
    df = add_bp_category(df)
    df = add_bmi_category(df)
    return df


def engineer_brfss(df):
    df = add_bmi_category(df)
    df = add_unhealthy_days(df)
    return df


DATASETS = {
    "cardio_train": {
        "path": "data/processed/cardio_train.csv",
        "target": "cardio",
        "label": "Modèle clinique (Cardio Train)",
        "engineer": engineer_cardio,
    },
    "heart_disease": {
        "path": "data/processed/heart_disease.csv",
        "target": "HeartDiseaseorAttack",
        "label": "Modèle lifestyle (BRFSS)",
        "engineer": engineer_brfss,
    },
}


def build_models(scale_pos_weight):
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1, eval_metric="logloss",
            scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=-1
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300, learning_rate=0.1, class_weight="balanced",
            random_state=42, n_jobs=-1, verbose=-1
        ),
    }


def train_dataset(name, config):
    print(f"\n{'=' * 70}\n{config['label']} — {name}\n{'=' * 70}")

    df = pd.read_csv(config["path"])
    df = config["engineer"](df)
    y = df[config["target"]].astype(int)
    X = df.drop(columns=[config["target"]])
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    results = {}
    fitted_models = {}
    for model_name, model in build_models(scale_pos_weight).items():
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        y_proba = model.predict_proba(X_test_s)[:, 1]

        metrics = compute_metrics(y_test, y_pred, y_proba)
        results[model_name] = {
            **metrics,
            "confusion_matrix": compute_confusion_matrix(y_test, y_pred),
            "roc_curve": compute_roc_curve(y_test, y_proba),
        }
        fitted_models[model_name] = model
        print(
            f"{model_name:22s} | accuracy={metrics['accuracy']:.3f}  "
            f"f1={metrics['f1']:.3f}  roc_auc={metrics['roc_auc']:.3f}"
        )

    # Le ROC-AUC est le critère de sélection : robuste au déséquilibre des classes,
    # contrairement à l'accuracy (voir dataset BRFSS, ~10% de cas positifs).
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best_model = fitted_models[best_name]
    print(f"\n→ Meilleur modèle : {best_name} (ROC-AUC = {results[best_name]['roc_auc']:.3f})")

    importance = None
    if hasattr(best_model, "feature_importances_"):
        importance = dict(zip(feature_names, best_model.feature_importances_.tolist()))
    elif hasattr(best_model, "coef_"):
        importance = dict(zip(feature_names, best_model.coef_[0].tolist()))

    out_dir = MODELS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, out_dir / "best_model.pkl")
    joblib.dump(scaler, out_dir / "scaler.pkl")
    joblib.dump(feature_names, out_dir / "feature_names.pkl")

    summary = {
        "dataset": name,
        "label": config["label"],
        "target": config["target"],
        "n_train": len(X_train),
        "n_test": len(X_test),
        "best_model": best_name,
        "feature_importance": importance,
        "models": results,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    for name, config in DATASETS.items():
        train_dataset(name, config)


if __name__ == "__main__":
    main()
