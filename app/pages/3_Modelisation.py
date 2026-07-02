"""
Objectif :
- Montrer les résultats ML (accuracy, matrice de confusion, courbe ROC)
- Expliquer quel modèle a été choisi et pourquoi
"""

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.append(str(Path(__file__).parent.parent))
from components.cards import load_css
from components.theme import COLORS, style

st.set_page_config(page_title="Modélisation", page_icon="🤖", layout="wide")
load_css()

st.title("Modélisation")
st.caption("Comparaison de 4 modèles de Machine Learning, entraînés séparément sur chaque dataset patients")

st.divider()

DATASETS = {
    "cardio_train": {"path": "models/cardio_train/metrics.json", "icon": "🏥", "title": "Modèle clinique — Cardio Train"},
    "heart_disease": {"path": "models/heart_disease/metrics.json", "icon": "📋", "title": "Modèle lifestyle — BRFSS"},
}

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Précision",
    "recall": "Rappel",
    "f1": "F1-score",
    "roc_auc": "ROC-AUC",
}


@st.cache_data
def load_metrics(path):
    with open(path) as f:
        return json.load(f)


missing = [d for d in DATASETS.values() if not Path(d["path"]).exists()]
if missing:
    st.info("🚧 Aucun modèle entraîné pour l'instant. Lance `python -m src.train_model` pour générer les résultats.")
    st.stop()

tab1, tab2 = st.tabs([f"{d['icon']} {d['title']}" for d in DATASETS.values()])

for tab, (name, config) in zip([tab1, tab2], DATASETS.items()):
    with tab:
        summary = load_metrics(config["path"])
        models = summary["models"]
        best_name = summary["best_model"]

        st.caption(
            f"Entraîné sur {summary['n_train']:,} patients · testé sur {summary['n_test']:,} · "
            f"cible : `{summary['target']}`"
        )

        # ── Tableau comparatif ────────────────────────────────────────────────
        st.subheader("Comparaison des modèles")

        comp = pd.DataFrame({
            model_name: {METRIC_LABELS[k]: v for k, v in metrics.items() if k in METRIC_LABELS}
            for model_name, metrics in models.items()
        }).T
        comp = comp.round(3)

        # Colonne validation croisée (moyenne ± écart-type sur 5 folds) si disponible
        if all("cv_roc_auc_mean" in m for m in models.values()):
            comp["ROC-AUC (CV 5-fold)"] = [
                f"{models[m]['cv_roc_auc_mean']:.3f} ± {models[m]['cv_roc_auc_std']:.3f}"
                for m in comp.index
            ]

        numeric_cols = [c for c in comp.columns if comp[c].dtype != object]
        st.dataframe(
            comp.style.highlight_max(axis=0, color=COLORS["accent"] + "22", subset=numeric_cols),
            use_container_width=True,
        )
        st.caption(
            f"🏆 Modèle retenu : **{best_name}** — sélection basée sur le ROC-AUC "
            "(robuste au déséquilibre des classes, contrairement à l'accuracy seule). "
            "La colonne CV donne la moyenne ± écart-type sur 5 folds stratifiés du jeu d'entraînement : "
            "un écart-type faible indique des performances stables, non dues au hasard du découpage."
        )

        st.divider()

        best = models[best_name]

        col_roc, col_cm = st.columns(2)

        # ── Courbe ROC ──────────────────────────────────────────────────────
        with col_roc:
            st.subheader("Courbe ROC")
            fig_roc = go.Figure()
            for model_name, metrics in models.items():
                fig_roc.add_trace(go.Scatter(
                    x=metrics["roc_curve"]["fpr"],
                    y=metrics["roc_curve"]["tpr"],
                    mode="lines",
                    name=f"{model_name} (AUC={metrics['roc_auc']:.3f})",
                    line=dict(width=3 if model_name == best_name else 1.5),
                ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines",
                name="Aléatoire", line=dict(dash="dash", color=COLORS["grid"]),
            ))
            fig_roc.update_layout(
                xaxis_title="Taux de faux positifs",
                yaxis_title="Taux de vrais positifs",
                legend=dict(orientation="h", yanchor="bottom", y=-0.4),
            )
            style(fig_roc, height=420)
            st.plotly_chart(fig_roc, use_container_width=True)

        # ── Matrice de confusion ──────────────────────────────────────────────
        with col_cm:
            st.subheader(f"Matrice de confusion — {best_name}")
            cm = best["confusion_matrix"]
            fig_cm = px.imshow(
                cm,
                x=["Prédit sain", "Prédit malade"],
                y=["Sain", "Malade"],
                color_continuous_scale=["#f9fafb", COLORS["accent"]],
                text_auto=True,
            )
            fig_cm.update_layout(coloraxis_showscale=False)
            style(fig_cm, height=420)
            st.plotly_chart(fig_cm, use_container_width=True)

        st.divider()

        # ── Feature importance ────────────────────────────────────────────────
        st.subheader(f"Facteurs les plus influents — {best_name}")
        importance = summary.get("feature_importance")
        if importance:
            imp_df = (
                pd.Series(importance)
                .abs()
                .sort_values(ascending=False)
                .head(12)
                .sort_values()
                .reset_index()
            )
            imp_df.columns = ["Variable", "Importance"]
            fig_imp = px.bar(imp_df, x="Importance", y="Variable", orientation="h")
            fig_imp.update_traces(marker_color=COLORS["accent"])
            style(fig_imp, height=420, showlegend=False)
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.caption("Feature importance non disponible pour ce modèle.")
