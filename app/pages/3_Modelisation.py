"""
Objectif :
- Montrer les résultats ML (accuracy, matrice de confusion, courbe ROC)
- Expliquer quel modèle a été choisi et pourquoi
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from components.cards import load_css, hero, coming_soon

st.set_page_config(page_title="Modélisation", page_icon="🤖", layout="wide")
load_css()

hero(
    kicker="Machine Learning",
    title="Modélisation",
    subtitle="Comparaison des modèles entraînés sur les deux cohortes de patients.",
)

coming_soon(
    "Résultats des modèles à venir",
    "Cette page présentera les performances des modèles entraînés sur Cardio Train et BRFSS, "
    "ainsi que les critères ayant guidé le choix du modèle final.",
    [
        "Accuracy, precision, recall, F1-score, ROC-AUC par modèle",
        "Matrice de confusion et courbe ROC interactives",
        "Feature importance — quelles variables pèsent le plus",
        "Comparaison Logistic Regression / Random Forest / XGBoost / LightGBM",
    ],
)
