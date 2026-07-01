"""
Objectif :
- Interface utilisateur pour prédire le risque cardiovasculaire individuel
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from components.cards import load_css, hero, coming_soon

st.set_page_config(page_title="Prédiction", page_icon="🩺", layout="wide")
load_css()

hero(
    kicker="Simulateur",
    title="Prédiction du risque",
    subtitle="Estime ton risque cardiovasculaire personnel à partir de quelques informations.",
)

coming_soon(
    "Simulateur en construction",
    "Une fois les modèles entraînés (page Modélisation), cette page permettra de saisir "
    "un profil et d'obtenir une estimation du risque cardiovasculaire en temps réel.",
    [
        "Formulaire guidé : âge, sexe, IMC, tension, cholestérol, habitudes de vie",
        "Choix entre le modèle clinique (Cardio Train) et le modèle lifestyle (BRFSS)",
        "Résultat sous forme de score de risque avec explication des facteurs",
        "Rappel : usage pédagogique, ne remplace pas un avis médical",
    ],
)
