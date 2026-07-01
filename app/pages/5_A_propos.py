"""
Objectif :
- Montrer la maturité du projet : sources, limites, biais, améliorations futures
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from components.cards import load_css, hero, section_header

st.set_page_config(page_title="À propos", page_icon="📖", layout="wide")
load_css()

hero(
    kicker="Méthodologie & transparence",
    title="À propos",
    subtitle="Sources, limites et avertissements — pour lire ce projet avec le recul nécessaire.",
)

section_header("01 — Sources", "D'où viennent les données")
st.markdown("""
| Dataset | Source | Portée |
|---|---|---|
| Heart Disease Health Indicators (BRFSS 2015) | [Kaggle](https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-dataset) | 253 680 répondants, USA |
| Cardiovascular Disease Dataset | [Kaggle](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset) | 70 000 patients, Russie |
| Mortalité cardiovasculaire mondiale | OMS (GHE) / Our World in Data | 194 pays, 2000–2023 |
""")

st.divider()

section_header("02 — Limites", "Ce que ce projet ne fait pas")
st.markdown("""
- Les modèles reposent sur des **données historiques et déclaratives**, pas sur un bilan médical complet.
- Les prédictions **ne remplacent pas un diagnostic médical**.
- Certains facteurs de risque connus (génétique, antécédents familiaux détaillés, imagerie médicale) ne sont pas présents dans les datasets utilisés.
- Le dataset BRFSS est **fortement déséquilibré** (10% de cas positifs) : les métriques de performance doivent être lues en tenant compte de ce déséquilibre.
""")

st.divider()

section_header("03 — Biais possibles", "Points de vigilance")
st.markdown("""
- **Biais géographique** : le dataset Cardio Train (Russie) et le BRFSS (USA) ne représentent pas la diversité mondiale des populations.
- **Biais déclaratif** : le tabac, l'alcool ou l'activité physique sont auto-déclarés dans le BRFSS, avec un risque de sous-déclaration.
- **Biais temporel** : les données mondiales et patients ne couvrent pas exactement les mêmes périodes.
""")

st.divider()

section_header("04 — Améliorations futures", "Pistes d'évolution")
st.markdown("""
- Entraîner et déployer les modèles de prédiction (Logistic Regression, Random Forest, XGBoost, LightGBM).
- Ajouter un simulateur interactif de scénarios individuels.
- Enrichir les données mondiales avec des facteurs socio-économiques supplémentaires (accès aux soins, pollution).
- Valider les modèles sur une cohorte externe pour mesurer la généralisation.
""")

st.divider()

st.info(
    "🔒 **Avertissement médical** — Cette application a une vocation pédagogique et de "
    "sensibilisation. Les résultats fournis ne constituent pas un diagnostic médical et ne "
    "doivent pas remplacer l'avis d'un professionnel de santé."
)

st.caption("Projet de Data Science appliqué à la santé cardiovasculaire — Ynov Bachelor 3, année universitaire 2025–2026.")
