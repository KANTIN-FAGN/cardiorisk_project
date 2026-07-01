"""
Objectif :
- Montrer la maturité du projet : sources, méthodologie, résultats, limites, biais, améliorations futures
"""

import json
import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).parent.parent))
from components.cards import load_css

st.set_page_config(page_title="À propos", page_icon="📖", layout="wide")
load_css()

st.title("À propos")
st.caption("Sources, méthodologie, résultats, limites et avertissements")

st.divider()

st.subheader("Sources")
st.markdown("""
| Dataset | Source | Portée |
|---|---|---|
| Heart Disease Health Indicators (BRFSS 2015) | [Kaggle](https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-dataset) | 253 680 répondants, USA |
| Cardiovascular Disease Dataset | [Kaggle](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset) | 70 000 patients, Russie |
| Mortalité cardiovasculaire mondiale | OMS (GHE) / Our World in Data | 194 pays, 2000–2023 |
""")

st.divider()

st.subheader("Méthodologie")
st.markdown("""
1. **Nettoyage** — suppression des doublons et valeurs aberrantes (tension artérielle incohérente, IMC hors bornes physiologiques), encodage des variables catégorielles.
2. **Exploration** — analyse univariée et bivariée, tests statistiques (Chi², Mann-Whitney) pour valider l'association entre chaque facteur et le risque cardiovasculaire (page *Analyse des patients*).
3. **Modélisation** — 4 algorithmes entraînés et comparés par dataset (Logistic Regression, Random Forest, XGBoost, LightGBM), avec gestion du déséquilibre des classes (`class_weight`/`scale_pos_weight`). Sélection du meilleur modèle par **ROC-AUC**, plus robuste que l'accuracy face au déséquilibre du dataset BRFSS.
4. **Application** — restitution interactive des analyses et des modèles sous forme de dashboard Streamlit (carte mondiale, exploration patients, comparaison de modèles, simulateur de prédiction).
""")

st.divider()

st.subheader("Résultats des modèles")


@st.cache_data
def load_metrics(path):
    with open(path) as f:
        return json.load(f)


try:
    m_cardio = load_metrics("models/cardio_train/metrics.json")
    m_hd = load_metrics("models/heart_disease/metrics.json")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🏥 Modèle clinique — Cardio Train**")
        best = m_cardio["models"][m_cardio["best_model"]]
        st.markdown(f"""
- Meilleur modèle : **{m_cardio['best_model']}**
- ROC-AUC : **{best['roc_auc']:.3f}** · Accuracy : {best['accuracy']:.3f} · F1 : {best['f1']:.3f}
- Entraîné sur {m_cardio['n_train']:,} patients, testé sur {m_cardio['n_test']:,}
""")
    with col2:
        st.markdown("**📋 Modèle lifestyle — BRFSS**")
        best_hd = m_hd["models"][m_hd["best_model"]]
        st.markdown(f"""
- Meilleur modèle : **{m_hd['best_model']}**
- ROC-AUC : **{best_hd['roc_auc']:.3f}** · Accuracy : {best_hd['accuracy']:.3f} · F1 : {best_hd['f1']:.3f}
- Entraîné sur {m_hd['n_train']:,} répondants, testé sur {m_hd['n_test']:,}
""")
    st.caption("Détail complet des 4 modèles comparés, courbes ROC et matrices de confusion : page *Modélisation*.")
except FileNotFoundError:
    st.info("🚧 Modèles pas encore entraînés — lance `python -m src.train_model`.")

st.divider()

st.subheader("Limites")
st.markdown("""
- Les modèles reposent sur des **données historiques et déclaratives**, pas sur un bilan médical complet.
- Les prédictions **ne remplacent pas un diagnostic médical**.
- Certains facteurs de risque connus (génétique, antécédents familiaux détaillés, imagerie médicale) ne sont pas présents dans les datasets utilisés.
- Le dataset BRFSS est **fortement déséquilibré** (10% de cas positifs) : les métriques de performance doivent être lues en tenant compte de ce déséquilibre.
- Les "profils d'exemple" de la page *Prédiction* sont issus du dataset d'entraînement : ils illustrent le comportement du modèle mais ne constituent pas une évaluation indépendante.
""")

st.divider()

st.subheader("Biais possibles")
st.markdown("""
- **Biais géographique** : le dataset Cardio Train (Russie) et le BRFSS (USA) ne représentent pas la diversité mondiale des populations.
- **Biais déclaratif** : le tabac, l'alcool ou l'activité physique sont auto-déclarés dans le BRFSS, avec un risque de sous-déclaration.
- **Biais temporel** : les données mondiales et patients ne couvrent pas exactement les mêmes périodes.
""")

st.divider()

st.subheader("Améliorations futures")
st.markdown("""
- Valider les modèles sur une cohorte externe pour mesurer la généralisation.
- Ajouter un intervalle de confiance ou une validation croisée aux métriques de performance.
- Enrichir les données mondiales avec des facteurs socio-économiques supplémentaires (accès aux soins, pollution).
- Explorer des variables dérivées (feature engineering) pour affiner la prédiction lifestyle.
""")

st.divider()

st.info(
    "🔒 **Avertissement médical** — Cette application a une vocation pédagogique et de "
    "sensibilisation. Les résultats fournis ne constituent pas un diagnostic médical et ne "
    "doivent pas remplacer l'avis d'un professionnel de santé."
)

st.caption("Projet de Data Science appliqué à la santé cardiovasculaire — Ynov Bachelor 3, année universitaire 2025–2026.")
