import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from components.cards import load_css

st.set_page_config(
    page_title="CardioRisk Explorer",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_css()

# ── En-tête ───────────────────────────────────────────────────────────────────

st.title("CardioRisk Explorer")
st.markdown("""
Projet d'analyse et de prédiction des maladies cardiovasculaires,
combinant des données mondiales et deux populations de patients distinctes.
""")

st.divider()

# ── Contexte ──────────────────────────────────────────────────────────────────

st.subheader("Contexte")
st.markdown("""
Les maladies cardiovasculaires sont la **première cause de mortalité dans le monde**,
responsables de près de **18 millions de décès par an** selon l'OMS.

Ce projet explore les facteurs de risque à deux niveaux :
- **À l'échelle mondiale** : évolution de la mortalité par pays, par revenu, par sexe
- **À l'échelle individuelle** : profils de patients et prédiction du risque personnel
""")

st.divider()

# ── Navigation rapide ─────────────────────────────────────────────────────────

st.subheader("Que souhaitez-vous faire ?")

nav_cards = [
    ("🌍", "Analyse mondiale", "Mortalité cardiovasculaire par pays, richesse et sexe.", "pages/1_Analyse_globale.py", "#eff6ff"),
    ("🔬", "Analyse patients", "Explorer les facteurs de risque sur deux cohortes.", "pages/2_Analyse_patients.py", "#f0fdf4"),
    ("🤖", "Modélisation", "Comparaison des modèles de Machine Learning.", "pages/3_Modelisation.py", "#faf5ff"),
    ("🩺", "Prédiction", "Estimer un risque cardiovasculaire individuel.", "pages/4_Prediction.py", "#fff7ed"),
]

cols = st.columns(4)
for col, (icon, title, desc, page, bg) in zip(cols, nav_cards):
    with col:
        with st.container(border=True):
            st.markdown(
                f"<div style='font-size:1.8rem;background:{bg};width:2.8rem;height:2.8rem;"
                f"display:flex;align-items:center;justify-content:center;border-radius:12px;"
                f"margin-bottom:0.5rem'>{icon}</div>"
                f"<div style='font-weight:600;font-size:1.02rem;margin-bottom:0.2rem'>{title}</div>"
                f"<div style='color:#6b7280;font-size:0.85rem;min-height:2.6rem'>{desc}</div>",
                unsafe_allow_html=True,
            )
            st.page_link(page, label="Accéder →")

st.divider()

# ── Datasets ──────────────────────────────────────────────────────────────────

st.subheader("Les données utilisées")

# --- Données mondiales ---
st.markdown("#### 🌍 Données mondiales — Our World in Data / OMS")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
**Taux de mortalité standardisé**

Taux de décès cardiovasculaires pour 100 000 habitants,
ajusté pour neutraliser les différences de structure d'âge entre pays.

- Source : OMS / GHE
- Couverture : 194 pays, 2000–2021
- Utilisation : carte mondiale comparative
""")

with col2:
    st.markdown("""
**Mortalité vs PIB par habitant**

Croise le taux de mortalité cardiovasculaire avec le niveau de richesse des pays.

- Source : Our World in Data
- Couverture : 194 pays, 2000–2021
- Utilisation : scatter plot richesse / santé
""")

with col3:
    st.markdown("""
**Évolution Hommes vs Femmes**

Suivi historique de la mortalité cardiovasculaire séparément
pour les hommes et les femmes.

- Source : Our World in Data
- Couverture : 117 pays, 1950–2023
- Utilisation : courbes d'évolution
""")

st.divider()

# --- Données patients ---
st.markdown("#### 🏥 Données patients — 2 populations distinctes")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("""
**Cardio Train — données médicales (Russie)**

Dataset issu d'examens médicaux en milieu hospitalier.
Chaque ligne = un patient avec des mesures cliniques objectives.

| Variable | Description |
|---|---|
| Âge | En années |
| Taille / Poids | Mesures objectives |
| Pression artérielle | Systolique et diastolique |
| Cholestérol | 3 niveaux (normal / élevé / très élevé) |
| Glucose | 3 niveaux |
| Tabac / Alcool / Activité | Auto-déclaré |
| **Cardio** | **Cible : maladie cardiovasculaire** |

- **68 606 patients** · Équilibré (50% positifs)
- Adapté à un **modèle médical précis**
""")

with col_b:
    st.markdown("""
**BRFSS 2015 — lifestyle auto-déclaré (USA)**

Enquête téléphonique nationale américaine sur les comportements
de santé. Données déclaratives, sans mesures cliniques précises.

| Variable | Description |
|---|---|
| Hypertension | Oui / Non (déclaré) |
| Cholestérol élevé | Oui / Non (déclaré) |
| IMC | Calculé |
| Tabac / Alcool | Oui / Non |
| Activité physique | Oui / Non |
| Diabète / AVC | Antécédents |
| **HeartDiseaseorAttack** | **Cible : cardiopathie** |

- **229 781 répondants** · Déséquilibré (10% positifs)
- Adapté à un **modèle grand public** sans bilan médical
""")

st.divider()

# ── Objectifs du projet ───────────────────────────────────────────────────────

st.subheader("Objectifs du projet")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
**1. Explorer**

Analyser les facteurs de risque cardiovasculaire à l'échelle mondiale
et sur les deux populations de patients.
""")

with col2:
    st.markdown("""
**2. Comprendre**

Identifier quelles variables sont statistiquement associées
au risque cardiovasculaire.
""")

with col3:
    st.markdown("""
**3. Prédire**

Entraîner deux modèles ML :
- Un pour les personnes avec un bilan médical complet
- Un pour toute personne avec des données de style de vie
""")

st.divider()
st.caption("Navigue dans les pages via le menu latéral pour explorer les données et les modèles.")
