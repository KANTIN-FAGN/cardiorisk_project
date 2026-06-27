import streamlit as st
import pandas as pd
import plotly.express as px
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

LAYOUT = dict(paper_bgcolor="#161b22", plot_bgcolor="#161b22", font_color="#e2e8f0")

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/cardio_train.csv")

df = load_data()

st.title("CardioRisk Explorer")
st.caption("Analyse et prédiction basées sur les données de santé")

st.divider()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Patients", f"{len(df):,}")
c2.metric("Variables", len(df.columns))
c3.metric("Prévalence cardio", f"{df['cardio'].mean()*100:.1f}%")
c4.metric("Âge moyen", f"{df['age'].mean():.0f} ans")

st.divider()

col_left, col_right = st.columns([1.2, 1])

with col_left:
    st.subheader("À propos")
    st.markdown("""
Dataset d'examens médicaux collecté en Russie sur des adultes de 30 à 65 ans.

| Catégorie | Variables |
|---|---|
| Objectif | Âge, Taille, Poids, Pression artérielle |
| Examen | Cholestérol, Glucose |
| Lifestyle | Tabac, Alcool, Activité physique |
| Cible | Maladie cardiovasculaire (oui / non) |
    """)

with col_right:
    st.subheader("Distribution de la cible")
    target_df = df["cardio"].map({0: "Sain", 1: "Malade"}).value_counts().reset_index()
    target_df.columns = ["Statut", "Nombre"]
    fig = px.pie(
        target_df, values="Nombre", names="Statut",
        color="Statut",
        color_discrete_map={"Sain": "#3b82f6", "Malade": "#ef4444"},
        hole=0.5, template="plotly_dark"
    )
    fig.update_traces(textinfo="percent+label", textfont_size=13)
    fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=280, showlegend=False, **LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("Aperçu")
st.dataframe(df.head(8).rename(columns={
    "age": "Âge", "height": "Taille", "weight": "Poids",
    "ap_hi": "PA syst.", "ap_lo": "PA diast.",
    "cholesterol": "Cholestérol", "gluc": "Glucose",
    "smoke": "Tabac", "alco": "Alcool", "active": "Activité",
    "cardio": "Cardio", "gender_male": "Homme", "BMI": "IMC"
}), use_container_width=True, hide_index=True)
