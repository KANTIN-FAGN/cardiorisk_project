"""
Analyse exploratoire du dataset Cardio Train (EDA)
Reproduit les analyses du notebook 02_eda.ipynb
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import chi2_contingency, mannwhitneyu
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from components.cards import load_css

st.set_page_config(page_title="Analyse patients", page_icon="🔬", layout="wide")
load_css()

st.title("Analyse des patients")
st.caption("Exploration du dataset cardiovasculaire — 68 606 patients")

PLOTLY_LAYOUT = dict(paper_bgcolor="#161b22", plot_bgcolor="#161b22", font_color="#e2e8f0")

# ── Chargement des données ──────────────────────────────────────────────────

@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/cardio_train.csv")
    return df

df = load_data()

# Mapping des labels lisibles
CHOL_LABELS = {1: "Normal", 2: "Élevé", 3: "Très élevé"}
GLUC_LABELS = {1: "Normal", 2: "Élevé", 3: "Très élevé"}
GENDER_LABELS = {0: "Femme", 1: "Homme"}
SMOKE_LABELS = {0: "Non-fumeur", 1: "Fumeur"}
ALCO_LABELS = {0: "Non", 1: "Oui"}
ACTIVE_LABELS = {0: "Sédentaire", 1: "Actif"}
CARDIO_LABELS = {0: "Sain", 1: "Malade"}

# ── Filtres sidebar ──────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Filtres")
    age_range = st.slider(
        "Âge",
        int(df["age"].min()), int(df["age"].max()),
        (int(df["age"].min()), int(df["age"].max()))
    )
    gender_filter = st.multiselect(
        "Sexe",
        options=[0, 1],
        default=[0, 1],
        format_func=lambda x: GENDER_LABELS[x]
    )
    cardio_filter = st.multiselect(
        "Statut cardiovasculaire",
        options=[0, 1],
        default=[0, 1],
        format_func=lambda x: CARDIO_LABELS[x]
    )

mask = (
    df["age"].between(*age_range) &
    df["gender_male"].isin(gender_filter) &
    df["cardio"].isin(cardio_filter)
)
dff = df[mask].copy()

st.caption(f"{len(dff):,} patients sélectionnés sur {len(df):,}")

st.divider()

# ── 1. Vue d'ensemble ────────────────────────────────────────────────────────

st.subheader("Vue d'ensemble du dataset")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Patients filtrés", f"{len(dff):,}")
col2.metric("Prévalence cardio", f"{dff['cardio'].mean()*100:.1f}%")
col3.metric("Âge moyen", f"{dff['age'].mean():.1f} ans")
col4.metric("IMC moyen", f"{dff['BMI'].mean():.1f}")

with st.expander("Statistiques descriptives"):
    rename_cols = {
        "age": "Âge", "height": "Taille (cm)", "weight": "Poids (kg)",
        "ap_hi": "Pression systolique", "ap_lo": "Pression diastolique",
        "cholesterol": "Cholestérol", "gluc": "Glucose",
        "smoke": "Tabac", "alco": "Alcool", "active": "Activité",
        "cardio": "Cardio", "gender_male": "Homme", "BMI": "IMC"
    }
    st.dataframe(
        dff.rename(columns=rename_cols).describe().round(2),
        use_container_width=True
    )

st.divider()

# ── 2. Distribution de la cible ──────────────────────────────────────────────

st.subheader("Distribution des maladies cardiovasculaires")

col_a, col_b = st.columns(2)

with col_a:
    target_df = dff["cardio"].map(CARDIO_LABELS).value_counts().reset_index()
    target_df.columns = ["Statut", "Nombre"]
    fig = px.bar(
        target_df, x="Statut", y="Nombre",
        color="Statut",
        color_discrete_map={"Sain": "#00d4ff", "Malade": "#ef4444"},
        text="Nombre", title="Répartition sains / malades",
        template="plotly_dark"
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, **PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    gender_df = dff["gender_male"].map(GENDER_LABELS).value_counts().reset_index()
    gender_df.columns = ["Sexe", "Nombre"]
    fig2 = px.bar(
        gender_df, x="Sexe", y="Nombre",
        color="Sexe",
        color_discrete_map={"Femme": "#f472b6", "Homme": "#60a5fa"},
        text="Nombre", title="Répartition par sexe",
        template="plotly_dark"
    )
    fig2.update_traces(textposition="outside")
    fig2.update_layout(showlegend=False, **PLOTLY_LAYOUT)
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── 3. Distribution de l'âge ─────────────────────────────────────────────────

st.subheader("Distribution de l'âge par sexe")

age_df = dff.copy()
age_df["Sexe"] = age_df["gender_male"].map(GENDER_LABELS)

fig_age = px.histogram(
    age_df, x="age", color="Sexe",
    color_discrete_map={"Femme": "#f472b6", "Homme": "#60a5fa"},
    nbins=35, barmode="overlay", opacity=0.7,
    labels={"age": "Âge (années)"},
    title="Distribution de l'âge selon le sexe",
    template="plotly_dark"
)
fig_age.update_layout(**PLOTLY_LAYOUT)
st.plotly_chart(fig_age, use_container_width=True)

st.divider()

# ── 4. Risque par variable catégorielle ───────────────────────────────────────

st.subheader("Risque cardiovasculaire par facteur")

def plot_categorical_risk(df, col, labels, title):
    risk = df.groupby(col)["cardio"].mean().mul(100).reset_index()
    risk[col] = risk[col].map(labels)
    risk.columns = ["Catégorie", "Risque (%)"]

    fig = px.bar(
        risk, x="Catégorie", y="Risque (%)",
        color="Catégorie", title=title,
        text=risk["Risque (%)"].apply(lambda x: f"{x:.1f}%"),
        template="plotly_dark"
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, yaxis_range=[0, 100], **PLOTLY_LAYOUT)

    # Test Chi²
    table = pd.crosstab(df[col], df["cardio"])
    chi2, p, _, _ = chi2_contingency(table)

    return fig, chi2, p

categorical_vars = [
    ("gender_male", GENDER_LABELS, "Risque par sexe"),
    ("active", ACTIVE_LABELS, "Risque selon l'activité physique"),
    ("smoke", SMOKE_LABELS, "Risque selon le tabagisme"),
    ("alco", ALCO_LABELS, "Risque selon la consommation d'alcool"),
    ("cholesterol", CHOL_LABELS, "Risque selon le taux de cholestérol"),
    ("gluc", GLUC_LABELS, "Risque selon le taux de glucose"),
]

# 2 colonnes de graphiques
for i in range(0, len(categorical_vars), 2):
    cols = st.columns(2)
    for j, col_pair in enumerate(categorical_vars[i:i+2]):
        col_name, labels, title = col_pair
        fig, chi2, p = plot_categorical_risk(dff, col_name, labels, title)
        with cols[j]:
            st.plotly_chart(fig, use_container_width=True)
            sig = "✅ Association significative" if p < 0.05 else "❌ Non significative"
            st.caption(f"Chi² = {chi2:.2f} | p-value = {p:.2e} | {sig}")

st.divider()

# ── 5. Risque par groupe d'âge ────────────────────────────────────────────────

st.subheader("Risque cardiovasculaire par groupe d'âge")

age_df2 = dff.copy()
age_df2["Groupe d'âge"] = pd.cut(
    age_df2["age"],
    bins=[29, 39, 49, 59, 70],
    labels=["30-39", "40-49", "50-59", "60-69"]
)

age_risk = age_df2.groupby("Groupe d'âge", observed=True)["cardio"].mean().mul(100).reset_index()
age_risk.columns = ["Groupe d'âge", "Risque (%)"]

fig_age_risk = px.bar(
    age_risk, x="Groupe d'âge", y="Risque (%)",
    color="Risque (%)",
    color_continuous_scale=["#1e3a5f", "#ef4444"],
    text=age_risk["Risque (%)"].apply(lambda x: f"{x:.1f}%"),
    title="Prévalence cardiovasculaire par groupe d'âge",
    template="plotly_dark"
)
fig_age_risk.update_traces(textposition="outside")
fig_age_risk.update_layout(yaxis_range=[0, 100], coloraxis_showscale=False, **PLOTLY_LAYOUT)

table_age = pd.crosstab(age_df2["Groupe d'âge"], age_df2["cardio"])
chi2_age, p_age, _, _ = chi2_contingency(table_age)

st.plotly_chart(fig_age_risk, use_container_width=True)
st.caption(
    f"Chi² = {chi2_age:.2f} | p-value = {p_age:.2e} | "
    "✅ Association significative" if p_age < 0.05 else "❌ Non significative"
)

st.divider()

# ── 6. Variables numériques ───────────────────────────────────────────────────

st.subheader("Variables numériques selon le statut cardiovasculaire")

numeric_vars = [
    ("age", "Âge (années)"),
    ("BMI", "IMC (kg/m²)"),
    ("ap_hi", "Pression systolique (mmHg)"),
    ("ap_lo", "Pression diastolique (mmHg)"),
]

for i in range(0, len(numeric_vars), 2):
    cols = st.columns(2)
    for j, (col_name, label) in enumerate(numeric_vars[i:i+2]):
        with cols[j]:
            box_df = dff.copy()
            box_df["Statut"] = box_df["cardio"].map(CARDIO_LABELS)
            fig = px.box(
                box_df, x="Statut", y=col_name,
                color="Statut",
                color_discrete_map={"Sain": "#00d4ff", "Malade": "#ef4444"},
                labels={col_name: label},
                title=f"{label} selon le statut",
                points=False,
                template="plotly_dark"
            )
            fig.update_layout(showlegend=False, **PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

            g0 = dff[dff["cardio"] == 0][col_name].dropna()
            g1 = dff[dff["cardio"] == 1][col_name].dropna()
            _, p_mw = mannwhitneyu(g0, g1, alternative="two-sided")
            mean0, mean1 = g0.mean(), g1.mean()
            sig = "✅ Significatif" if p_mw < 0.05 else "❌ Non significatif"
            st.caption(
                f"Sain : {mean0:.1f} | Malade : {mean1:.1f} | "
                f"p-value = {p_mw:.2e} | {sig}"
            )

st.divider()

# ── 7. Matrice de corrélation ─────────────────────────────────────────────────

st.subheader("Matrice de corrélation")

corr_cols = ["age", "BMI", "ap_hi", "ap_lo", "cholesterol", "gluc",
             "smoke", "alco", "active", "gender_male", "cardio"]
corr = dff[corr_cols].corr().round(2)

readable = {
    "age": "Âge", "BMI": "IMC", "ap_hi": "PA syst.", "ap_lo": "PA diast.",
    "cholesterol": "Cholestérol", "gluc": "Glucose", "smoke": "Tabac",
    "alco": "Alcool", "active": "Activité", "gender_male": "Homme", "cardio": "Cardio"
}
corr_renamed = corr.rename(index=readable, columns=readable)

fig_corr = px.imshow(
    corr_renamed,
    color_continuous_scale="RdBu_r",
    zmin=-1, zmax=1,
    text_auto=True,
    title="Corrélation entre les variables",
    template="plotly_dark"
)
fig_corr.update_layout(height=500, **PLOTLY_LAYOUT)
st.plotly_chart(fig_corr, use_container_width=True)
