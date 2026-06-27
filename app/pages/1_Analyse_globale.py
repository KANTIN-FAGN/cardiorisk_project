"""
Analyse mondiale des maladies cardiovasculaires
3 datasets OMS : carte choroplèthe, PIB vs mortalité, évolution H/F
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from components.cards import load_css

st.set_page_config(page_title="Analyse mondiale", page_icon="🌍", layout="wide")
load_css()

LAYOUT = dict(paper_bgcolor="#161b22", plot_bgcolor="#161b22", font_color="#e2e8f0")

st.title("Analyse mondiale")
st.caption("Mortalité cardiovasculaire dans le monde — données OMS / Our World in Data")

st.divider()

# ── Chargement ────────────────────────────────────────────────────────────────

@st.cache_data
def load_world_data():
    gdp = pd.read_csv("data/processed/world_death_rate_gdp.csv")
    gender = pd.read_csv("data/processed/world_death_rate_gender.csv")
    standardized = pd.read_csv("data/processed/world_death_rate_standardized.csv")
    return gdp, gender, standardized

gdp_df, gender_df, std_df = load_world_data()

# ── 1. Carte choroplèthe ──────────────────────────────────────────────────────

st.subheader("Taux de mortalité cardiovasculaire par pays")

year_std = st.slider(
    "Année",
    int(std_df["Year"].min()),
    int(std_df["Year"].max()),
    int(std_df["Year"].max()),
    key="slider_map"
)

map_df = std_df[std_df["Year"] == year_std].dropna(subset=["Code", "DeathRate"])
# Exclure les agrégats régionaux (pas de code ISO à 3 lettres)
map_df = map_df[map_df["Code"].str.len() == 3]

fig_map = px.choropleth(
    map_df,
    locations="Code",
    color="DeathRate",
    hover_name="Entity",
    color_continuous_scale="Reds",
    range_color=(map_df["DeathRate"].quantile(0.05), map_df["DeathRate"].quantile(0.95)),
    labels={"DeathRate": "Décès / 100k"},
    title=f"Taux de mortalité cardiovasculaire standardisé par âge ({year_std})",
    template="plotly_dark"
)
fig_map.update_layout(
    height=500,
    geo=dict(bgcolor="#161b22", showframe=False, showcoastlines=True, coastlinecolor="#30363d"),
    coloraxis_colorbar=dict(title="Décès<br>/ 100k"),
    margin=dict(t=40, b=0, l=0, r=0),
    **LAYOUT
)
st.plotly_chart(fig_map, use_container_width=True)

st.divider()

# ── 2. Mortalité vs PIB ───────────────────────────────────────────────────────

st.subheader("Mortalité cardiovasculaire vs PIB par habitant")

year_gdp = st.slider(
    "Année",
    int(gdp_df["Year"].min()),
    int(gdp_df["Year"].max()),
    2019,
    key="slider_gdp"
)

income_labels = {
    "Low-income countries": "Pays à faible revenu",
    "Lower-middle-income countries": "Revenu intermédiaire inférieur",
    "Upper-middle-income countries": "Revenu intermédiaire supérieur",
    "High-income countries": "Pays à revenu élevé",
}
income_colors = {
    "Pays à faible revenu": "#ef4444",
    "Revenu intermédiaire inférieur": "#f97316",
    "Revenu intermédiaire supérieur": "#eab308",
    "Pays à revenu élevé": "#3b82f6",
}

gdp_year = (
    gdp_df[gdp_df["Year"] == year_gdp]
    .dropna(subset=["GDP", "DeathRate", "Population", "IncomeGroup"])
    .copy()
)
gdp_year = gdp_year[gdp_year["Code"].str.len() == 3]
gdp_year["IncomeGroup"] = gdp_year["IncomeGroup"].map(income_labels)

fig_gdp = px.scatter(
    gdp_year,
    x="GDP",
    y="DeathRate",
    color="IncomeGroup",
    size="Population",
    hover_name="Entity",
    size_max=50,
    log_x=True,
    color_discrete_map=income_colors,
    labels={
        "GDP": "PIB par habitant (USD, échelle log)",
        "DeathRate": "Décès cardiovasculaires / 100k",
        "IncomeGroup": "Groupe de revenus"
    },
    title=f"Mortalité cardiovasculaire selon le niveau de richesse ({year_gdp})",
    template="plotly_dark"
)
fig_gdp.update_layout(height=480, **LAYOUT)
st.plotly_chart(fig_gdp, use_container_width=True)

st.divider()

# ── 3. Évolution Hommes vs Femmes ─────────────────────────────────────────────

st.subheader("Évolution du taux de mortalité : Hommes vs Femmes")

# Top pays les plus peuplés / connus pour le sélecteur
top_entities = (
    gender_df[gender_df["Year"] == 2020]
    .dropna(subset=["Men", "Women"])
    .query("Code.str.len() == 3", engine="python")
    .sort_values("Men", ascending=False)["Entity"]
    .head(40)
    .tolist()
)

selected = st.multiselect(
    "Pays",
    options=sorted(gender_df["Entity"].unique()),
    default=["France", "United States", "Russia", "Brazil", "Germany"]
)

if selected:
    gender_sel = gender_df[gender_df["Entity"].isin(selected)].dropna(subset=["Men", "Women"])
    gender_long = gender_sel.melt(
        id_vars=["Entity", "Year"],
        value_vars=["Women", "Men"],
        var_name="Sexe",
        value_name="DeathRate"
    )
    gender_long["Sexe"] = gender_long["Sexe"].map({"Women": "Femmes", "Men": "Hommes"})

    fig_gender = px.line(
        gender_long,
        x="Year", y="DeathRate",
        color="Entity",
        line_dash="Sexe",
        labels={
            "Year": "Année",
            "DeathRate": "Décès / 100k",
            "Entity": "Pays",
            "Sexe": "Sexe"
        },
        title="Évolution du taux de mortalité cardiovasculaire par pays et par sexe",
        template="plotly_dark"
    )
    fig_gender.update_layout(height=480, **LAYOUT)
    st.plotly_chart(fig_gender, use_container_width=True)
else:
    st.info("Sélectionne au moins un pays.")
