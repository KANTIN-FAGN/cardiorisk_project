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
from components.theme import style, SEQUENTIAL_SCALE, COLORS

st.set_page_config(page_title="Analyse mondiale", page_icon="🌍", layout="wide")
load_css()

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

col_ctrl, _ = st.columns([1, 2])
with col_ctrl:
    year_std = st.slider(
        "Année",
        int(gdp_df["Year"].min()),
        int(gdp_df["Year"].max()),
        2019,
        key="slider_map"
    )

# Jointure std + population (depuis gdp_df) pour calculer les décès absolus
map_base = std_df[std_df["Year"] == year_std][["Code", "Entity", "DeathRate"]]
pop_year = gdp_df[gdp_df["Year"] == year_std][["Code", "Population"]].dropna(subset=["Population"])
map_df = map_base.merge(pop_year, on="Code", how="left")
map_df["TotalDeaths"] = (map_df["DeathRate"] * map_df["Population"] / 100_000).round(0)

col_map, col_bar = st.columns([1.6, 1])

with col_map:
    fig_map = px.choropleth(
        map_df.dropna(subset=["TotalDeaths"]),
        locations="Code",
        color="TotalDeaths",
        hover_name="Entity",
        hover_data={"DeathRate": ":.0f", "TotalDeaths": ":,.0f", "Code": False},
        color_continuous_scale=SEQUENTIAL_SCALE,
        range_color=(0, map_df["TotalDeaths"].quantile(0.92)),
        labels={"TotalDeaths": "Décès totaux", "DeathRate": "Pour 100k"},
        title=f"Décès cardiovasculaires totaux ({year_std})",
    )
    style(
        fig_map,
        height=420,
        geo=dict(bgcolor=COLORS["bg"], showframe=False, showcoastlines=True, coastlinecolor=COLORS["grid"]),
        coloraxis_colorbar=dict(title="Décès<br>totaux"),
        margin=dict(t=40, b=0, l=0, r=0),
    )
    st.plotly_chart(fig_map, use_container_width=True, config={"scrollZoom": False, "doubleClick": False, "displayModeBar": False})
    st.caption("Décès totaux = taux pour 100k × population. Les pays sans donnée de population apparaissent en gris.")

with col_bar:
    top_n = map_df.dropna(subset=["TotalDeaths"]).nlargest(15, "TotalDeaths").copy()
    top_n = top_n.sort_values("TotalDeaths")
    top_n["label"] = top_n["TotalDeaths"].apply(lambda x: f"{x/1e6:.2f}M" if x >= 1e6 else f"{x/1e3:.0f}k")
    fig_bar = px.bar(
        top_n, x="TotalDeaths", y="Entity",
        orientation="h",
        text="label",
        color="TotalDeaths",
        color_continuous_scale=SEQUENTIAL_SCALE,
        custom_data=["label"],
        labels={"TotalDeaths": "Décès totaux", "Entity": ""},
        title=f"Top 15 pays ({year_std})",
    )
    fig_bar.update_traces(
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>%{customdata[0]} décès<extra></extra>"
    )
    style(
        fig_bar,
        height=420,
        showlegend=False,
        coloraxis_showscale=False,
        margin=dict(t=40, b=0, l=0, r=60),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

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
    labels={
        "GDP": "PIB par habitant (USD, échelle log)",
        "DeathRate": "Décès cardiovasculaires / 100k",
        "IncomeGroup": "Groupe de revenus"
    },
    title=f"Mortalité cardiovasculaire selon le niveau de richesse ({year_gdp})",
)
style(fig_gdp, height=480)
st.plotly_chart(fig_gdp, use_container_width=True)

st.divider()

# ── 3. Évolution Hommes vs Femmes ─────────────────────────────────────────────

st.subheader("Évolution du taux de mortalité : Hommes vs Femmes")

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
    )
    style(fig_gender, height=480)
    st.plotly_chart(fig_gender, use_container_width=True)
else:
    st.info("Sélectionne au moins un pays.")
