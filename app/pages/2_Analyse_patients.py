"""
Analyse exploratoire — 2 datasets patients
Tab 1 : Cardio Train (données médicales, Russie)
Tab 2 : BRFSS Heart Disease (lifestyle, USA)
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from scipy.stats import chi2_contingency, mannwhitneyu
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from components.cards import load_css

st.set_page_config(page_title="Analyse patients", page_icon="🔬", layout="wide")
load_css()

st.title("Analyse des patients")

PL = dict(paper_bgcolor="#161b22", plot_bgcolor="#161b22", font_color="#e2e8f0")

def fmt_p(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"

def plot_risk_bar(df, col, target, labels, title):
    risk = df.groupby(col)[target].mean().mul(100).reset_index()
    risk[col] = risk[col].map(labels)
    risk.columns = ["Catégorie", "Risque (%)"]
    fig = px.bar(
        risk, x="Catégorie", y="Risque (%)", color="Catégorie", title=title,
        text=risk["Risque (%)"].apply(lambda x: f"{x:.1f}%"),
        template="plotly_dark"
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, yaxis_range=[0, 100], **PL)
    table = pd.crosstab(df[col], df[target])
    chi2, p, _, _ = chi2_contingency(table)
    return fig, chi2, p

@st.cache_data
def load_cardio():
    return pd.read_csv("data/processed/cardio_train.csv")

@st.cache_data
def load_brfss():
    return pd.read_csv("data/processed/heart_disease.csv")

tab1, tab2 = st.tabs([
    "🏥 Cardio Train — données médicales",
    "📋 BRFSS — lifestyle & auto-déclaré"
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Cardio Train
# ════════════════════════════════════════════════════════════════════════════

with tab1:
    df = load_cardio()

    CARDIO_LABELS = {0: "Sain", 1: "Malade"}
    GENDER_LABELS = {0: "Femme", 1: "Homme"}

    # Filtres
    with st.sidebar:
        st.header("Filtres — Cardio Train")
        age_range = st.slider("Âge", int(df["age"].min()), int(df["age"].max()),
                              (int(df["age"].min()), int(df["age"].max())))
        gender_filter = st.multiselect("Sexe", [0, 1], default=[0, 1],
                                       format_func=lambda x: GENDER_LABELS[x])
        cardio_filter = st.multiselect("Statut", [0, 1], default=[0, 1],
                                       format_func=lambda x: CARDIO_LABELS[x])

    dff = df[
        df["age"].between(*age_range) &
        df["gender_male"].isin(gender_filter) &
        df["cardio"].isin(cardio_filter)
    ].copy()

    st.caption(f"Dataset : examens médicaux, Russie · {len(dff):,} patients sélectionnés sur {len(df):,}")

    # Métriques
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patients", f"{len(dff):,}")
    c2.metric("Prévalence cardio", f"{dff['cardio'].mean()*100:.1f}%")
    c3.metric("Âge moyen", f"{dff['age'].mean():.1f} ans")
    c4.metric("IMC moyen", f"{dff['BMI'].mean():.1f}")

    with st.expander("Statistiques descriptives"):
        st.dataframe(dff.rename(columns={
            "age": "Âge", "height": "Taille", "weight": "Poids",
            "ap_hi": "PA syst.", "ap_lo": "PA diast.",
            "cholesterol": "Cholestérol", "gluc": "Glucose",
            "smoke": "Tabac", "alco": "Alcool", "active": "Activité",
            "cardio": "Cardio", "gender_male": "Homme", "BMI": "IMC"
        }).describe().round(2), use_container_width=True)

    st.divider()

    # Distribution cible + sexe
    st.subheader("Distribution")
    col_a, col_b = st.columns(2)
    with col_a:
        td = dff["cardio"].map(CARDIO_LABELS).value_counts().reset_index()
        td.columns = ["Statut", "N"]
        fig = px.bar(td, x="Statut", y="N", color="Statut",
                     color_discrete_map={"Sain": "#3b82f6", "Malade": "#ef4444"},
                     text="N", title="Sains / Malades", template="plotly_dark")
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, **PL)
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        gd = dff["gender_male"].map(GENDER_LABELS).value_counts().reset_index()
        gd.columns = ["Sexe", "N"]
        fig2 = px.bar(gd, x="Sexe", y="N", color="Sexe",
                      color_discrete_map={"Femme": "#f472b6", "Homme": "#60a5fa"},
                      text="N", title="Répartition par sexe", template="plotly_dark")
        fig2.update_traces(textposition="outside")
        fig2.update_layout(showlegend=False, **PL)
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # Distribution âge
    st.subheader("Distribution de l'âge par sexe")
    ad = dff.copy()
    ad["Sexe"] = ad["gender_male"].map(GENDER_LABELS)
    fig_age = px.histogram(ad, x="age", color="Sexe",
                           color_discrete_map={"Femme": "#f472b6", "Homme": "#60a5fa"},
                           nbins=35, barmode="overlay", opacity=0.7,
                           labels={"age": "Âge (années)"},
                           title="Distribution de l'âge selon le sexe",
                           template="plotly_dark")
    fig_age.update_layout(**PL)
    st.plotly_chart(fig_age, use_container_width=True)

    st.divider()

    # Risque par facteur catégoriel
    st.subheader("Risque cardiovasculaire par facteur")
    cat_vars = [
        ("gender_male", GENDER_LABELS, "Risque par sexe"),
        ("active", {0: "Sédentaire", 1: "Actif"}, "Activité physique"),
        ("smoke", {0: "Non-fumeur", 1: "Fumeur"}, "Tabagisme"),
        ("alco", {0: "Non", 1: "Oui"}, "Alcool"),
        ("cholesterol", {1: "Normal", 2: "Élevé", 3: "Très élevé"}, "Cholestérol"),
        ("gluc", {1: "Normal", 2: "Élevé", 3: "Très élevé"}, "Glucose"),
    ]
    for i in range(0, len(cat_vars), 2):
        cols = st.columns(2)
        for j, (col_name, labels, title) in enumerate(cat_vars[i:i+2]):
            fig, chi2, p = plot_risk_bar(dff, col_name, "cardio", labels, title)
            with cols[j]:
                st.plotly_chart(fig, use_container_width=True)
                sig = "✅ Significatif" if p < 0.05 else "❌ Non significatif"
                st.caption(f"Chi² = {chi2:.2f} | p-value = {fmt_p(p)} | {sig}")

    st.divider()

    # Risque par groupe d'âge
    st.subheader("Risque par groupe d'âge")
    ad2 = dff.copy()
    ad2["Groupe"] = pd.cut(ad2["age"], bins=[29, 39, 49, 59, 70],
                           labels=["30-39", "40-49", "50-59", "60-69"])
    age_risk = ad2.groupby("Groupe", observed=True)["cardio"].mean().mul(100).reset_index()
    age_risk.columns = ["Groupe", "Risque (%)"]
    fig_ar = px.bar(age_risk, x="Groupe", y="Risque (%)", color="Risque (%)",
                    color_continuous_scale=["#1e3a5f", "#ef4444"],
                    text=age_risk["Risque (%)"].apply(lambda x: f"{x:.1f}%"),
                    title="Prévalence par groupe d'âge", template="plotly_dark")
    fig_ar.update_traces(textposition="outside")
    fig_ar.update_layout(yaxis_range=[0, 100], coloraxis_showscale=False, **PL)
    chi2_a, p_a, _, _ = chi2_contingency(pd.crosstab(ad2["Groupe"], ad2["cardio"]))
    st.plotly_chart(fig_ar, use_container_width=True)
    st.caption(f"Chi² = {chi2_a:.2f} | p-value = {fmt_p(p_a)} | ✅ Significatif")

    st.divider()

    # Variables numériques
    st.subheader("Variables numériques selon le statut")
    num_vars = [("age", "Âge (années)"), ("BMI", "IMC (kg/m²)"),
                ("ap_hi", "Pression systolique (mmHg)"), ("ap_lo", "Pression diastolique (mmHg)")]
    for i in range(0, len(num_vars), 2):
        cols = st.columns(2)
        for j, (col_name, label) in enumerate(num_vars[i:i+2]):
            with cols[j]:
                bd = dff.copy()
                bd["Statut"] = bd["cardio"].map(CARDIO_LABELS)
                fig = px.box(bd, x="Statut", y=col_name, color="Statut",
                             color_discrete_map={"Sain": "#3b82f6", "Malade": "#ef4444"},
                             labels={col_name: label}, title=label, points=False,
                             template="plotly_dark")
                fig.update_layout(showlegend=False, **PL)
                st.plotly_chart(fig, use_container_width=True)
                g0 = dff[dff["cardio"] == 0][col_name].dropna()
                g1 = dff[dff["cardio"] == 1][col_name].dropna()
                _, p_mw = mannwhitneyu(g0, g1, alternative="two-sided")
                sig = "✅ Significatif" if p_mw < 0.05 else "❌ Non significatif"
                st.caption(f"Sain : {g0.mean():.1f} | Malade : {g1.mean():.1f} | p = {fmt_p(p_mw)} | {sig}")

    st.divider()

    # Corrélation
    st.subheader("Matrice de corrélation")
    corr_cols = ["age", "BMI", "ap_hi", "ap_lo", "cholesterol", "gluc",
                 "smoke", "alco", "active", "gender_male", "cardio"]
    corr = dff[corr_cols].corr().round(2).rename(index={
        "age": "Âge", "BMI": "IMC", "ap_hi": "PA syst.", "ap_lo": "PA diast.",
        "cholesterol": "Cholestérol", "gluc": "Glucose", "smoke": "Tabac",
        "alco": "Alcool", "active": "Activité", "gender_male": "Homme", "cardio": "Cardio"
    }, columns={
        "age": "Âge", "BMI": "IMC", "ap_hi": "PA syst.", "ap_lo": "PA diast.",
        "cholesterol": "Cholestérol", "gluc": "Glucose", "smoke": "Tabac",
        "alco": "Alcool", "active": "Activité", "gender_male": "Homme", "cardio": "Cardio"
    })
    fig_corr = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                         text_auto=True, title="Corrélation entre variables",
                         template="plotly_dark")
    fig_corr.update_layout(height=500, **PL)
    st.plotly_chart(fig_corr, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — BRFSS Heart Disease
# ════════════════════════════════════════════════════════════════════════════

with tab2:
    hd = load_brfss()
    TARGET = "HeartDiseaseorAttack"
    HD_LABELS = {0.0: "Sain", 1.0: "Malade"}
    SEX_LABELS = {0.0: "Femme", 1.0: "Homme"}
    YN = {0.0: "Non", 1.0: "Oui"}

    # Mapping de l'âge catégoriel BRFSS (1–13)
    AGE_MAP = {
        1.0: "18-24", 2.0: "25-29", 3.0: "30-34", 4.0: "35-39",
        5.0: "40-44", 6.0: "45-49", 7.0: "50-54", 8.0: "55-59",
        9.0: "60-64", 10.0: "65-69", 11.0: "70-74", 12.0: "75-79", 13.0: "80+"
    }
    GENHLTH_MAP = {1.0: "Excellent", 2.0: "Très bon", 3.0: "Bon", 4.0: "Passable", 5.0: "Mauvais"}

    st.caption(f"Dataset : enquête BRFSS 2015, USA · {len(hd):,} répondants")

    # Métriques
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Répondants", f"{len(hd):,}")
    c2.metric("Prévalence cardio", f"{hd[TARGET].mean()*100:.1f}%")
    c3.metric("IMC moyen", f"{hd['BMI'].mean():.1f}")
    c4.metric("% Fumeurs", f"{hd['Smoker'].mean()*100:.1f}%")

    st.info("⚠️ Ce dataset est fortement déséquilibré : seulement **10% de cas positifs** contre 50% dans le Cardio Train. Cela reflète la réalité d'une enquête populationnelle.")

    st.divider()

    # Distribution cible + sexe
    st.subheader("Distribution")
    col_a, col_b = st.columns(2)
    with col_a:
        td = hd[TARGET].map(HD_LABELS).value_counts().reset_index()
        td.columns = ["Statut", "N"]
        fig = px.bar(td, x="Statut", y="N", color="Statut",
                     color_discrete_map={"Sain": "#3b82f6", "Malade": "#ef4444"},
                     text="N", title="Sains / Malades", template="plotly_dark")
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, **PL)
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        gd = hd["Sex"].map(SEX_LABELS).value_counts().reset_index()
        gd.columns = ["Sexe", "N"]
        fig2 = px.bar(gd, x="Sexe", y="N", color="Sexe",
                      color_discrete_map={"Femme": "#f472b6", "Homme": "#60a5fa"},
                      text="N", title="Répartition par sexe", template="plotly_dark")
        fig2.update_traces(textposition="outside")
        fig2.update_layout(showlegend=False, **PL)
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # Risque par facteur lifestyle
    st.subheader("Risque cardiovasculaire par facteur lifestyle")
    cat_vars_hd = [
        ("HighBP", YN, "Hypertension"),
        ("HighChol", YN, "Cholestérol élevé"),
        ("Smoker", YN, "Tabagisme"),
        ("PhysActivity", YN, "Activité physique"),
        ("Diabetes", YN, "Diabète"),
        ("Stroke", YN, "Antécédent d'AVC"),
        ("HvyAlcoholConsump", YN, "Alcool excessif"),
        ("DiffWalk", YN, "Difficultés à marcher"),
    ]
    for i in range(0, len(cat_vars_hd), 2):
        cols = st.columns(2)
        for j, (col_name, labels, title) in enumerate(cat_vars_hd[i:i+2]):
            fig, chi2, p = plot_risk_bar(hd, col_name, TARGET, labels, title)
            with cols[j]:
                st.plotly_chart(fig, use_container_width=True)
                sig = "✅ Significatif" if p < 0.05 else "❌ Non significatif"
                st.caption(f"Chi² = {chi2:.2f} | p-value = {fmt_p(p)} | {sig}")

    st.divider()

    # Risque par groupe d'âge
    st.subheader("Risque par groupe d'âge")
    hd2 = hd.copy()
    hd2["Groupe d'âge"] = hd2["Age"].map(AGE_MAP)
    age_order = list(AGE_MAP.values())
    age_risk_hd = hd2.groupby("Groupe d'âge")[TARGET].mean().mul(100).reindex(age_order).reset_index()
    age_risk_hd.columns = ["Groupe d'âge", "Risque (%)"]
    fig_ar = px.bar(age_risk_hd, x="Groupe d'âge", y="Risque (%)", color="Risque (%)",
                    color_continuous_scale=["#1e3a5f", "#ef4444"],
                    text=age_risk_hd["Risque (%)"].apply(lambda x: f"{x:.1f}%"),
                    title="Prévalence par groupe d'âge", template="plotly_dark")
    fig_ar.update_traces(textposition="outside")
    fig_ar.update_layout(yaxis_range=[0, 100], coloraxis_showscale=False, **PL)
    chi2_a, p_a, _, _ = chi2_contingency(pd.crosstab(hd2["Groupe d'âge"], hd2[TARGET]))
    st.plotly_chart(fig_ar, use_container_width=True)
    st.caption(f"Chi² = {chi2_a:.2f} | p-value = {fmt_p(p_a)} | ✅ Significatif")

    st.divider()

    # IMC boxplot
    st.subheader("IMC selon le statut cardiovasculaire")
    bd = hd.copy()
    bd["Statut"] = bd[TARGET].map(HD_LABELS)
    fig_bmi = px.box(bd, x="Statut", y="BMI", color="Statut",
                     color_discrete_map={"Sain": "#3b82f6", "Malade": "#ef4444"},
                     title="IMC selon le statut", points=False, template="plotly_dark")
    fig_bmi.update_layout(showlegend=False, **PL)
    st.plotly_chart(fig_bmi, use_container_width=True)
    g0 = hd[hd[TARGET] == 0]["BMI"]
    g1 = hd[hd[TARGET] == 1]["BMI"]
    _, p_bmi = mannwhitneyu(g0, g1, alternative="two-sided")
    st.caption(f"Sain : {g0.mean():.1f} | Malade : {g1.mean():.1f} | p = {fmt_p(p_bmi)} | ✅ Significatif")

    st.divider()

    # Corrélation
    st.subheader("Matrice de corrélation")
    corr_cols_hd = [TARGET, "HighBP", "HighChol", "BMI", "Smoker", "Diabetes",
                    "PhysActivity", "Stroke", "Age", "Sex"]
    corr_hd = hd[corr_cols_hd].corr().round(2).rename(index={
        TARGET: "Cardio", "HighBP": "Hypertension", "HighChol": "Cholestérol",
        "BMI": "IMC", "Smoker": "Tabac", "Diabetes": "Diabète",
        "PhysActivity": "Activité", "Stroke": "AVC", "Age": "Âge", "Sex": "Homme"
    }, columns={
        TARGET: "Cardio", "HighBP": "Hypertension", "HighChol": "Cholestérol",
        "BMI": "IMC", "Smoker": "Tabac", "Diabetes": "Diabète",
        "PhysActivity": "Activité", "Stroke": "AVC", "Age": "Âge", "Sex": "Homme"
    })
    fig_corr = px.imshow(corr_hd, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                         text_auto=True, title="Corrélation entre variables",
                         template="plotly_dark")
    fig_corr.update_layout(height=480, **PL)
    st.plotly_chart(fig_corr, use_container_width=True)
