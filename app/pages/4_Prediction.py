"""
Objectif :
- Interface utilisateur pour prédire le risque cardiovasculaire individuel
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))
from components.cards import load_css
from src.feature_engineering import add_bmi_category, add_bp_category, add_unhealthy_days, bmi_category, bp_category
from src.predict import load_bundle, pick_example_profiles, predict_risk

st.set_page_config(page_title="Prédiction", page_icon="🩺", layout="wide")
load_css()

st.title("Prédiction du risque")
st.caption("Estime le risque cardiovasculaire à partir d'un profil clinique ou d'habitudes de vie")

st.divider()

CARDIO_LABELS = {0: "Sain", 1: "Malade"}
HD_LABELS = {0.0: "Sain", 1.0: "Malade"}
AGE_MAP = {
    1.0: "18-24", 2.0: "25-29", 3.0: "30-34", 4.0: "35-39",
    5.0: "40-44", 6.0: "45-49", 7.0: "50-54", 8.0: "55-59",
    9.0: "60-64", 10.0: "65-69", 11.0: "70-74", 12.0: "75-79", 13.0: "80+"
}
GENHLTH_MAP = {1.0: "Excellent", 2.0: "Très bon", 3.0: "Bon", 4.0: "Passable", 5.0: "Mauvais"}
EDUCATION_MAP = {
    1.0: "Jamais scolarisé / maternelle",
    2.0: "École élémentaire",
    3.0: "Lycée non terminé",
    4.0: "Diplôme du secondaire",
    5.0: "Études supérieures partielles",
    6.0: "Diplôme universitaire",
}
INCOME_MAP = {
    1.0: "< 10 000 $", 2.0: "10-15 000 $", 3.0: "15-20 000 $", 4.0: "20-25 000 $",
    5.0: "25-35 000 $", 6.0: "35-50 000 $", 7.0: "50-75 000 $", 8.0: "75 000 $ ou plus",
}
DIABETES_MAP = {0.0: "Non", 1.0: "Pré-diabète", 2.0: "Oui"}


@st.cache_data
def load_cardio_df():
    df = pd.read_csv("data/processed/cardio_train.csv")
    df = add_bp_category(df)
    df = add_bmi_category(df)
    return df


@st.cache_data
def load_brfss_df():
    df = pd.read_csv("data/processed/heart_disease.csv")
    df = add_bmi_category(df)
    df = add_unhealthy_days(df)
    return df


@st.cache_resource
def get_bundle(name):
    return load_bundle(name)


@st.cache_data
def get_examples(dataset_name, target_col):
    df = load_cardio_df() if dataset_name == "cardio_train" else load_brfss_df()
    bundle = get_bundle(dataset_name)
    return pick_example_profiles(df, target_col, bundle)


LEVEL_ICONS = {"Sain": "🟢", "Modéré": "🟡", "Malade": "🔴"}


def render_example_cards(examples, session_key, summary_fn):
    st.markdown("**Profils d'exemple** — sélectionnés par le modèle dans le dataset selon leur risque prédit")
    cols = st.columns(3)
    for col, (level, profile) in zip(cols, examples.items()):
        with col:
            with st.container(border=True):
                st.markdown(f"##### {LEVEL_ICONS[level]} {level}")
                st.caption(f"Risque prédit : {profile['risk'] * 100:.0f}%")
                for line in summary_fn(profile["data"]):
                    st.caption(line)
                if st.button("Utiliser ce profil", key=f"{session_key}_{level}", use_container_width=True):
                    st.session_state[session_key] = profile["data"]


def cardio_summary(row):
    sexe = "Homme" if row["gender_male"] == 1 else "Femme"
    return [
        f"{int(row['age'])} ans, {sexe}",
        f"Tension : {int(row['ap_hi'])}/{int(row['ap_lo'])} mmHg",
        f"IMC : {row['BMI']:.1f}",
    ]


def brfss_summary(row):
    sexe = "Homme" if row["Sex"] == 1 else "Femme"
    bp = "Hypertension" if row["HighBP"] == 1 else "Tension normale"
    return [
        f"{AGE_MAP.get(row['Age'], '?')}, {sexe}",
        bp,
        f"IMC : {row['BMI']:.0f}",
    ]


def show_result(proba, true_label=None):
    pred_label = "Malade" if proba >= 0.5 else "Sain"
    icon = "🔴" if proba >= 0.5 else "🟢"
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Risque estimé", f"{proba * 100:.0f}%")
    with col2:
        st.progress(min(max(proba, 0.0), 1.0))
        st.markdown(f"{icon} **Prédiction : {pred_label}**")
    if true_label is not None:
        st.caption(
            f"Profil réel (tiré du dataset) : **{true_label}** — utile pour tester le modèle, "
            "mais ce profil a pu faire partie des données d'entraînement."
        )


try:
    bundle_cardio = get_bundle("cardio_train")
    bundle_brfss = get_bundle("heart_disease")
except FileNotFoundError:
    st.info("🚧 Aucun modèle entraîné pour l'instant. Lance `python -m src.train_model` pour générer les modèles.")
    st.stop()

tab1, tab2 = st.tabs([
    "🏥 Modèle clinique — Cardio Train",
    "📋 Modèle lifestyle — BRFSS",
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Modèle clinique (Cardio Train)
# ════════════════════════════════════════════════════════════════════════════

with tab1:
    st.caption(f"Modèle utilisé : **{bundle_cardio['model_name']}** (ROC-AUC = {bundle_cardio['roc_auc']:.3f})")

    render_example_cards(get_examples("cardio_train", "cardio"), "cardio_profile", cardio_summary)
    st.divider()

    profile = st.session_state.get("cardio_profile", {})

    with st.form("form_cardio"):
        c1, c2, c3 = st.columns(3)
        age = c1.slider("Âge", 18, 90, int(profile.get("age", 50)))
        gender = c2.radio("Sexe", ["Femme", "Homme"], index=int(profile.get("gender_male", 0)), horizontal=True)
        active = c3.radio("Activité physique", ["Sédentaire", "Actif"], index=int(profile.get("active", 1)), horizontal=True)

        c4, c5 = st.columns(2)
        height = c4.number_input("Taille (cm)", 100, 250, int(profile.get("height", 165)))
        weight = c5.number_input("Poids (kg)", 28, 200, int(profile.get("weight", 70)))
        bmi = weight / (height / 100) ** 2
        st.caption(f"IMC calculé : **{bmi:.1f}**")

        c6, c7 = st.columns(2)
        ap_hi = c6.slider("Pression systolique (mmHg)", 60, 240, int(profile.get("ap_hi", 120)))
        ap_lo = c7.slider("Pression diastolique (mmHg)", 40, 200, int(profile.get("ap_lo", 80)))

        c8, c9, c10 = st.columns(3)
        cholesterol = c8.selectbox(
            "Cholestérol", [1, 2, 3], index=int(profile.get("cholesterol", 1)) - 1,
            format_func=lambda x: {1: "Normal", 2: "Élevé", 3: "Très élevé"}[x]
        )
        gluc = c9.selectbox(
            "Glucose", [1, 2, 3], index=int(profile.get("gluc", 1)) - 1,
            format_func=lambda x: {1: "Normal", 2: "Élevé", 3: "Très élevé"}[x]
        )
        smoke = c10.radio("Tabac", ["Non", "Oui"], index=int(profile.get("smoke", 0)), horizontal=True)

        alco = st.radio("Alcool", ["Non", "Oui"], index=int(profile.get("alco", 0)), horizontal=True)

        submitted_cardio = st.form_submit_button("Prédire le risque")

    if submitted_cardio:
        input_row = {
            "age": age, "height": height, "weight": weight,
            "ap_hi": ap_hi, "ap_lo": ap_lo,
            "cholesterol": cholesterol, "gluc": gluc,
            "smoke": 1 if smoke == "Oui" else 0,
            "alco": 1 if alco == "Oui" else 0,
            "active": 1 if active == "Actif" else 0,
            "gender_male": 1 if gender == "Homme" else 0,
            "BMI": bmi,
            "bp_category": bp_category(ap_hi, ap_lo),
            "bmi_category": bmi_category(bmi),
        }
        proba, _ = predict_risk(bundle_cardio, input_row)
        true_label = CARDIO_LABELS.get(int(profile["cardio"])) if "cardio" in profile else None
        st.divider()
        show_result(proba, true_label)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Modèle lifestyle (BRFSS)
# ════════════════════════════════════════════════════════════════════════════

with tab2:
    st.caption(f"Modèle utilisé : **{bundle_brfss['model_name']}** (ROC-AUC = {bundle_brfss['roc_auc']:.3f})")

    render_example_cards(get_examples("heart_disease", "HeartDiseaseorAttack"), "brfss_profile", brfss_summary)
    st.divider()

    p = st.session_state.get("brfss_profile", {})

    with st.form("form_brfss"):
        st.markdown("**Profil**")
        c1, c2, c3, c4 = st.columns(4)
        sex = c1.radio("Sexe", ["Femme", "Homme"], index=int(p.get("Sex", 0)), horizontal=True)
        age_cat = c2.selectbox(
            "Groupe d'âge", list(AGE_MAP.keys()),
            index=list(AGE_MAP.keys()).index(p.get("Age", 7.0)) if p.get("Age") in AGE_MAP else 6,
            format_func=lambda x: AGE_MAP[x]
        )
        education = c3.selectbox(
            "Éducation", list(EDUCATION_MAP.keys()),
            index=list(EDUCATION_MAP.keys()).index(p.get("Education", 4.0)) if p.get("Education") in EDUCATION_MAP else 3,
            format_func=lambda x: EDUCATION_MAP[x]
        )
        income = c4.selectbox(
            "Revenu", list(INCOME_MAP.keys()),
            index=list(INCOME_MAP.keys()).index(p.get("Income", 5.0)) if p.get("Income") in INCOME_MAP else 4,
            format_func=lambda x: INCOME_MAP[x]
        )

        st.markdown("**Indicateurs cliniques**")
        c5, c6, c7, c8 = st.columns(4)
        high_bp = c5.radio("Hypertension", ["Non", "Oui"], index=int(p.get("HighBP", 0)), horizontal=True)
        high_chol = c6.radio("Cholestérol élevé", ["Non", "Oui"], index=int(p.get("HighChol", 0)), horizontal=True)
        chol_check = c7.radio("Bilan cholestérol < 5 ans", ["Non", "Oui"], index=int(p.get("CholCheck", 1)), horizontal=True)
        bmi_hd = c8.number_input("IMC", 12, 98, int(p.get("BMI", 27)))

        gen_hlth = st.select_slider(
            "Santé générale perçue", options=list(GENHLTH_MAP.keys()),
            value=p.get("GenHlth", 3.0) if p.get("GenHlth") in GENHLTH_MAP else 3.0,
            format_func=lambda x: GENHLTH_MAP[x]
        )

        st.markdown("**Habitudes de vie**")
        c9, c10, c11, c12 = st.columns(4)
        smoker = c9.radio("Tabac", ["Non", "Oui"], index=int(p.get("Smoker", 0)), horizontal=True)
        phys_activity = c10.radio("Activité physique", ["Non", "Oui"], index=int(p.get("PhysActivity", 1)), horizontal=True)
        fruits = c11.radio("Fruits quotidiens", ["Non", "Oui"], index=int(p.get("Fruits", 1)), horizontal=True)
        veggies = c12.radio("Légumes quotidiens", ["Non", "Oui"], index=int(p.get("Veggies", 1)), horizontal=True)
        hvy_alcohol = st.radio("Consommation excessive d'alcool", ["Non", "Oui"], index=int(p.get("HvyAlcoholConsump", 0)), horizontal=True)

        st.markdown("**Antécédents & accès aux soins**")
        c13, c14, c15 = st.columns(3)
        stroke = c13.radio("Antécédent d'AVC", ["Non", "Oui"], index=int(p.get("Stroke", 0)), horizontal=True)
        diabetes = c14.selectbox(
            "Diabète", list(DIABETES_MAP.keys()),
            index=list(DIABETES_MAP.keys()).index(p.get("Diabetes", 0.0)) if p.get("Diabetes") in DIABETES_MAP else 0,
            format_func=lambda x: DIABETES_MAP[x]
        )
        diff_walk = c15.radio("Difficulté à marcher", ["Non", "Oui"], index=int(p.get("DiffWalk", 0)), horizontal=True)

        c16, c17 = st.columns(2)
        any_healthcare = c16.radio("Couverture santé", ["Non", "Oui"], index=int(p.get("AnyHealthcare", 1)), horizontal=True)
        no_doc_cost = c17.radio("Soins renoncés (coût)", ["Non", "Oui"], index=int(p.get("NoDocbcCost", 0)), horizontal=True)

        c18, c19 = st.columns(2)
        ment_hlth = c18.slider("Jours de santé mentale difficile (30 derniers jours)", 0, 30, int(p.get("MentHlth", 0)))
        phys_hlth = c19.slider("Jours de santé physique difficile (30 derniers jours)", 0, 30, int(p.get("PhysHlth", 0)))

        submitted_brfss = st.form_submit_button("Prédire le risque")

    if submitted_brfss:
        yn = lambda v: 1.0 if v == "Oui" else 0.0
        input_row = {
            "HighBP": yn(high_bp), "HighChol": yn(high_chol), "CholCheck": yn(chol_check),
            "BMI": float(bmi_hd), "Smoker": yn(smoker), "Stroke": yn(stroke),
            "Diabetes": diabetes, "PhysActivity": yn(phys_activity), "Fruits": yn(fruits),
            "Veggies": yn(veggies), "HvyAlcoholConsump": yn(hvy_alcohol),
            "AnyHealthcare": yn(any_healthcare), "NoDocbcCost": yn(no_doc_cost),
            "GenHlth": gen_hlth, "MentHlth": float(ment_hlth), "PhysHlth": float(phys_hlth),
            "DiffWalk": yn(diff_walk), "Sex": 1.0 if sex == "Homme" else 0.0,
            "Age": age_cat, "Education": education, "Income": income,
            "bmi_category": bmi_category(bmi_hd),
            "unhealthy_days": float(ment_hlth) + float(phys_hlth),
        }
        proba, _ = predict_risk(bundle_brfss, input_row)
        true_label = HD_LABELS.get(p["HeartDiseaseorAttack"]) if "HeartDiseaseorAttack" in p else None
        st.divider()
        show_result(proba, true_label)
