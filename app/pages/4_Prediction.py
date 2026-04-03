"""
Objectif :
- Interface utilisateur pour prédire
"""

import streamlit as st
from src.predict import load_model, predict

# Charger modèle
model = load_model("models/best_model.pkl")

st.title("Prédiction du risque")

# Inputs utilisateur :
age = st.slider("Age", 20, 100)
chol = st.number_input("Cholestérol")
bp = st.number_input("Pression sanguine")

# Bouton
if st.button("Prédire"):
    input_data = {
        "age": age,
        "chol": chol,
        "bp": bp
    }

    result = predict(model, input_data)

    # Afficher résultat :
    # Risque faible / élevé