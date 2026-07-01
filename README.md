# ❤️ CardioPredict - Analyse et Prédiction des Maladies Cardiovasculaires

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red)
![Plotly](https://img.shields.io/badge/Plotly-Visualization-purple)

---

# 📖 Présentation du projet

Les maladies cardiovasculaires représentent la première cause de mortalité dans le monde. Selon l'Organisation Mondiale de la Santé (OMS), elles sont responsables de plusieurs millions de décès chaque année et constituent un enjeu majeur de santé publique.

L'identification précoce des facteurs de risque permet de mettre en place des stratégies de prévention efficaces afin de réduire la mortalité et d'améliorer la qualité de vie des patients.

Ce projet vise à développer une application web interactive combinant :

* Data Science ;
* Machine Learning ;
* Visualisation de données ;
* Cartographie mondiale ;
* Analyse statistique.

L'objectif est de proposer un outil capable de prédire le risque cardiovasculaire individuel tout en offrant une vision globale de l'impact des maladies cardiovasculaires à travers le monde.

---

# ❓ Problématique

Les maladies cardiovasculaires sont influencées par de nombreux facteurs tels que :

* l'âge ;
* le sexe ;
* le mode de vie ;
* le diabète ;
* l'hypertension ;
* le cholestérol ;
* l'obésité ;
* l'environnement socio-économique.

Par ailleurs, la mortalité cardiovasculaire varie fortement selon les pays et les régions du monde.

Dans ce contexte, il est légitime de s'interroger sur l'impact de ces facteurs ainsi que sur la capacité des algorithmes de Machine Learning à prédire le risque cardiovasculaire.

### Problématique du projet

> Quels sont les principaux facteurs associés aux maladies cardiovasculaires à l'échelle individuelle et mondiale, et comment les modèles de Machine Learning peuvent-ils être utilisés pour prédire le risque cardiovasculaire ?

---

# 🎯 Objectifs

## Objectif 1 : Comprendre les facteurs de risque

Identifier les facteurs démographiques, comportementaux et cliniques associés aux maladies cardiovasculaires.

## Objectif 2 : Développer des modèles prédictifs

Créer plusieurs modèles de Machine Learning capables d'estimer le risque cardiovasculaire.

## Objectif 3 : Comparer différents types de données

Comparer :

* un modèle basé sur des habitudes de vie ;
* un modèle basé sur des données cliniques.

## Objectif 4 : Étudier les disparités géographiques

Analyser les différences de mortalité cardiovasculaire entre les pays.

## Objectif 5 : Développer une application interactive

Permettre aux utilisateurs de réaliser une estimation personnalisée de leur risque cardiovasculaire.

---

# 🔬 Hypothèses de recherche

### H1 : Facteurs comportementaux

Les habitudes de vie telles que le tabagisme, la consommation d'alcool, l'obésité et le manque d'activité physique augmentent le risque cardiovasculaire.

### H2 : Données cliniques

Les données cliniques permettent d'obtenir des prédictions plus précises que les seules données comportementales.

### H3 : Influence géographique

La mortalité cardiovasculaire varie significativement selon les pays et les régions du monde.

### H4 : Performance du Machine Learning

Les algorithmes de Machine Learning sont capables de fournir une estimation pertinente du risque cardiovasculaire.

---

# 📊 Jeux de données utilisés

## 1. Heart Disease Health Indicators (BRFSS 2015)

Source :

https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-dataset

### Description

Le Behavioral Risk Factor Surveillance System (BRFSS) est une enquête annuelle réalisée par le CDC aux États-Unis.

Cette enquête collecte des informations sur :

* les comportements à risque ;
* les maladies chroniques ;
* l'état de santé général ;
* les habitudes de vie.

### Caractéristiques

* 253 680 observations
* 22 variables
* Classification binaire

### Variable cible

```text
HeartDiseaseorAttack

0 = Pas de maladie cardiovasculaire
1 = Maladie cardiovasculaire ou infarctus
```

### Utilisation

Ce dataset est utilisé pour développer le modèle de prédiction rapide basé sur les habitudes de vie.

---

## 2. Cardiovascular Disease Dataset

Source :

https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset

### Description

Ce dataset contient des informations médicales réelles permettant d'étudier les facteurs cliniques associés aux maladies cardiovasculaires.

### Caractéristiques

* 70 000 observations
* 13 variables

### Variables principales

* Age
* Gender
* Height
* Weight
* Systolic Blood Pressure
* Diastolic Blood Pressure
* Cholesterol
* Glucose
* Smoking
* Alcohol Consumption
* Physical Activity

### Variable cible

```text
cardio

0 = Absence de maladie cardiovasculaire
1 = Présence de maladie cardiovasculaire
```

### Utilisation

Ce dataset est utilisé pour construire un modèle clinique plus précis.

---

## 3. Données mondiales de mortalité cardiovasculaire

### Fichiers utilisés

```text
death-rate-from-cardiovascular-disease-age-standardized-ghe.csv

cardiovascular-disease-death-rate-over-time-males-vs-females.csv

cardiovascular-death-rate-vs-gdp-per-capita.csv
```

### Utilisation

Ces données permettent :

* l'analyse mondiale ;
* la cartographie interactive ;
* l'étude des différences hommes/femmes ;
* l'analyse de l'impact du niveau de richesse des pays.

---

# 🏗️ Architecture du projet

```text
cardiorisk_project/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── notebooks/
│   ├── cardio_train/       (01_data_cleaning, 02_eda, 03_modeling, 04_interpretation)
│   └── heart_disease/      (idem)
│
├── models/
│   ├── cardio_train/       (best_model.pkl, scaler.pkl, feature_names.pkl, metrics.json)
│   └── heart_disease/      (idem)
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── predict.py
│   └── utils.py
│
├── app/
│   ├── Home.py
│   ├── pages/               (Analyse mondiale, Analyse patients, Modélisation, Prédiction, À propos)
│   ├── components/          (cards.py, theme.py)
│   └── assets/              (styles.css)
│
├── reports/
│
├── .streamlit/config.toml
├── pyproject.toml
└── README.md
```

---

# 🌐 Fonctionnalités de l'application

## 🟢 Analyse rapide

Prédiction basée sur :

* âge
* sexe
* IMC
* tabagisme
* alcool
* activité physique
* diabète
* hypertension
* cholestérol

Objectif :

Permettre une évaluation rapide du risque cardiovasculaire.

---

## 🔵 Analyse clinique

Prédiction basée sur :

* pression artérielle
* cholestérol
* glucose
* âge
* poids
* taille
* activité physique

Objectif :

Obtenir une estimation plus précise du risque cardiovasculaire.

---

## 🌍 Carte mondiale

Visualisation interactive :

* taux de mortalité cardiovasculaire ;
* comparaison des pays ;
* analyse géographique.

---

## 📈 Statistiques mondiales

Analyse :

* évolution temporelle ;
* comparaison hommes/femmes ;
* mortalité selon le PIB ;
* facteurs de risque.

---

# 🤖 Modèles de Machine Learning

Les modèles testés :

* Logistic Regression
* Random Forest
* XGBoost
* LightGBM

---

# 📏 Métriques d'évaluation

Les performances sont évaluées à l'aide de :

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC Score
* Confusion Matrix

Le **ROC-AUC** est utilisé comme critère de sélection du meilleur modèle : contrairement à l'accuracy,
il reste fiable même sur un dataset déséquilibré (le cas du BRFSS, ~10% de cas positifs).

---

# 📈 Résultats obtenus

| Dataset | Modèles comparés | Meilleur modèle | ROC-AUC |
|---|---|---|---|
| Cardio Train (clinique) | Logistic Regression, Random Forest, XGBoost, LightGBM | **Random Forest** | 0.807 |
| BRFSS (lifestyle) | Logistic Regression, Random Forest, XGBoost, LightGBM | **LightGBM** | 0.836 |

Détail des 4 modèles, courbes ROC, matrices de confusion et feature importance : page *Modélisation*
de l'application, ou `models/<dataset>/metrics.json` après entraînement.

---

# 📊 Visualisations

Le projet inclut :

* Histogrammes
* Heatmaps
* Matrices de corrélation
* Courbes ROC
* Feature Importance
* Cartes interactives
* Graphiques temporels

---

# 🛠️ Technologies utilisées

## Data Science

* Python
* Pandas
* NumPy

## Machine Learning

* Scikit-Learn
* XGBoost
* LightGBM

## Visualisation

* Matplotlib
* Seaborn
* Plotly

## Application Web

* Streamlit

## Cartographie

* Plotly Maps
* GeoJSON

---

# 🚀 Installation

Le projet est géré avec [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/votre-compte/cardiorisk-project.git

cd cardiorisk-project

uv sync
```

Cette commande installe automatiquement toutes les dépendances listées dans `pyproject.toml`
(pandas, scikit-learn, xgboost, lightgbm, streamlit, plotly...) dans un environnement virtuel `.venv/`.

> LightGBM nécessite `libomp` sur macOS : `brew install libomp` si l'entraînement échoue avec une erreur `libomp.dylib`.

---

# ▶️ Lancer l'application

```bash
uv run streamlit run app/Home.py
```

---

# 🤖 Entraîner les modèles

Les modèles ne sont pas versionnés dans le dépôt (fichiers binaires). Pour les (ré)entraîner :

```bash
uv run python -m src.train_model
```

Ce script entraîne 4 modèles (Logistic Regression, Random Forest, XGBoost, LightGBM) sur chacun des
deux datasets patients, sélectionne le meilleur par ROC-AUC, et sauvegarde le modèle, le scaler,
les features et les métriques dans `models/<dataset>/`. La page *Prédiction* de l'application en a besoin
pour fonctionner.

---

# 📋 Résultats attendus

Le projet doit permettre :

* l'identification des facteurs de risque majeurs ;
* la comparaison entre modèle comportemental et modèle clinique ;
* la prédiction du risque cardiovasculaire ;
* la visualisation des tendances mondiales ;
* l'exploration interactive des données.

---

# ⚠️ Limites

* Les modèles reposent sur des données historiques.
* Les prédictions ne remplacent pas un diagnostic médical.
* Certains facteurs médicaux ne sont pas présents dans les datasets utilisés.

---

# 🔒 Avertissement médical

Cette application a une vocation pédagogique et de sensibilisation.

Les résultats fournis ne constituent pas un diagnostic médical et ne doivent pas remplacer l'avis d'un professionnel de santé.

---

# 👨‍💻 Auteur

Projet de Data Science appliqué à la santé cardiovasculaire.

Année universitaire : 2025 - 2026
