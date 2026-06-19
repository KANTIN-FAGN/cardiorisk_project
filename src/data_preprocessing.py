import pandas as pd

def load_dataset(path, sep=None):
    """
    Charger un dataset CSV
    """
    if sep is not None:
        df = pd.read_csv(path, sep=sep)
    else:
        df = pd.read_csv(path, sep=None, engine="python")
    return df

def process_age(df, col):
    """
    Traiter l'âge en fonction du nombre de jours
    """
    df[col] = df[col].apply(lambda x: int(x / 365))
    return df

def process_gender(df, col):
    """
    Traiter le genre
    """
    df[col] = df[col].map({1: "male", 2: "female"})
    df = pd.get_dummies(df, columns=[col], drop_first=True, dtype=int)
    return df

def process_bmi(df):
    """
    Traiter le BMI
    """
    initial_rows = len(df)

    df["BMI"] = df["weight"] / (df["height"] / 100) ** 2
    df = df[(df["BMI"] >= 10) & (df["BMI"] <= 60)].copy()

    print(f"Nombre de lignes avant nettoyage : {initial_rows}")
    print(f"Nombre de lignes après nettoyage : {len(df)}")
    print(f"Nombre de lignes supprimées : {initial_rows - len(df)}")
    print(f"Pourcentage supprimé : {((initial_rows - len(df)) / initial_rows) * 100:.2f}%")

    return df

def process_ap(df):
    """
    Traiter la pression artériale
    """
    initial_rows = len(df)

    df = df[
        (df["ap_hi"] > 50) &
        (df["ap_lo"] > 30) &
        (df["ap_hi"] > df["ap_lo"]) &
        (df["ap_hi"] <= 250) &
        (df["ap_lo"] <= 200)
        ].copy()

    print(f"Nombre de lignes avant nettoyage : {initial_rows}")
    print(f"Nombre de lignes après nettoyage : {len(df)}")
    print(f"Nombre de lignes supprimées : {initial_rows - len(df)}")
    print(f"Pourcentage supprimé : {((initial_rows - len(df)) / initial_rows) * 100:.2f}%")

    return df

def delete_columns(df, columns_name):
    """
    Supprimer les colonnes inutiles
    """
    return df.drop(columns=[columns_name])

def oneHotEncoding(df):
    """
    Appliquer l'encodage one-hot aux variables catégorielles
    """
    return pd.get_dummies(df, drop_first=True)

def clean_dataset(df):
    """
    Nettoyage des données
    """
    # Supprimer les doublons
    df = df.drop_duplicates()

    # Gérer les valeurs manquantes
    for col in df.columns:
        if df[col].dtype in ["float64", "int64"]:
            # Remplacer par la médiane
            df[col] = df[col].fillna(df[col].median())
        else:
            # Remplacer par la valeur la plus fréquente
            df[col] = df[col].fillna(df[col].mode()[0])

    return df

def encode_dataset(df):
    """
    Encoder les variables catégorielles
    """
    # Exemple : one-hot encoding
    df = pd.get_dummies(df, drop_first=True)

    return df

def save_dataset(df, path):
    """
    Sauvegarder le dataset propre
    """
    df.to_csv(path, index=False)