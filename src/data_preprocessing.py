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

def process_age(age):
    """
    Traiter l'âge en fonction du nombre de jours
    """
    return int(age / 365)

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