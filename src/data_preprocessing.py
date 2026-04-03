import pandas as pd

def load_dataset(path):
    """
    Charger un dataset CSV
    """
    df = pd.read_csv(path)
    return df

def clean_dataset(df):
    """
    Nettoyage des données
    """
    # Supprimer les doublons
    df = df.drop_duplicates()

    # Gérer les valeurs manquantes
    # → stratégie simple mais propre
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