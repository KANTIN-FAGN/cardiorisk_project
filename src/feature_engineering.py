"""
Objectif :
- Créer de nouvelles variables utiles
- Améliorer les performances du modèle
"""


def bp_category(hi, lo):
    """
    Catégorise la pression artérielle selon les seuils de l'American Heart Association.
    0=Normale, 1=Élevée, 2=Hypertension stade 1, 3=Hypertension stade 2, 4=Crise hypertensive
    """
    if hi >= 180 or lo >= 120:
        return 4
    if hi >= 140 or lo >= 90:
        return 3
    if hi >= 130 or lo >= 80:
        return 2
    if hi >= 120 and lo < 80:
        return 1
    return 0


def bmi_category(bmi):
    """
    Catégorise l'IMC selon les seuils de l'OMS.
    0=Insuffisance pondérale, 1=Normal, 2=Surpoids, 3=Obésité
    """
    if bmi < 18.5:
        return 0
    if bmi < 25:
        return 1
    if bmi < 30:
        return 2
    return 3


def add_bp_category(df, hi_col="ap_hi", lo_col="ap_lo"):
    df = df.copy()
    df["bp_category"] = [bp_category(h, l) for h, l in zip(df[hi_col], df[lo_col])]
    return df


def add_bmi_category(df, bmi_col="BMI"):
    df = df.copy()
    df["bmi_category"] = df[bmi_col].apply(bmi_category)
    return df


def add_unhealthy_days(df, mental_col="MentHlth", physical_col="PhysHlth"):
    """Combine les jours de santé mentale et physique difficile (0-60) en un seul indicateur."""
    df = df.copy()
    df["unhealthy_days"] = df[mental_col] + df[physical_col]
    return df
