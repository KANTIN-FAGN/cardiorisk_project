"""
Nettoyage des 3 datasets mondiaux OMS / Our World in Data.
Produit des fichiers propres dans data/processed/.

Usage : uv run python src/clean_world_data.py
"""

import pandas as pd
from pathlib import Path

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)

ISO3 = r"^[A-Z]{3}$"  # Regex pour un code pays ISO 3166-1 alpha-3 valide


def _keep_countries_only(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime les lignes dont le Code n'est pas un code ISO 3166-1 alpha-3."""
    before = len(df)
    df = df[df["Code"].notna() & df["Code"].str.match(ISO3, na=False)].copy()
    removed = before - len(df)
    print(f"  Agrégats régionaux supprimés : {removed}")
    return df


def _report_nulls(df: pd.DataFrame, label: str) -> None:
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    if nulls.empty:
        print(f"  [{label}] Aucune valeur nulle")
    else:
        print(f"  [{label}] Valeurs nulles :\n{nulls.to_string()}")


# ── 1. Taux de mortalité standardisé par âge ──────────────────────────────────

print("\n=== 1. death-rate standardized ===")

std = pd.read_csv(RAW / "death-rate-from-cardiovascular-disease-age-standardized-ghe.csv")
std.columns = ["Entity", "Code", "Year", "DeathRate"]

_report_nulls(std, "avant")
std = _keep_countries_only(std)
std = std.dropna(subset=["DeathRate"])
std = std.drop_duplicates()
_report_nulls(std, "après")

print(f"  Lignes finales : {len(std)} | Pays : {std['Entity'].nunique()} | Années : {std['Year'].min()}–{std['Year'].max()}")
std.to_csv(PROCESSED / "world_death_rate_standardized.csv", index=False)
print("  -> Sauvegardé : data/processed/world_death_rate_standardized.csv")


# ── 2. Mortalité vs PIB par habitant ──────────────────────────────────────────

print("\n=== 2. death-rate vs GDP ===")

gdp = pd.read_csv(RAW / "cardiovascular-death-rate-vs-gdp-per-capita.csv")
gdp.columns = ["Entity", "Code", "Year", "DeathRate", "GDP", "Population", "IncomeGroup"]

_report_nulls(gdp, "avant")
gdp = _keep_countries_only(gdp)

# Supprimer les lignes sans DeathRate (variable principale)
before = len(gdp)
gdp = gdp.dropna(subset=["DeathRate"])
print(f"  Lignes sans DeathRate supprimées : {before - len(gdp)}")

# GDP et Population : garder les lignes même si manquants (utile pour scatter partiel)
# IncomeGroup : remplir par le mode du pays quand possible
gdp["IncomeGroup"] = (
    gdp.groupby("Entity")["IncomeGroup"]
    .transform(lambda s: s.fillna(s.mode().iloc[0]) if not s.mode().empty else s)
)

gdp = gdp.drop_duplicates()
_report_nulls(gdp, "après")

print(f"  Lignes finales : {len(gdp)} | Pays : {gdp['Entity'].nunique()} | Années : {gdp['Year'].min()}–{gdp['Year'].max()}")
gdp.to_csv(PROCESSED / "world_death_rate_gdp.csv", index=False)
print("  -> Sauvegardé : data/processed/world_death_rate_gdp.csv")


# ── 3. Évolution Hommes vs Femmes ─────────────────────────────────────────────

print("\n=== 3. death-rate males vs females ===")

gender = pd.read_csv(RAW / "cardiovascular-disease-death-rate-over-time-males-vs-females.csv")
gender.columns = ["Entity", "Code", "Year", "Women", "Men"]

_report_nulls(gender, "avant")
gender = _keep_countries_only(gender)
gender = gender.dropna(subset=["Women", "Men"])
gender = gender.drop_duplicates()
_report_nulls(gender, "après")

print(f"  Lignes finales : {len(gender)} | Pays : {gender['Entity'].nunique()} | Années : {gender['Year'].min()}–{gender['Year'].max()}")
gender.to_csv(PROCESSED / "world_death_rate_gender.csv", index=False)
print("  -> Sauvegardé : data/processed/world_death_rate_gender.csv")

print("\n✓ Nettoyage terminé.")
