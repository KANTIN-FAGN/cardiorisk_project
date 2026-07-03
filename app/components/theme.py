"""Design tokens partagés entre le CSS et les graphiques Plotly."""

COLORS = {
    "bg": "#ffffff",
    "grid": "#e5e7eb",
    "text": "#374151",
    "text_strong": "#111827",
    "accent": "#e11d48",
    "good": "#059669",
    "neutral": "#475569",
}

COLORWAY = ["#e11d48", "#059669", "#475569", "#f59e0b", "#6366f1"]
SEQUENTIAL_SCALE = ["#fde8ec", "#e11d48"]

STATUS_COLORS = {"Sain": COLORS["good"], "Malade": COLORS["accent"]}
GENDER_COLORS = {"Homme": COLORS["neutral"], "Femme": COLORS["accent"]}

BASE_LAYOUT = dict(
    paper_bgcolor=COLORS["bg"],
    plot_bgcolor=COLORS["bg"],
    font=dict(family="-apple-system, Segoe UI, Roboto, sans-serif", color=COLORS["text"], size=12.5),
    title_font=dict(family="-apple-system, Segoe UI, Roboto, sans-serif", size=15, color=COLORS["text_strong"]),
    colorway=COLORWAY,
    margin=dict(t=44, l=10, r=10, b=10),
    xaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"], linecolor=COLORS["grid"]),
    yaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"], linecolor=COLORS["grid"]),
)


def style(fig, **overrides):
    """Applique le thème visuel commun à une figure Plotly Express."""
    fig.update_layout(**BASE_LAYOUT)
    if overrides:
        fig.update_layout(**overrides)
    return fig


def fmt_p(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def sig_label(p, threshold=0.05):
    return "✅ Significatif" if p < threshold else "❌ Non significatif"


# Paliers de risque partagés (gauge + libellés)
RISK_LOW = "#059669"    # vert  — < 25%
RISK_MID = "#f59e0b"    # ambre — 25–50%
RISK_HIGH = "#e11d48"   # rouge — >= 50%


def risk_level(proba):
    """(libellé, couleur) selon la probabilité de risque."""
    if proba >= 0.5:
        return "Risque élevé", RISK_HIGH
    if proba >= 0.25:
        return "Risque modéré", RISK_MID
    return "Risque faible", RISK_LOW


def risk_gauge(proba, height=260):
    """Compteur (jauge) de risque type mockup CardioPred, façon speedomètre coloré."""
    import plotly.graph_objects as go

    pct = proba * 100
    label, color = risk_level(proba)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 40, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": COLORS["grid"],
                     "tickvals": [0, 25, 50, 75, 100]},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": COLORS["bg"],
            "borderwidth": 0,
            "steps": [
                {"range": [0, 25], "color": "#ecfdf5"},
                {"range": [25, 50], "color": "#fffbeb"},
                {"range": [50, 100], "color": "#fef2f2"},
            ],
            "threshold": {"line": {"color": color, "width": 4}, "thickness": 0.8, "value": pct},
        },
    ))
    fig.update_layout(
        height=height,
        margin=dict(t=10, b=0, l=20, r=20),
        paper_bgcolor=COLORS["bg"],
        font=dict(family="-apple-system, Segoe UI, Roboto, sans-serif"),
    )
    return fig, label, color
