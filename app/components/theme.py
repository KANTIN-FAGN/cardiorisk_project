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
