"""
Design tokens partagés entre le CSS et les graphiques Plotly,
pour garantir une identité visuelle cohérente sur tout le dashboard.
"""

COLORS = {
    "bg": "#12151c",
    "surface": "#161a23",
    "grid": "#242a35",
    "text": "#c7cdd8",
    "text_strong": "#f2f4f8",
    "text_muted": "#8890a0",
    "accent": "#e8637a",   # corail — "malade" / alerte
    "accent2": "#f0946f",  # ambre chaud
    "good": "#4fd1ae",     # sarcelle — "sain"
    "warn": "#e8b04b",     # or
    "blue": "#6ea8e8",     # bleu froid — "homme"
    "muted": "#5b6472",
}

# Palette catégorielle par défaut pour tous les graphiques
COLORWAY = ["#e8637a", "#4fd1ae", "#e8b04b", "#6ea8e8", "#c9a6f5", "#f0946f", "#5b6472"]

# Échelle continue "corail" — remplace les "Reds"/"RdBu_r" génériques de Plotly
SEQUENTIAL_SCALE = ["#1c212b", "#5c3040", "#a3455a", "#e8637a", "#ffb199"]
DIVERGING_SCALE = ["#4fd1ae", "#1c212b", "#e8637a"]

# Mappings sémantiques réutilisés dans les pages patients
STATUS_COLORS = {"Sain": COLORS["good"], "Malade": COLORS["accent"]}
GENDER_COLORS = {"Homme": COLORS["blue"], "Femme": COLORS["accent2"]}

BASE_LAYOUT = dict(
    paper_bgcolor=COLORS["bg"],
    plot_bgcolor=COLORS["bg"],
    font=dict(family="Inter, -apple-system, sans-serif", color=COLORS["text"], size=12.5),
    title_font=dict(family="'Newsreader', Georgia, serif", size=17, color=COLORS["text_strong"]),
    colorway=COLORWAY,
    margin=dict(t=48, l=10, r=10, b=10),
    hoverlabel=dict(bgcolor=COLORS["surface"], font_family="Inter, sans-serif", bordercolor=COLORS["grid"]),
    xaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"], linecolor=COLORS["grid"]),
    yaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"], linecolor=COLORS["grid"]),
)


def style(fig, **overrides):
    """Applique le thème visuel commun à une figure Plotly Express."""
    fig.update_layout(**BASE_LAYOUT)
    if overrides:
        fig.update_layout(**overrides)
    return fig


def sig_html(p, threshold=0.05):
    """Badge HTML compact pour une p-value (remplace les '✅ Significatif' en texte brut)."""
    if p < threshold:
        return '<span class="badge badge-good">Significatif</span>'
    return '<span class="badge badge-muted">Non significatif</span>'


def fmt_p(p):
    return "&lt; 0.001" if p < 0.001 else f"{p:.3f}"
