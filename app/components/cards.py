import streamlit as st
from pathlib import Path


def load_css():
    css_path = Path(__file__).parent.parent / "assets" / "styles.css"
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def hero(kicker: str, title: str, subtitle: str, chips: list[str] | None = None):
    """Bandeau d'en-tête premium pour la page d'accueil."""
    chips_html = ""
    if chips:
        items = "".join(f'<span class="hero-chip">{c}</span>' for c in chips)
        chips_html = f'<div class="hero-chips">{items}</div>'

    st.markdown(
        f"""
        <div class="hero">
            <span class="kicker">{kicker}</span>
            <h1 class="hero-title">{title}</h1>
            <p class="hero-subtitle">{subtitle}</p>
            {chips_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(kicker: str, title: str, subtitle: str | None = None):
    """En-tête de section avec label d'accent, remplace st.subheader."""
    sub_html = f'<p class="section-subtitle">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f"""
        <div class="section-header">
            <span class="kicker">{kicker}</span>
            <h2 class="section-title">{title}</h2>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def stat_caption(*parts: str):
    """Ligne de légende sous un graphique, à base de badges HTML (voir theme.sig_html)."""
    st.markdown(f'<div class="stat-line">{" &nbsp;·&nbsp; ".join(parts)}</div>', unsafe_allow_html=True)


def coming_soon(title: str, description: str, tasks: list[str]):
    """État vide soigné pour les pages encore en construction (modèle non entraîné, etc.)."""
    items = "".join(f"<li>{t}</li>" for t in tasks)
    st.markdown(
        f"""
        <div class="empty-state">
            <span class="kicker">En construction</span>
            <h2 class="section-title">{title}</h2>
            <p class="section-subtitle">{description}</p>
            <ul class="empty-state-list">{items}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
