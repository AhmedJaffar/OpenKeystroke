"""Dashboard design system: tokens, shared components and Plotly styling (single source for all workspaces)."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
import streamlit as st

# Okabe-Ito colour-blind-safe categorical palette
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#7F7F7F"]
STATUS = {  # status -> (glyph, hex): glyph + label so state never relies on colour alone
    "COMPLETED": ("●", "#1a7f37"),
    "RUNNING": ("◐", "#0969da"),
    "FAILED": ("✕", "#cf222e"),
    "CANCELLED": ("■", "#6e7781"),
    "CREATED": ("○", "#9a6700"),
    "VALIDATED": ("◇", "#9a6700"),
}
SPACE = {"xs": "0.25rem", "sm": "0.5rem", "md": "1rem", "lg": "1.5rem"}

_CSS = """
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1280px;}
.ok-title {font-size: 1.55rem; font-weight: 650; margin: 0 0 .15rem 0; letter-spacing: -.01em;}
.ok-sub {opacity: .75; margin: 0 0 1rem 0; font-size: .95rem;}
.ok-badge {display:inline-block; padding: .05rem .55rem; border-radius: 999px; font-size: .78rem; font-weight: 600; border: 1px solid currentColor;}
.ok-synth {border-left: 4px solid #9a6700; padding: .5rem .8rem; background: rgba(154,103,0,.10); border-radius: 4px; font-size: .9rem; margin: .25rem 0 .75rem 0;}
.ok-empty {border: 1px dashed rgba(128,128,128,.55); border-radius: 8px; padding: 1.25rem; text-align: center; margin: .5rem 0 1rem 0;}
.ok-empty b {display:block; margin-bottom: .25rem;}
[data-testid="stMetricValue"] {font-size: 1.5rem;}
button:focus-visible, [role="radio"]:focus-visible {outline: 3px solid #0969da !important; outline-offset: 2px;}
</style>
"""


def setup(title: str = "OpenKeystroke") -> None:
    st.set_page_config(page_title=title, page_icon="🎙", layout="wide", initial_sidebar_state="expanded")
    st.markdown(_CSS, unsafe_allow_html=True)


def header(title: str, subtitle: str = "") -> None:
    st.markdown(f'<h1 class="ok-title">{title}</h1><p class="ok-sub">{subtitle}</p>', unsafe_allow_html=True)


def badge(status: str) -> str:
    glyph, color = STATUS.get(status, ("•", "#6e7781"))
    return f'<span class="ok-badge" style="color:{color}">{glyph} {status}</span>'


def synthetic_notice(is_synthetic: bool = True) -> None:
    if is_synthetic:
        st.markdown(
            '<div class="ok-synth"><b>Synthetic data.</b> Results describe controlled simulated transients and are not evidence of real-world performance.</div>',
            unsafe_allow_html=True,
        )


def empty_state(title: str, hint: str) -> None:
    st.markdown(f'<div class="ok-empty"><b>{title}</b>{hint}</div>', unsafe_allow_html=True)


def style_fig(fig: go.Figure, height: int = 340, legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=56, b=10),
        colorway=PALETTE,
        showlegend=legend,
        template="plotly",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
    )
    fig.update_xaxes(gridcolor="rgba(128,128,128,.25)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,.25)", zeroline=False)
    return fig


def show(fig: go.Figure, key: str, **kw: Any) -> None:
    st.plotly_chart(style_fig(fig, **kw), width="stretch", key=key)
