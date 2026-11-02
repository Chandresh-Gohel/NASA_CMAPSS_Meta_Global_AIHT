"""Chart styling shared by the app and the evaluation figures.

Conventions: one fixed colour per subset, decision colours always paired with an
icon and a label, ink-coloured text, one y axis per chart.
"""
from __future__ import annotations

SUBSET_ORDER = ["FD001", "FD002", "FD003", "FD004"]

CATEGORICAL = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
}
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
ACTION_STATUS = {
    "INSPECT": "critical",
    "ABSTAIN": "warning",
    "NO_INSPECTION": "good",
}
ACTION_ICONS = {
    "INSPECT": "🔧",
    "ABSTAIN": "⚠️",
    "NO_INSPECTION": "✅",
}
CHROME = {
    "light": {"surface": "#fcfcfb", "page": "#f9f9f7", "text": "#0b0b0b", "text2": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7"},
    "dark": {"surface": "#1a1a19", "page": "#0d0d0d", "text": "#f0efec", "text2": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835"},
}
DE_EMPHASIS = {"light": "#b5b3ab", "dark": "#5c5b56"}
DIVERGING = {"light": {"neg": "#e34948", "pos": "#2a78d6"}, "dark": {"neg": "#e66767", "pos": "#3987e5"}}


def subset_color(subset: str, mode: str = "light") -> str:
    idx = SUBSET_ORDER.index(subset) if subset in SUBSET_ORDER else len(SUBSET_ORDER)
    return CATEGORICAL[mode][idx % len(CATEGORICAL[mode])]


def action_color(action: str) -> str:
    return STATUS[ACTION_STATUS.get(action, "warning")]


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def style_mpl_axes(ax, mode: str = "light") -> None:
    """Hairline y-grid, recessive spines, ink-coloured text for matplotlib panels."""
    c = CHROME[mode]
    ax.set_facecolor(c["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(c["axis"])
        ax.spines[side].set_linewidth(1)
    ax.grid(True, axis="y", color=c["grid"], linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(colors=c["muted"], labelcolor=c["text2"], length=0)
    ax.title.set_color(c["text"])
    ax.xaxis.label.set_color(c["text2"])
    ax.yaxis.label.set_color(c["text2"])
