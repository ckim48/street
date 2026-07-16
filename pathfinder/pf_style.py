"""Shared figure styling for PathFinder maps.

One palette + legend/marker helpers so every map (base, R1, R2, R3) reads as one
system: a recessive street-grid context, the redlined zone and highway barrier as
semantic fills, and point/line marks that sit ON the map (white halo) with an
always-present legend.  Colours are the validated data-viz categorical palette
(colour-blind-safe, fixed order) mapped to roles -- red is reserved for the
"redline" meaning, so the two highway-side anchors use the max-separation
blue/orange pair instead of red/blue.
"""
from __future__ import annotations

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ---- validated categorical palette (light surface), by role ----------------
BLUE = "#2a78d6"; AQUA = "#1baf7a"; YELLOW = "#eda100"; GREEN = "#008300"
VIOLET = "#4a3aa7"; RED = "#e34948"; MAGENTA = "#e87ba4"; ORANGE = "#eb6834"
INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"; GRID = "#e1e0d9"

C = dict(
    base="#c7c5bd",        # context street grid (recessive)
    omega="#8a8880",       # study-area outline
    holc_d=RED,            # redlined (HOLC-D) community
    barrier="#6f6d68",     # highway right-of-way band
    highway=INK,           # highway centreline
    side_a=BLUE,           # barrier anchors, side A
    side_b=ORANGE,         # barrier anchors, side B
    added=GREEN,           # restored / added streets
    removed=VIOLET,        # removed streets (R1)
    landmark=BLUE,         # sampled landmarks (R1)
    water=BLUE,            # water / no-build exclusion
    anchor=INK,            # neighbourhood centre star
    demolished=RED,        # streets demolished by the highway (R2)
)
# HOLC grade -> colour (A green, B blue, C yellow, D red -- the redlining scheme)
GRADE_COLOR = {"A": GREEN, "B": BLUE, "C": YELLOW, "D": RED}


def dot(color, ms=6.5):
    """point-marker kwargs with a white halo so it sits on the map."""
    return dict(marker="o", markersize=ms, markeredgecolor="white",
                markeredgewidth=1.2, linestyle="none", color=color)


def star(color=INK, ms=17):
    return dict(marker="*", markersize=ms, markeredgecolor="white",
                markeredgewidth=1.1, linestyle="none", color=color)


def _handle(kind, color, label):
    if kind == "fill":
        return Patch(facecolor=color, alpha=0.16, edgecolor=color,
                     linewidth=0.8, label=label)
    if kind == "band":
        return Patch(facecolor=color, alpha=0.35, edgecolor="none", label=label)
    if kind == "line":
        return Line2D([0], [0], color=color, lw=2.6, label=label)
    if kind == "thin":
        return Line2D([0], [0], color=color, lw=1.0, label=label)
    if kind == "dashed":
        return Line2D([0], [0], color=color, lw=2.0, ls=(0, (3, 2)), label=label)
    if kind == "dot":
        return Line2D([0], [0], **dot(color, ms=8), label=label)
    if kind == "star":
        return Line2D([0], [0], **star(color), label=label)
    raise ValueError(kind)


def legend(ax, entries, loc="upper left", ncol=1, fontsize=8):
    """entries: list of (kind, color, label); kind in
    {fill, band, line, thin, dashed, dot, star}."""
    handles = [_handle(k, c, l) for k, c, l in entries]
    lg = ax.legend(handles=handles, loc=loc, ncol=ncol, fontsize=fontsize,
                   frameon=True, facecolor="white", edgecolor=GRID,
                   framealpha=0.93, labelcolor=INK2, borderpad=0.7,
                   handlelength=1.6, labelspacing=0.5)
    lg.set_zorder(30)
    return lg


def title(ax, text, fontsize=10):
    ax.set_title(text, fontsize=fontsize, color=INK)
