"""Render the S4 Stage-1 talk figures as PNGs from s4_analysis. Re-run any time results change.

Figures (replay/figures/):
  fig1_headline   strict + relaxed per model, release order, 95% CIs, Phi-3 baseline band behind them
  fig2_gemma_pair paired Gemma-2 -> Gemma 4: attempts (relaxed) vs exact (strict), the metric-flip story
  fig3_fidelity   Phi-3 successes vs controls, both metrics, with exact-test p-values

Design: dataviz skill. One axis per panel; categorical slots blue/orange (validated); recessive hairline grid;
value labels in ink, not the mark color; CIs as whiskers.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s4_analysis as A  # noqa: E402

FIGDIR = A.REPLAY / "figures"
# dataviz reference palette (light) + chrome tokens
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "savefig.dpi": 150, "savefig.bbox": "tight", "font.size": 10,
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlesize": 12, "axes.titlepad": 10,
    "axes.edgecolor": BASE, "axes.linewidth": 1, "text.color": INK,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK2,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 1,
})


def _style(ax, pct_axis="x", xmax=None):
    ax.spines[["top", "right"]].set_visible(False)
    if pct_axis == "x":
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(axis="y", visible=False)
        if xmax:
            ax.set_xlim(0, xmax)
    else:
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(axis="x", visible=False)
        if xmax:
            ax.set_ylim(0, xmax)


def fig_headline():
    rows = A.headline_table()
    rows = sorted(rows, key=lambda r: r["year"])
    names = [r["name"] for r in rows]
    y = np.arange(len(rows))[::-1]          # newest at top
    h = 0.38
    phi3 = next(r for r in rows if "Phi-3" in r["name"])

    fig, ax = plt.subplots(figsize=(9.2, 0.95 * len(rows) + 1.6))
    # Phi-3 baseline reference bands (the original model's own replay rate = the 2024 yardstick)
    for val, col in ((phi3["strict"], BLUE), (phi3["relaxed"], ORANGE)):
        ax.axvspan(0, val, color=col, alpha=0.06)
        ax.axvline(val, color=col, lw=1, ls=(0, (4, 3)), alpha=0.6)
    top_y = len(rows) - 1
    ax.text(phi3["strict"], top_y + 0.45, "original (strict)", color=BLUE, fontsize=7.5, va="bottom", ha="center")
    ax.text(phi3["relaxed"], top_y + 0.45, "original (relaxed)", color=ORANGE, fontsize=7.5, va="bottom", ha="center")

    def whisk(vals, lo, hi, off, color, label):
        ax.barh(y + off, vals, height=h, color=color, label=label, zorder=3)
        ax.errorbar(vals, y + off, xerr=[np.array(vals) - lo, hi - np.array(vals)],
                    fmt="none", ecolor=INK2, elinewidth=1, capsize=3, zorder=4)
        for yi, v, k, n in zip(y + off, vals, [r_["strict_k" if color == BLUE else "relaxed_k"] for r_ in rows],
                               [r_["evaluable"] for r_ in rows]):
            ax.text(v + 0.004, yi, f"{v:.1%}", va="center", ha="left", color=INK, fontsize=8.5)

    whisk([r["strict"] for r in rows], np.array([r["strict_lo"] for r in rows]),
          np.array([r["strict_hi"] for r in rows]), h / 2, BLUE, "strict (challenge exact-match)")
    whisk([r["relaxed"] for r in rows], np.array([r["relaxed_lo"] for r in rows]),
          np.array([r["relaxed_hi"] for r in rows]), -h / 2, ORANGE, "relaxed (figure leaked, any wording)")

    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['name']}\n{r['year']}  {r['draws']}×{r['attacks']}" for r in rows], fontsize=9)
    _style(ax, "x", xmax=max(0.3, max(r["relaxed_hi"] for r in rows) + 0.04))
    ax.set_title("Do 2024 S4 attacks still exfiltrate on newer models?")
    ax.set_xlabel("per-draw leak rate (pooled over draws) on the 212 attacks that fully succeeded in 2024")
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    ax.set_ylim(-0.7, top_y + 1.0)
    fig.subplots_adjust(bottom=0.16)
    fig.text(0.01, 0.015, "Each attack fully exfiltrated $9.378M in 2024. Rate pooled over draws; whiskers 95% CI. "
             "Dashed lines mark the original model's own replay rate — read newer models against that, not 100%.",
             fontsize=7.5, color=MUTED)
    fig.savefig(FIGDIR / "fig1_headline.png")
    plt.close(fig)


def fig_consistency():
    """Of attacks that leak on >=1 draw, how many leak on EVERY draw (relaxed). Reproducible > flaky for an attacker."""
    rows = sorted(A.consistency(), key=lambda r: r["year"])
    names = [f"{r['name']}\n{r['year']}" for r in rows]
    ever = [r["leak"]["ever"] for r in rows]
    always = [r["leak"]["always"] for r in rows]
    some = [r["leak"]["some"] for r in rows]
    y = np.arange(len(rows))[::-1]
    fig, ax = plt.subplots(figsize=(8.6, 0.8 * len(rows) + 1.4))
    bh = 0.5
    ax.barh(y, always, bh, color=ORANGE, label="leaks on every draw (reproducible)", zorder=3)
    ax.barh(y, some, bh, left=[a + 2 * (a > 0) / 100 for a in always], color=BASE,     # 2px-ish surface gap
            label="leaks on some draws only (flaky)", zorder=3)
    for yi, a, e in zip(y, always, ever):
        if e:
            ax.text(e + 0.3, yi, f"{a}/{e}  ({a/e:.0%} reproducible)", va="center", ha="left", color=INK, fontsize=9)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9)
    ax.set_xlim(0, max(ever) + 14 if ever else 1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", visible=False)
    ax.set_title("Are the leaks reliable? Attacks that leak on every draw vs some")
    ax.set_xlabel("distinct attacks that leak the figure on ≥ 1 of 3 draws (relaxed)")
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    fig.subplots_adjust(bottom=0.2)
    fig.text(0.01, 0.02, "3 draws per attack, temperature 1.0. A reproducible leak (orange) is one an attacker can "
             "rely on; the 2024-era models mostly leak by chance (gray).", fontsize=7.5, color=MUTED)
    fig.savefig(FIGDIR / "fig4_consistency.png")
    plt.close(fig)


def fig_gemma_pair():
    gp = A.gemma_pair()
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    groups = ["Attempted to leak\n(relaxed)", "Exact exfiltration\n(strict)"]
    g2 = [gp["leak"]["g2"] / gp["n"], gp["complied"]["g2"] / gp["n"]]
    g4 = [gp["leak"]["g4"] / gp["n"], gp["complied"]["g4"] / gp["n"]]
    x = np.arange(2); w = 0.36
    b2 = ax.bar(x - w / 2, g2, w, color=BASE, label="Gemma-2-27B (2024)", zorder=3)
    b4 = ax.bar(x + w / 2, g4, w, color=BLUE, label="Gemma 4 31B (2026)", zorder=3)
    for bars, vals in ((b2, g2), (b4, g4)):
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.1%}", ha="center", va="bottom",
                    color=INK, fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(groups)
    _style(ax, "y", xmax=max(g2 + g4) + 0.05)
    ax.set_title("Newer model: fewer attempts, but every one exact")
    ax.set_ylabel("share of jointly-evaluable attacks")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    pL, pS = gp["leak"]["p"], gp["complied"]["p"]
    ax.annotate(f"attempts fall (p={pL:.2f})", (0, max(g2[0], g4[0])), xytext=(0, 14),
                textcoords="offset points", ha="center", color=INK2, fontsize=8)
    ax.annotate(f"exact rises, n.s. (p={pS:.2f})", (1, max(g2[1], g4[1])), xytext=(0, 14),
                textcoords="offset points", ha="center", color=INK2, fontsize=8)
    fig.subplots_adjust(bottom=0.26)
    fig.text(0.01, 0.02, f"n={gp['n']} attacks both models could process. Attempt→exact conversion: "
             f"Gemma-2 {gp['g2_conv'][0]}/{gp['g2_conv'][1]}, Gemma 4 {gp['g4_conv'][0]}/{gp['g4_conv'][1]}. "
             "Metric choice flips which looks worse.", fontsize=7.5, color=MUTED)
    fig.savefig(FIGDIR / "fig2_gemma_pair.png")
    plt.close(fig)


def fig_fidelity():
    fid = A.fidelity()
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    metrics = ["leak", "complied"]
    labels = ["Attempted to leak\n(relaxed)", "Exact exfiltration\n(strict)"]
    succ = [fid[m]["succ_k"] / fid[m]["succ_n"] for m in metrics]
    ctrl = [fid[m]["ctrl_k"] / fid[m]["ctrl_n"] for m in metrics]
    x = np.arange(2); w = 0.36
    bs = ax.bar(x - w / 2, succ, w, color=BLUE, label=f"2024 successes (n={fid['complied']['succ_n']})", zorder=3)
    bc = ax.bar(x + w / 2, ctrl, w, color=BASE, label=f"2024 failures / controls (n={fid['complied']['ctrl_n']})", zorder=3)
    for bars, vals in ((bs, succ), (bc, ctrl)):
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.1%}", ha="center", va="bottom",
                    color=INK, fontsize=9)
    for xi, m in zip(x, metrics):
        top = max(succ[list(metrics).index(m)], ctrl[list(metrics).index(m)])
        ax.annotate(f"p={fid[m]['p']:.1g}", (xi, top), xytext=(0, 16), textcoords="offset points",
                    ha="center", color=INK2, fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    _style(ax, "y", xmax=max(succ + ctrl) + 0.07)
    ax.set_title("Replay fidelity: 2024 winners beat 2024 losers on Phi-3")
    ax.set_ylabel("share of attacks (Phi-3 replay, draw 0)")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    fig.subplots_adjust(bottom=0.26)
    fig.text(0.01, 0.02, "Same model (original Phi-3-medium), same levels. The replay preserves attack strength, "
             "so low absolute reproduction is calibration, not a broken harness.", fontsize=7.5, color=MUTED)
    fig.savefig(FIGDIR / "fig3_fidelity.png")
    plt.close(fig)


def build_all():
    FIGDIR.mkdir(exist_ok=True)
    fig_headline()
    fig_gemma_pair()
    fig_fidelity()
    fig_consistency()
    print(f"wrote figures -> {FIGDIR}")
    for p in sorted(FIGDIR.glob("*.png")):
        print(f"  {p.name}  {p.stat().st_size // 1024} KB")


if __name__ == "__main__":
    build_all()
