"""
Visualization module for Fuzzy Inference System.
Creates dark-mode, publication-grade Matplotlib plots for:
1. Input 1 (CIBIL Score) Membership Functions with live operating point
2. Input 2 (Monthly Income) Membership Functions with live operating point
3. Aggregated Output (Loan Approval Chance) with Centroid defuzzification
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Headless backend for web servers
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, Any


# Modern Banking Dark Theme Palette
STYLE_CONFIG = {
    "bg_color": "#0d1424",        # Deep navy card background
    "plot_bg": "#111b33",         # Plot area background
    "grid_color": "#1f2e4d",      # Subtle grid lines
    "text_color": "#E2E8F0",      # Bright clean text
    "muted_text": "#94A3B8",      # Subtitle/axis label text
    "cibil_poor": "#EF4444",      # Soft Crimson
    "cibil_avg": "#F59E0B",       # Amber Orange
    "cibil_good": "#10B981",      # Emerald Green
    "income_low": "#F97316",      # Vibrant Orange
    "income_med": "#38BDF8",      # Sky Blue
    "income_high": "#6366F1",     # Indigo
    "output_reject": "#F43F5E",   # Rose Red
    "output_hold": "#FBBF24",     # Golden Amber
    "output_approve": "#22C55E",  # Vivid Green
    "accent_orange": "#FF7043",
    "accent_blue": "#00B4D8"
}


def _apply_dark_theme(fig, ax):
    """Applies high-end dark mode aesthetics to a matplotlib figure."""
    fig.patch.set_facecolor(STYLE_CONFIG["bg_color"])
    ax.set_facecolor(STYLE_CONFIG["plot_bg"])
    ax.tick_params(colors=STYLE_CONFIG["muted_text"], labelsize=10)
    for spine in ax.spines.values():
        spine.set_color("#233554")
        spine.set_linewidth(1.2)
    ax.grid(True, linestyle="--", alpha=0.35, color=STYLE_CONFIG["grid_color"])


def plot_cibil_mfs(fis, current_cibil: float, cibil_degs: Dict[str, float]) -> plt.Figure:
    """Plots CIBIL Score Membership Functions and marks current input."""
    fig, ax = plt.subplots(figsize=(6.8, 3.2), dpi=140)
    _apply_dark_theme(fig, ax)

    x = fis.cibil_universe
    poor_y = fis._compute_mf(x, *fis.cibil_params["Poor"])
    avg_y = fis._compute_mf(x, *fis.cibil_params["Average"])
    good_y = fis._compute_mf(x, *fis.cibil_params["Good"])

    # Base curves
    ax.plot(x, poor_y, label="Poor (≤640)", color=STYLE_CONFIG["cibil_poor"], lw=2.2)
    ax.plot(x, avg_y, label="Average (580-770)", color=STYLE_CONFIG["cibil_avg"], lw=2.2)
    ax.plot(x, good_y, label="Good (≥720)", color=STYLE_CONFIG["cibil_good"], lw=2.2)

    # Shaded subtle fills
    ax.fill_between(x, 0, poor_y, color=STYLE_CONFIG["cibil_poor"], alpha=0.08)
    ax.fill_between(x, 0, avg_y, color=STYLE_CONFIG["cibil_avg"], alpha=0.08)
    ax.fill_between(x, 0, good_y, color=STYLE_CONFIG["cibil_good"], alpha=0.08)

    # Current Operating Point
    ax.axvline(x=current_cibil, color=STYLE_CONFIG["accent_orange"], linestyle="--", lw=2, alpha=0.9, label=f"Input: {int(current_cibil)}")
    
    # Active membership dots
    for term, color in [("Poor", STYLE_CONFIG["cibil_poor"]), ("Average", STYLE_CONFIG["cibil_avg"]), ("Good", STYLE_CONFIG["cibil_good"])]:
        deg = cibil_degs[term]
        if deg > 0.01:
            ax.plot(current_cibil, deg, "o", color=color, markersize=8, markeredgecolor="#FFFFFF", markeredgewidth=1.5)
            ax.annotate(f"μ={deg:.2f}", (current_cibil, deg), textcoords="offset points", xytext=(8, 4),
                        color="#FFFFFF", fontsize=9, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", fc=color, ec="none", alpha=0.85))

    ax.set_title("CIBIL Score Fuzzy Partitions & Active Membership", fontsize=11, fontweight="bold", color=STYLE_CONFIG["text_color"], pad=10)
    ax.set_xlabel("CIBIL Score (300 - 900)", fontsize=9, color=STYLE_CONFIG["muted_text"])
    ax.set_ylabel("Degree μ(x)", fontsize=9, color=STYLE_CONFIG["muted_text"])
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlim(300, 900)
    ax.legend(loc="upper right", fontsize=8, facecolor=STYLE_CONFIG["bg_color"], edgecolor="#233554", labelcolor=STYLE_CONFIG["text_color"])
    fig.tight_layout()
    return fig


def plot_income_mfs(fis, current_income: float, income_degs: Dict[str, float]) -> plt.Figure:
    """Plots Monthly Income Membership Functions and marks current input."""
    fig, ax = plt.subplots(figsize=(6.8, 3.2), dpi=140)
    _apply_dark_theme(fig, ax)

    x = fis.income_universe
    low_y = fis._compute_mf(x, *fis.income_params["Low"])
    med_y = fis._compute_mf(x, *fis.income_params["Medium"])
    high_y = fis._compute_mf(x, *fis.income_params["High"])

    # X axis in thousands for cleaner labels
    x_k = x / 1000.0
    current_k = current_income / 1000.0

    ax.plot(x_k, low_y, label="Low (≤₹60k)", color=STYLE_CONFIG["income_low"], lw=2.2)
    ax.plot(x_k, med_y, label="Medium (₹45k-135k)", color=STYLE_CONFIG["income_med"], lw=2.2)
    ax.plot(x_k, high_y, label="High (≥₹110k)", color=STYLE_CONFIG["income_high"], lw=2.2)

    ax.fill_between(x_k, 0, low_y, color=STYLE_CONFIG["income_low"], alpha=0.08)
    ax.fill_between(x_k, 0, med_y, color=STYLE_CONFIG["income_med"], alpha=0.08)
    ax.fill_between(x_k, 0, high_y, color=STYLE_CONFIG["income_high"], alpha=0.08)

    # Current Operating Point
    ax.axvline(x=current_k, color=STYLE_CONFIG["accent_blue"], linestyle="--", lw=2, alpha=0.9, label=f"Input: ₹{current_k:.1f}k")

    for term, color in [("Low", STYLE_CONFIG["income_low"]), ("Medium", STYLE_CONFIG["income_med"]), ("High", STYLE_CONFIG["income_high"])]:
        deg = income_degs[term]
        if deg > 0.01:
            ax.plot(current_k, deg, "o", color=color, markersize=8, markeredgecolor="#FFFFFF", markeredgewidth=1.5)
            ax.annotate(f"μ={deg:.2f}", (current_k, deg), textcoords="offset points", xytext=(8, 4),
                        color="#FFFFFF", fontsize=9, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", fc=color, ec="none", alpha=0.85))

    ax.set_title("Monthly Income Fuzzy Partitions & Active Membership", fontsize=11, fontweight="bold", color=STYLE_CONFIG["text_color"], pad=10)
    ax.set_xlabel("Monthly Income (₹ in Thousands)", fontsize=9, color=STYLE_CONFIG["muted_text"])
    ax.set_ylabel("Degree μ(x)", fontsize=9, color=STYLE_CONFIG["muted_text"])
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlim(15, 250)
    ax.legend(loc="upper right", fontsize=8, facecolor=STYLE_CONFIG["bg_color"], edgecolor="#233554", labelcolor=STYLE_CONFIG["text_color"])
    fig.tight_layout()
    return fig


def plot_defuzzification_results(fis, inference_result: Dict[str, Any]) -> plt.Figure:
    """
    Plots the Mamdani Aggregated Output Surface and Defuzzified Centroid (Center of Gravity).
    """
    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=140)
    _apply_dark_theme(fig, ax)

    y = fis.approval_universe
    agg = inference_result["aggregated_curve"]
    centroid = inference_result["crisp_score"]
    term_curves = inference_result["term_curves"]

    # Plot individual baseline linguistic curves (faint dashed lines)
    ax.plot(y, fis.output_mfs["Reject"], linestyle=":", color=STYLE_CONFIG["output_reject"], lw=1.2, alpha=0.45, label="Reject Base")
    ax.plot(y, fis.output_mfs["Hold"], linestyle=":", color=STYLE_CONFIG["output_hold"], lw=1.2, alpha=0.45, label="Hold Base")
    ax.plot(y, fis.output_mfs["Approve"], linestyle=":", color=STYLE_CONFIG["output_approve"], lw=1.2, alpha=0.45, label="Approve Base")

    # Plot implied truncated curves for each term
    ax.plot(y, term_curves["Reject"], color=STYLE_CONFIG["output_reject"], lw=1.8, label="Implied Reject")
    ax.plot(y, term_curves["Hold"], color=STYLE_CONFIG["output_hold"], lw=1.8, label="Implied Hold")
    ax.plot(y, term_curves["Approve"], color=STYLE_CONFIG["output_approve"], lw=1.8, label="Implied Approve")

    # Highlight Aggregated Fuzzy Set (Mamdani MAX S-Norm)
    ax.fill_between(y, 0, agg, color="#3B82F6", alpha=0.35, label="Aggregated Fuzzy Set μ_agg(y)")
    ax.plot(y, agg, color="#60A5FA", lw=2.5)

    # Plot Defuzzified Centroid
    ax.axvline(x=centroid, color="#F97316", linestyle="-", lw=2.8, alpha=0.95, label=f"Centroid (CoG) = {centroid:.1f}%")
    ax.plot(centroid, 0.02, "^", color="#F97316", markersize=12, markeredgecolor="#FFFFFF", markeredgewidth=1.8)

    # Centroid callout badge
    y_centroid_h = float(np.interp(centroid, y, agg))
    ax.annotate(
        f"Centroid: {centroid:.1f}%\nDecision: {inference_result['decision']}",
        xy=(centroid, max(0.25, y_centroid_h)),
        xytext=(centroid + (10 if centroid < 60 else -35), 0.72),
        arrowprops=dict(facecolor="#F97316", edgecolor="#FFFFFF", arrowstyle="->", lw=1.5),
        fontsize=9, fontweight="bold", color="#FFFFFF",
        bbox=dict(boxstyle="round,pad=0.4", fc="#1E293B", ec="#F97316", lw=1.5, alpha=0.95)
    )

    ax.set_title("Mamdani Aggregated Output Space & Centroid Defuzzification", fontsize=11, fontweight="bold", color=STYLE_CONFIG["text_color"], pad=10)
    ax.set_xlabel("Loan Approval Chance (%)", fontsize=9, color=STYLE_CONFIG["muted_text"])
    ax.set_ylabel("Aggregated Membership μ_agg(y)", fontsize=9, color=STYLE_CONFIG["muted_text"])
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlim(0, 100)
    ax.legend(loc="upper left", fontsize=7.5, facecolor=STYLE_CONFIG["bg_color"], edgecolor="#233554", labelcolor=STYLE_CONFIG["text_color"], ncol=2)
    fig.tight_layout()
    return fig
