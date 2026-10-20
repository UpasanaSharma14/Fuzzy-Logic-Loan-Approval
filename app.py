"""
=============================================================================
CrediFuzz AI - Next-Generation Intelligent Credit Underwriting System
Self-Constructed Mamdani Fuzzy Inference System for Loan Approval
=============================================================================
Modular Frontend built with Streamlit & Custom CSS Glassmorphism
Zero reliance on scikit-fuzzy; powered by pure NumPy mathematical functions.
"""

import os
import textwrap
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import matplotlib.pyplot as plt

# Import self-made modular fuzzy logic engine and dark-mode visualizer
from fuzzy_engine import LoanFuzzyInferenceSystem
from visualization import plot_cibil_mfs, plot_income_mfs, plot_defuzzification_results

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CrediFuzz AI | Intelligent Underwriting System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Helper function to reliably render HTML/SVG without markdown indentation conflicts
def render_html(html_content: str):
    """Safely renders HTML without markdown whitespace conflicts."""
    st.html(textwrap.dedent(html_content).strip())


# -----------------------------------------------------------------------------
# 2. SLEEK DARK MODE & BLUE/ORANGE BRANDING CSS INJECTION
# -----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
    color: #F1F5F9;
}

/* Background canvas */
.stApp {
    background: radial-gradient(circle at 10% 15%, #0e172a 0%, #070a12 90%) !important;
}

/* Top Padding reduction for low-scroll layout */
.block-container {
    padding-top: 1rem !important;
    padding-bottom: 2rem !important;
    max-width: 1400px !important;
}

/* Rounded banner styling */
img[data-testid="stImage"] {
    border-radius: 14px;
    border: 1px solid rgba(59, 130, 246, 0.25);
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.6);
    max-height: 200px;
    object-fit: cover;
}

/* Native Streamlit Bordered Container as Sleek Dark Glass Cards */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: linear-gradient(135deg, rgba(17, 24, 39, 0.90) 0%, rgba(13, 20, 36, 0.96) 100%) !important;
    border: 1px solid rgba(59, 130, 246, 0.22) !important;
    border-radius: 16px !important;
    padding: 16px 18px !important;
    margin-bottom: 12px !important;
    box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.06) !important;
    backdrop-filter: blur(12px) !important;
    transition: all 0.25s ease-in-out !important;
}

[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: rgba(255, 112, 67, 0.45) !important;
    box-shadow: 0 12px 30px -4px rgba(0, 0, 0, 0.65), 0 0 14px rgba(255, 112, 67, 0.12) !important;
}

/* Glowing Pulsing Status Badges */
.pulse-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 14px;
    border-radius: 9999px;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* Metric Display Tiles */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin: 10px 0 0 0;
}

.metric-tile {
    background: rgba(13, 22, 41, 0.7);
    border: 1px solid rgba(59, 130, 246, 0.18);
    border-radius: 12px;
    padding: 10px 12px;
    text-align: center;
    transition: transform 0.2s;
}

.metric-tile:hover {
    transform: translateY(-2px);
    border-color: rgba(0, 180, 216, 0.5);
}

.metric-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 2px;
}

.metric-value {
    font-size: 1.25rem;
    font-weight: 800;
    color: #FFFFFF;
}

/* Preset buttons styling */
div.stButton > button {
    background: linear-gradient(135deg, #1E3A8A 0%, #1D4ED8 100%) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(96, 165, 250, 0.4) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    padding: 6px 12px !important;
    box-shadow: 0 4px 12px rgba(29, 78, 216, 0.25) !important;
    transition: all 0.2s ease !important;
    width: 100%;
}

div.stButton > button:hover {
    background: linear-gradient(135deg, #EA580C 0%, #F97316 100%) !important;
    border-color: rgba(251, 146, 60, 0.6) !important;
    color: #FFFFFF !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 16px rgba(249, 115, 22, 0.4) !important;
}

/* Custom Streamlit Sliders Customization */
div[data-testid="stThumbValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    color: #FF9800 !important;
    font-weight: bold !important;
}

/* Custom Mamdani Rule Execution Matrix */
.matrix-grid {
    display: grid;
    grid-template-columns: 80px repeat(3, 1fr);
    gap: 6px;
    margin-top: 8px;
    font-size: 0.8rem;
}

.matrix-header {
    background: rgba(30, 41, 59, 0.8);
    padding: 6px;
    text-align: center;
    border-radius: 6px;
    font-weight: 700;
    color: #94A3B8;
}

.matrix-cell {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(59, 130, 246, 0.15);
    padding: 8px 6px;
    text-align: center;
    border-radius: 6px;
    transition: all 0.2s;
}

.matrix-cell-active {
    background: linear-gradient(135deg, rgba(249, 115, 22, 0.25) 0%, rgba(37, 99, 235, 0.25) 100%);
    border: 1.5px solid #F97316;
    box-shadow: 0 0 10px rgba(249, 115, 22, 0.35);
    font-weight: 700;
}

/* Progress bar inside cards */
.fuzz-meter-bg {
    background: rgba(255, 255, 255, 0.08);
    height: 7px;
    border-radius: 999px;
    overflow: hidden;
    margin-top: 3px;
}
.fuzz-meter-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.3s ease;
}
</style>
"""

render_html(CUSTOM_CSS)


# -----------------------------------------------------------------------------
# 3. DYNAMIC HIGH-TECH GAUGE CHART (SVG VIA COMPONENTS)
# -----------------------------------------------------------------------------
def render_modern_gauge(score: float, decision: str, decision_color: str, risk_badge: str):
    """Renders a responsive, high-precision SVG semi-circular gauge inside an isolated component."""
    rotation_deg = -90.0 + (score / 100.0) * 180.0
    rad = np.radians(rotation_deg)
    nx = 150.0 + 75.0 * np.cos(rad)
    ny = 135.0 + 75.0 * np.sin(rad)

    badge_bg = f"{decision_color}22"
    badge_border = f"{decision_color}66"

    gauge_html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700;800&family=JetBrains+Mono:wght@700;800&display=swap');
body {{
    margin: 0;
    padding: 0;
    background: transparent;
    font-family: 'Outfit', sans-serif;
    color: #FFFFFF;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    overflow: hidden;
}}
.gauge-container {{
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    width: 100%;
}}
.score-text {{
    font-size: 2.1rem;
    font-weight: 800;
    font-family: 'JetBrains Mono', monospace;
    color: #FFFFFF;
    text-shadow: 0 0 16px {decision_color}88;
    margin-top: -6px;
    line-height: 1;
}}
.pulse-badge {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    background: {badge_bg};
    color: {decision_color};
    border: 1px solid {badge_border};
    box-shadow: 0 0 14px {decision_color}44;
    margin-top: 6px;
}}
</style>
</head>
<body>
<div class="gauge-container">
    <svg viewBox="0 0 300 165" style="width: 100%; max-width: 275px; filter: drop-shadow(0 6px 12px rgba(0,0,0,0.5));" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <linearGradient id="gaugeGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stop-color="#EF4444" />
                <stop offset="38%" stop-color="#F59E0B" />
                <stop offset="68%" stop-color="#10B981" />
                <stop offset="100%" stop-color="#00E676" />
            </linearGradient>
            <filter id="gaugeGlow">
                <feGaussianBlur stdDeviation="2.5" result="blur"/>
                <feMerge>
                    <feMergeNode in="blur"/>
                    <feMergeNode in="SourceGraphic"/>
                </feMerge>
            </filter>
        </defs>

        <!-- Background Track -->
        <path d="M 32 135 A 118 118 0 0 1 268 135" fill="none" stroke="#1E293B" stroke-width="20" stroke-linecap="round"/>

        <!-- Colored Gauge Arc -->
        <path d="M 32 135 A 118 118 0 0 1 268 135" fill="none" stroke="url(#gaugeGradient)" stroke-width="18" stroke-linecap="round"/>

        <!-- Needle -->
        <line x1="150" y1="130" x2="{nx:.2f}" y2="{ny:.2f}" stroke="#F97316" stroke-width="4.2" stroke-linecap="round" filter="url(#gaugeGlow)"/>
        <line x1="150" y1="130" x2="{nx:.2f}" y2="{ny:.2f}" stroke="#FFFFFF" stroke-width="1.8" stroke-linecap="round"/>

        <!-- Center Hub -->
        <circle cx="150" cy="130" r="8.5" fill="#0F172A" stroke="#F97316" stroke-width="2.5"/>
        <circle cx="150" cy="130" r="3.5" fill="#FFFFFF"/>

        <!-- Endpoints -->
        <text x="32" y="156" font-family="'Outfit', sans-serif" font-size="9.5" font-weight="700" fill="#EF4444">0% REJECT</text>
        <text x="126" y="38" font-family="'Outfit', sans-serif" font-size="9.5" font-weight="600" fill="#F59E0B">HOLD (50%)</text>
        <text x="206" y="156" font-family="'Outfit', sans-serif" font-size="9.5" font-weight="700" fill="#10B981">100% APPROVE</text>
    </svg>

    <div class="score-text">
        {score:.1f}<span style="font-size: 1.25rem; color: #94A3B8;">%</span>
    </div>
    <div class="pulse-badge">
        ● {decision}
    </div>
    <div style="font-size: 0.72rem; font-weight: 600; color: #64748B; margin-top: 4px; letter-spacing: 0.05em;">
        RISK: <span style="color: {decision_color}; font-weight: 700;">{risk_badge}</span>
    </div>
</div>
</body>
</html>
"""
    components.html(gauge_html, height=215, scrolling=False)


# -----------------------------------------------------------------------------
# 4. INITIALIZE FUZZY SYSTEM IN SESSION STATE
# -----------------------------------------------------------------------------
if "fis" not in st.session_state:
    st.session_state.fis = LoanFuzzyInferenceSystem()

fis = st.session_state.fis

# Default values for inputs
if "cibil_val" not in st.session_state:
    st.session_state.cibil_val = 745
if "income_val" not in st.session_state:
    st.session_state.income_val = 85000


# Preset scenario setter callbacks
def set_scenario(cibil, income):
    st.session_state.cibil_val = cibil
    st.session_state.income_val = income


# -----------------------------------------------------------------------------
# 5. HEADER & BANNER SECTION
# -----------------------------------------------------------------------------
banner_path = os.path.join(os.path.dirname(__file__), "banner.jpg")

if os.path.exists(banner_path):
    st.image(banner_path, use_container_width=True)

header_html = """
<div style="margin-top: 6px; margin-bottom: 14px;">
    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
        <div>
            <div style="display: inline-flex; align-items: center; gap: 6px; background: rgba(59, 130, 246, 0.15); border: 1px solid rgba(59, 130, 246, 0.35); padding: 3px 10px; border-radius: 999px; margin-bottom: 6px;">
                <span style="color: #60A5FA; font-size: 0.75rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase;">
                    ⚡ Self-Constructed NumPy Fuzzy Inference Engine • Mamdani Rule Base
                </span>
            </div>
            <h1 style="font-size: 1.95rem; font-weight: 800; color: #FFFFFF; margin: 0; letter-spacing: -0.02em;">
                CrediFuzz <span style="color: #FF7043;">AI Underwriting</span> Portal
            </h1>
            <p style="color: #94A3B8; font-size: 0.9rem; margin: 3px 0 0 0;">
                Autonomous multi-criteria loan eligibility evaluation handling epistemic uncertainty with Centroid defuzzification.
            </p>
        </div>
        <div style="display: flex; gap: 8px;">
            <div style="background: rgba(15, 23, 42, 0.85); border: 1px solid rgba(59, 130, 246, 0.25); border-radius: 10px; padding: 8px 14px; text-align: right;">
                <div style="font-size: 0.68rem; color: #94A3B8; font-weight: 600;">ALGORITHM</div>
                <div style="font-size: 0.88rem; font-weight: 800; color: #38BDF8;">Mamdani Min-Max (NumPy)</div>
            </div>
            <div style="background: rgba(15, 23, 42, 0.85); border: 1px solid rgba(249, 115, 22, 0.25); border-radius: 10px; padding: 8px 14px; text-align: right;">
                <div style="font-size: 0.68rem; color: #94A3B8; font-weight: 600;">DEFUZZIFICATION</div>
                <div style="font-size: 0.88rem; font-weight: 800; color: #FB923C;">Centroid (CoG)</div>
            </div>
        </div>
    </div>
</div>
"""
render_html(header_html)


# -----------------------------------------------------------------------------
# 6. MAIN TWO-COLUMN DASHBOARD LAYOUT
# -----------------------------------------------------------------------------
col_left, col_right = st.columns([0.88, 1.12], gap="medium")


# =============================================================================
# LEFT COLUMN: INPUT CONTROLS & FUZZIFICATION BREAKDOWN
# =============================================================================
with col_left:
    with st.container(border=True):
        render_html("""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <h3 style="margin: 0; font-size: 1.12rem; font-weight: 700; color: #FFFFFF;">
                    🎛️ Credit Assessment Parameters
                </h3>
                <span style="font-size: 0.72rem; background: rgba(59, 130, 246, 0.2); color: #60A5FA; padding: 3px 8px; border-radius: 6px; font-weight: 600;">
                    Step 1: Crisp Inputs
                </span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 6px;">
                <span style="font-weight: 600; font-size: 0.9rem; color: #E2E8F0;">CIBIL Credit Score</span>
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #FF7043; font-weight: 700;">
                    Range: 300 - 900
                </span>
            </div>
        """)

        cibil_input = st.slider(
            "CIBIL Score",
            min_value=300,
            max_value=900,
            value=st.session_state.cibil_val,
            step=5,
            label_visibility="collapsed",
            key="cibil_slider",
            on_change=lambda: st.session_state.update(cibil_val=st.session_state.cibil_slider)
        )

        render_html(f"""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 8px;">
                <span style="font-weight: 600; font-size: 0.9rem; color: #E2E8F0;">Monthly Net Income</span>
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #38BDF8; font-weight: 700;">
                    ₹{st.session_state.income_val:,.0f} / month
                </span>
            </div>
        """)

        income_input = st.slider(
            "Monthly Income",
            min_value=15000,
            max_value=250000,
            value=st.session_state.income_val,
            step=2500,
            label_visibility="collapsed",
            key="income_slider",
            on_change=lambda: st.session_state.update(income_val=st.session_state.income_slider)
        )

        render_html("<div style='margin-top: 10px; margin-bottom: 6px; font-size: 0.76rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em;'>Quick Scenario Presets:</div>")
        pcol1, pcol2 = st.columns(2)
        with pcol1:
            if st.button("🌟 Prime (790, ₹1.5L)"):
                set_scenario(790, 150000)
                st.rerun()
            if st.button("⚖️ Borderline (660, ₹70k)"):
                set_scenario(660, 70000)
                st.rerun()
        with pcol2:
            if st.button("⚠️ Subprime (480, ₹28k)"):
                set_scenario(480, 28000)
                st.rerun()
            if st.button("🔀 Low CIBIL + Rich (540, ₹2.1L)"):
                set_scenario(540, 210000)
                st.rerun()

    # -------------------------------------------------------------------------
    # RUN INFERENCE
    # -------------------------------------------------------------------------
    results = fis.infer(cibil_input, income_input)
    cibil_degs = results["cibil_degrees"]
    income_degs = results["income_degrees"]

    # Fuzzification Inspector Card
    with st.container(border=True):
        fuzz_html = """
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <h3 style="margin: 0; font-size: 1.05rem; font-weight: 700; color: #FFFFFF;">
                🔍 Fuzzification Membership Degrees
            </h3>
            <span style="font-size: 0.72rem; background: rgba(249, 115, 22, 0.2); color: #FB923C; padding: 3px 8px; border-radius: 6px; font-weight: 600;">
                Step 2: Partitioning
            </span>
        </div>
        <div style="font-size: 0.78rem; font-weight: 700; color: #94A3B8; margin-bottom: 4px;">CIBIL SCORE MEMBERSHIP:</div>
        """
        for term, color in [("Poor", "#EF4444"), ("Average", "#F59E0B"), ("Good", "#10B981")]:
            deg = cibil_degs[term]
            fuzz_html += f"""
            <div style="margin-top: 5px;">
                <div style="display: flex; justify-content: space-between; font-size: 0.78rem;">
                    <span style="color: {color}; font-weight: 600;">{term}</span>
                    <span style="font-family: 'JetBrains Mono', monospace; font-weight: 700; color: #FFFFFF;">μ = {deg:.3f}</span>
                </div>
                <div class="fuzz-meter-bg">
                    <div class="fuzz-meter-fill" style="width: {deg*100:.1f}%; background: {color};"></div>
                </div>
            </div>
            """

        fuzz_html += """
        <hr style="border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 10px 0;">
        <div style="font-size: 0.78rem; font-weight: 700; color: #94A3B8; margin-bottom: 4px;">MONTHLY INCOME MEMBERSHIP:</div>
        """
        for term, color in [("Low", "#F97316"), ("Medium", "#38BDF8"), ("High", "#818CF8")]:
            deg = income_degs[term]
            fuzz_html += f"""
            <div style="margin-top: 5px;">
                <div style="display: flex; justify-content: space-between; font-size: 0.78rem;">
                    <span style="color: {color}; font-weight: 600;">{term}</span>
                    <span style="font-family: 'JetBrains Mono', monospace; font-weight: 700; color: #FFFFFF;">μ = {deg:.3f}</span>
                </div>
                <div class="fuzz-meter-bg">
                    <div class="fuzz-meter-fill" style="width: {deg*100:.1f}%; background: {color};"></div>
                </div>
            </div>
            """
        render_html(fuzz_html)

    # Mamdani Rule Execution Matrix Grid
    with st.container(border=True):
        terms_income = ["Low", "Medium", "High"]
        terms_cibil = ["Poor", "Average", "Good"]

        conseq_map = {
            ("Poor", "Low"): ("Reject", 1),
            ("Poor", "Medium"): ("Reject", 2),
            ("Poor", "High"): ("Hold", 3),
            ("Average", "Low"): ("Reject", 4),
            ("Average", "Medium"): ("Hold", 5),
            ("Average", "High"): ("Approve", 6),
            ("Good", "Low"): ("Hold", 7),
            ("Good", "Medium"): ("Approve", 8),
            ("Good", "High"): ("Approve", 9),
        }

        matrix_html = """
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
            <h3 style="margin: 0; font-size: 1.02rem; font-weight: 700; color: #FFFFFF;">
                📋 3×3 Mamdani Decision Matrix
            </h3>
            <span style="font-size: 0.72rem; color: #94A3B8;">MIN Operator</span>
        </div>
        <div class="matrix-grid">
            <div class="matrix-header">CIBIL \\ Inc</div>
            <div class="matrix-header">Low</div>
            <div class="matrix-header">Medium</div>
            <div class="matrix-header">High</div>
        """

        for c_term in terms_cibil:
            matrix_html += f"<div class='matrix-header'>{c_term}</div>"
            for i_term in terms_income:
                conseq, r_id = conseq_map[(c_term, i_term)]
                premise_weight = min(cibil_degs[c_term], income_degs[i_term])
                is_active = premise_weight > 0.001
                active_class = "matrix-cell-active" if is_active else ""
                conseq_color = "#10B981" if conseq == "Approve" else "#F59E0B" if conseq == "Hold" else "#EF4444"
                weight_text = f"<div style='font-size: 0.68rem; color: #FB923C; font-weight:bold;'>w={premise_weight:.2f}</div>" if is_active else ""
                matrix_html += f"""
                    <div class="matrix-cell {active_class}">
                        <div style="font-weight: 700; color: {conseq_color};">{conseq}</div>
                        {weight_text}
                    </div>
                """

        matrix_html += "</div>"
        render_html(matrix_html)


# =============================================================================
# RIGHT COLUMN: CRISP RESULTS, GAUGE & DYNAMIC MATPLOTLIB VISUALIZATIONS
# =============================================================================
with col_right:
    # Top Results Banner Card with Dynamic SVG Gauge
    with st.container(border=True):
        res_col1, res_col2 = st.columns([1.05, 0.95], gap="small")
        with res_col1:
            render_html(f"""
                <div style="padding-top: 4px;">
                    <div style="font-size: 0.74rem; font-weight: 700; color: #38BDF8; letter-spacing: 0.08em; text-transform: uppercase;">
                        Lending Decision Output
                    </div>
                    <h2 style="font-size: 1.85rem; font-weight: 800; color: {results['decision_color']}; margin: 2px 0 8px 0;">
                        {results['decision']}
                    </h2>
                    <div style="background: rgba(255, 255, 255, 0.04); border-left: 3px solid {results['decision_color']}; padding: 10px 12px; border-radius: 0 8px 8px 0; margin-bottom: 8px;">
                        <div style="font-size: 0.82rem; color: #CBD5E1; line-height: 1.4;">
                            {results['recommendation']}
                        </div>
                    </div>
                    <div style="font-size: 0.78rem; color: #94A3B8;">
                        Credit Risk Tier: <span style="color: #FFFFFF; font-weight: 600;">{results['risk_tier']}</span>
                    </div>
                </div>
            """)

        with res_col2:
            render_modern_gauge(
                results["crisp_score"],
                results["decision"],
                results["decision_color"],
                results["risk_badge"]
            )

        # Key Quantitative Metrics
        render_html(f"""
            <div class="metric-grid">
                <div class="metric-tile">
                    <div class="metric-label">Crisp Centroid Score</div>
                    <div class="metric-value" style="color: #FB923C;">{results['crisp_score']:.1f}%</div>
                </div>
                <div class="metric-tile">
                    <div class="metric-label">Mamdani Active Rules</div>
                    <div class="metric-value" style="color: #38BDF8;">{len(results['active_rules'])} / 9</div>
                </div>
                <div class="metric-tile">
                    <div class="metric-label">Max Rule Fire Strength</div>
                    <div class="metric-value" style="color: #34D399;">
                        {max([r['weight'] for r in results['rule_results']]):.2f}
                    </div>
                </div>
            </div>
        """)

    # -------------------------------------------------------------------------
    # DYNAMIC VISUALIZATIONS SECTION (MATPLOTLIB)
    # -------------------------------------------------------------------------
    with st.container(border=True):
        render_html("""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <h3 style="margin: 0; font-size: 1.08rem; font-weight: 700; color: #FFFFFF;">
                    📈 Dynamic Membership & Defuzzification Visualizers
                </h3>
                <span style="font-size: 0.72rem; color: #38BDF8; background: rgba(56, 189, 248, 0.15); padding: 3px 8px; border-radius: 6px; font-weight: 600;">
                    Self-Generated via Matplotlib
                </span>
            </div>
        """)

        # Tabs for Visualizations
        tab1, tab2, tab3 = st.tabs([
            "🎯 Defuzzified Output Area",
            "📊 CIBIL Score Partitions",
            "💵 Income Level Partitions"
        ])

        with tab1:
            render_html("<div style='font-size: 0.78rem; color: #94A3B8; margin-bottom: 6px;'>Mamdani MIN implication curves aggregated with MAX operator; vertical line shows exact Center of Gravity (Centroid).</div>")
            fig_output = plot_defuzzification_results(fis, results)
            st.pyplot(fig_output, use_container_width=True)
            plt.close(fig_output)

        with tab2:
            render_html("<div style='font-size: 0.78rem; color: #94A3B8; margin-bottom: 6px;'>CIBIL universe partitions: Poor (Trapezoidal), Average (Triangular), Good (Trapezoidal).</div>")
            fig_cibil = plot_cibil_mfs(fis, cibil_input, cibil_degs)
            st.pyplot(fig_cibil, use_container_width=True)
            plt.close(fig_cibil)

        with tab3:
            render_html("<div style='font-size: 0.78rem; color: #94A3B8; margin-bottom: 6px;'>Monthly income partitions: Low (Trapezoidal), Medium (Triangular), High (Trapezoidal).</div>")
            fig_income = plot_income_mfs(fis, income_input, income_degs)
            st.pyplot(fig_income, use_container_width=True)
            plt.close(fig_income)

    # -------------------------------------------------------------------------
    # ACTIVE RULES AUDIT LOG
    # -------------------------------------------------------------------------
    with st.container(border=True):
        rule_log_html = """
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <h3 style="margin: 0; font-size: 1.02rem; font-weight: 700; color: #FFFFFF;">
                🔍 Active Rule Fire Trace
            </h3>
            <span style="font-size: 0.72rem; color: #A7F3D0; background: rgba(16, 185, 129, 0.15); padding: 3px 8px; border-radius: 6px; font-weight: 600;">
                Audit Trail
            </span>
        </div>
        """

        if results["active_rules"]:
            for r in results["active_rules"]:
                conseq_color = "#10B981" if r['consequent'] == "Approve" else "#F59E0B" if r['consequent'] == "Hold" else "#EF4444"
                rule_log_html += f"""
                <div style="background: rgba(15, 23, 42, 0.75); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 8px 12px; margin-bottom: 6px; display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="font-weight: 700; color: #38BDF8; font-size: 0.82rem;">Rule #{r['id']}:</span>
                        <span style="font-size: 0.82rem; color: #E2E8F0; margin-left: 6px;">{r['description']}</span>
                    </div>
                    <div style="text-align: right; min-width: 90px;">
                        <span style="background: {conseq_color}22; color: {conseq_color}; border: 1px solid {conseq_color}66; padding: 2px 7px; border-radius: 6px; font-size: 0.72rem; font-weight: 700;">
                            {r['consequent']}
                        </span>
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #FB923C; font-weight: 600; margin-top: 2px;">
                            Firing w = {r['weight']:.3f}
                        </div>
                    </div>
                </div>
                """
        else:
            rule_log_html += "<div style='color: #94A3B8; font-size: 0.82rem;'>No rules active.</div>"

        render_html(rule_log_html)


# -----------------------------------------------------------------------------
# 7. MATHEMATICAL FORMULATION EXPANDER (FOR DEMONSTRATION & MARKS)
# -----------------------------------------------------------------------------
with st.expander("📐 Mathematical Specification of the Self-Made Engine (NumPy Implementation)"):
    st.markdown("""
    ### 1. Triangular Membership Function
    $$\\mu_{\\text{tri}}(x; a, b, c) = \\max\\left(0, \\min\\left(\\frac{x - a}{b - a}, \\frac{c - x}{c - b}\\right)\\right)$$
    Used for:
    - **CIBIL Average**: $a=580, b=680, c=770$
    - **Income Medium**: $a=45\\text{k}, b=85\\text{k}, c=135\\text{k}$
    - **Approval Hold**: $a=35\\%, b=55\\%, c=75\\%$

    ### 2. Trapezoidal Membership Function
    $$\\mu_{\\text{trap}}(x; a, b, c, d) = \\max\\left(0, \\min\\left(\\frac{x - a}{b - a}, 1, \\frac{d - x}{d - c}\\right)\\right)$$
    Used for:
    - **CIBIL Poor**: $a=300, b=300, c=520, d=640$
    - **CIBIL Good**: $a=720, b=800, c=900, d=900$
    - **Income Low**: $a=15\\text{k}, b=15\\text{k}, c=35\\text{k}, d=60\\text{k}$
    - **Income High**: $a=110\\text{k}, b=160\\text{k}, c=250\\text{k}, d=250\\text{k}$
    - **Approval Reject**: $a=0, b=0, c=20, d=45$
    - **Approval Approve**: $a=65, b=82, c=100, d=100$

    ### 3. Mamdani Rule Evaluation & Implication
    For each rule $i$ with premise $A_i \\text{ AND } B_i \\implies C_i$:
    $$w_i = \\min(\\mu_{A_i}(x_{\\text{cibil}}), \\mu_{B_i}(y_{\\text{income}}))$$
    $$\\mu_{\\text{rule } i}(z) = \\min(w_i, \\mu_{C_i}(z))$$

    ### 4. Fuzzy Aggregation (S-Norm MAX)
    $$\\mu_{\\text{aggregated}}(z) = \\max_{i=1}^9 \\mu_{\\text{rule } i}(z)$$

    ### 5. Centroid (Center of Gravity) Defuzzification
    $$\\text{Centroid} = \\frac{\\int z \\cdot \\mu_{\\text{aggregated}}(z) \\, dz}{\\int \\mu_{\\text{aggregated}}(z) \\, dz} \\approx \\frac{\\sum_{k} z_k \\cdot \\mu_{\\text{aggregated}}(z_k)}{\\sum_{k} \\mu_{\\text{aggregated}}(z_k)}$$
    Discretized over $N=1001$ uniformly spaced points in the domain $[0, 100]$.
    """)

# Footer
render_html("""
<div style="text-align: center; color: #64748B; font-size: 0.78rem; margin-top: 20px; padding-top: 14px; border-top: 1px solid rgba(255,255,255,0.06);">
    CrediFuzz Intelligent Underwriting System • Built with Streamlit, NumPy & Custom Mamdani FIS Engine • Dark Mode Banking Edition
</div>
""")
