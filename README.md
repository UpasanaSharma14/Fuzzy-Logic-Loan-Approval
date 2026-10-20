# CrediFuzz AI — Fuzzy Inference System for Loan Approval

A standalone, production-ready implementation of a **Mamdani Fuzzy Inference System (FIS)** built from first principles using **NumPy** (without `scikit-fuzzy`), coupled with an interactive dark-mode **Streamlit** banking interface.

---

## 🌟 Key Features

1. **Self-Constructed Fuzzy Engine (NumPy Only)**:
   - **Custom Membership Functions**: Boundary-safe Triangular and Trapezoidal implementations.
   - **Inputs**:
     - **CIBIL Score** ($300 - 900$): Linguistic terms `Poor`, `Average`, `Good`.
     - **Monthly Income** ($₹15,000 - ₹250,000$): Linguistic terms `Low`, `Medium`, `High`.
   - **Output**:
     - **Loan Approval Chance** ($0 - 100\%$): Linguistic terms `Reject`, `Hold`, `Approve`.
   - **Mamdani Rule Base**: 9 exhaustive rules covering the full $3 \times 3$ input space using `MIN` for T-Norm (AND) and Mamdani `MIN` implication.
   - **Centroid Defuzzification**: Center of Gravity (CoG) calculated from scratch via numerical integration over $N = 1001$ discrete points.

2. **Fintech UI/UX (Streamlit & Custom CSS)**:
   - **Dark Mode Banking Theme**: Blue and Orange color scheme with custom glassmorphism cards.
   - **Dynamic SVG Gauge Chart**: Semi-circular gauge meter with glowing needle and categorical risk indicators.
   - **Interactive 3×3 Mamdani Decision Matrix**: Real-time visualization of premise activation weights.
   - **Dynamic Matplotlib Visualizers**: Dark-mode plots for input membership curves and the shaded output universe.
   - **Audit Trail**: Real-time breakdown of all active rules and firing strengths.

---

## 📂 Repository Structure

```
fuzzy/
├── app.py              # Streamlit dashboard, custom CSS, & dynamic SVG gauge
├── fuzzy_engine.py     # Pure NumPy Mamdani Fuzzy Inference Engine
├── visualization.py    # Dark-mode Matplotlib visualizers for MFs and output area
├── banner.jpg          # High-tech fintech dashboard hero banner
├── requirements.txt    # Project dependencies
├── .gitignore          # Git ignore rules for Python & environments
└── README.md           # Documentation
```

---

## 📐 Mathematical Formulation

### 1. Triangular Membership Function
$$\mu_{\text{tri}}(x; a, b, c) = \max\left(0, \min\left(\frac{x - a}{b - a}, \frac{c - x}{c - b}\right)\right)$$

### 2. Trapezoidal Membership Function
$$\mu_{\text{trap}}(x; a, b, c, d) = \max\left(0, \min\left(\frac{x - a}{b - a}, 1, \frac{d - x}{d - c}\right)\right)$$

### 3. Mamdani Rule Evaluation & Implication
For each rule $i$ with premise $A_i \text{ AND } B_i \implies C_i$:
$$w_i = \min(\mu_{A_i}(x_{\text{cibil}}), \mu_{B_i}(y_{\text{income}}))$$
$$\mu_{\text{rule } i}(z) = \min(w_i, \mu_{C_i}(z))$$

### 4. Fuzzy Aggregation (S-Norm MAX)
$$\mu_{\text{aggregated}}(z) = \max_{i=1}^9 \mu_{\text{rule } i}(z)$$

### 5. Centroid Defuzzification (Center of Gravity)
$$\text{Centroid} = \frac{\int z \cdot \mu_{\text{aggregated}}(z) \, dz}{\int \mu_{\text{aggregated}}(z) \, dz} \approx \frac{\sum_{k} z_k \cdot \mu_{\text{aggregated}}(z_k)}{\sum_{k} \mu_{\text{aggregated}}(z_k)}$$

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- Git

### Installation

1. Clone or navigate to the repository:
   ```bash
   git clone <repo-url>
   cd fuzzy_project
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch the Streamlit application:
   ```bash
   python -m streamlit run app.py
   ```

4. Open `http://localhost:8501` in your browser.
