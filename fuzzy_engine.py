"""
Fuzzy Inference System (FIS) for Loan Approval
Implemented from scratch using NumPy (No scikit-fuzzy)
======================================================
Implements:
1. Custom Triangular & Trapezoidal Membership Functions
2. Mamdani Rule Base Evaluation (MIN for AND, MAX for OR)
3. Centroid (Center of Gravity) Defuzzification
"""

import numpy as np
from typing import Dict, Tuple, List, Any


# =====================================================================
# 1. MEMBERSHIP FUNCTIONS (Vectorized & Scalar safe)
# =====================================================================

def triangular_mf(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Triangular membership function:
    trimf(x; a, b, c) = max(0, min((x-a)/(b-a), (c-x)/(c-b)))
    """
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    
    # Left slope: a <= x <= b
    if b > a:
        left_mask = (a <= x) & (x <= b)
        y[left_mask] = (x[left_mask] - a) / (b - a)
    elif b == a:
        y[x == a] = 1.0

    # Right slope: b < x <= c
    if c > b:
        right_mask = (b < x) & (x <= c)
        y[right_mask] = (c - x[right_mask]) / (c - b)
    elif c == b:
        y[x == b] = 1.0

    # Clamp bounds: exactly 1 at peak, 0 outside
    y[x == b] = 1.0
    return np.clip(y, 0.0, 1.0)


def trapezoidal_mf(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """
    Trapezoidal membership function:
    trapmf(x; a, b, c, d) = max(0, min((x-a)/(b-a), 1, (d-x)/(d-c)))
    """
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)

    # Left shoulder / slope
    if b > a:
        left_mask = (a <= x) & (x < b)
        y[left_mask] = (x[left_mask] - a) / (b - a)
    elif b == a:
        y[x <= a] = 1.0

    # Plateau: b <= x <= c
    plateau_mask = (b <= x) & (x <= c)
    y[plateau_mask] = 1.0

    # Right shoulder / slope
    if d > c:
        right_mask = (c < x) & (x <= d)
        y[right_mask] = (d - x[right_mask]) / (d - c)
    elif d == c:
        y[x >= c] = 1.0

    # Left boundary flat shoulder if a is min bound
    if a == b:
        y[x < a] = 1.0
    if c == d:
        y[x > d] = 1.0

    return np.clip(y, 0.0, 1.0)


# =====================================================================
# 2. FUZZY SYSTEM SPECIFICATIONS & LINGUISTIC VARIABLES
# =====================================================================

class LoanFuzzyInferenceSystem:
    def __init__(self):
        # Universes of Discourse
        self.cibil_universe = np.linspace(300, 900, 601)
        self.income_universe = np.linspace(15000, 250000, 1000)
        self.approval_universe = np.linspace(0, 100, 1001)

        # CIBIL Score Membership Definitions
        # Poor: Left-shoulder trapezoid [300, 300, 520, 640]
        # Average: Triangular [580, 680, 770]
        # Good: Right-shoulder trapezoid [720, 800, 900, 900]
        self.cibil_params = {
            "Poor": ("trap", (300, 300, 520, 640)),
            "Average": ("tri", (580, 680, 770)),
            "Good": ("trap", (720, 800, 900, 900))
        }

        # Monthly Income Membership Definitions (in INR ₹)
        # Low: Left-shoulder trapezoid [15k, 15k, 35k, 60k]
        # Medium: Triangular [45k, 85k, 135k]
        # High: Right-shoulder trapezoid [110k, 160k, 250k, 250k]
        self.income_params = {
            "Low": ("trap", (15000, 15000, 35000, 60000)),
            "Medium": ("tri", (45000, 85000, 135000)),
            "High": ("trap", (110000, 160000, 250000, 250000))
        }

        # Loan Approval Chance Membership Definitions (0 - 100%)
        # Reject: Left-shoulder trapezoid [0, 0, 20, 45]
        # Hold: Triangular [35, 55, 75]
        # Approve: Right-shoulder trapezoid [65, 82, 100, 100]
        self.approval_params = {
            "Reject": ("trap", (0, 0, 20, 45)),
            "Hold": ("tri", (35, 55, 75)),
            "Approve": ("trap", (65, 82, 100, 100))
        }

        # Precompute output membership curves over approval universe
        self.output_mfs = {
            name: self._compute_mf(self.approval_universe, *params)
            for name, params in self.approval_params.items()
        }

        # Mamdani Rule Base (3x3 = 9 exhaustive rules)
        # Structure: (Rule ID, CIBIL term, Income term, Consequent term, Description)
        self.rules = [
            (1, "Poor", "Low", "Reject", "IF CIBIL is Poor AND Income is Low THEN Approval is Reject"),
            (2, "Poor", "Medium", "Reject", "IF CIBIL is Poor AND Income is Medium THEN Approval is Reject"),
            (3, "Poor", "High", "Hold", "IF CIBIL is Poor AND Income is High THEN Approval is Hold"),
            (4, "Average", "Low", "Reject", "IF CIBIL is Average AND Income is Low THEN Approval is Reject"),
            (5, "Average", "Medium", "Hold", "IF CIBIL is Average AND Income is Medium THEN Approval is Hold"),
            (6, "Average", "High", "Approve", "IF CIBIL is Average AND Income is High THEN Approval is Approve"),
            (7, "Good", "Low", "Hold", "IF CIBIL is Good AND Income is Low THEN Approval is Hold"),
            (8, "Good", "Medium", "Approve", "IF CIBIL is Good AND Income is Medium THEN Approval is Approve"),
            (9, "Good", "High", "Approve", "IF CIBIL is Good AND Income is High THEN Approval is Approve"),
        ]

    def _compute_mf(self, x: np.ndarray, mf_type: str, params: tuple) -> np.ndarray:
        """Helper to evaluate triangular or trapezoidal membership function."""
        if mf_type == "tri":
            return triangular_mf(x, *params)
        elif mf_type == "trap":
            return trapezoidal_mf(x, *params)
        raise ValueError(f"Unknown MF type: {mf_type}")

    # =================================================================
    # 3. FUZZIFICATION
    # =================================================================
    def fuzzify(self, cibil: float, income: float) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Calculates the degree of membership for each linguistic term
        given crisp inputs for CIBIL and Monthly Income.
        """
        cibil_val = np.array([cibil])
        income_val = np.array([income])

        cibil_deg = {
            term: float(self._compute_mf(cibil_val, *params)[0])
            for term, params in self.cibil_params.items()
        }

        income_deg = {
            term: float(self._compute_mf(income_val, *params)[0])
            for term, params in self.income_params.items()
        }

        return cibil_deg, income_deg

    # =================================================================
    # 4. RULE EVALUATION & IMPLICATION (MAMDANI)
    # =================================================================
    def evaluate_rules(self, cibil_deg: Dict[str, float], income_deg: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Evaluates the Mamdani rule base:
        1. AND operator = MIN(mu_cibil, mu_income)
        2. Implication = Truncate (MIN) consequent membership function by rule fire strength
        """
        rule_evaluations = []

        for rule_id, cibil_term, income_term, consequent_term, desc in self.rules:
            # T-Norm (Mamdani AND = MIN)
            premise_weight = min(cibil_deg[cibil_term], income_deg[income_term])
            
            # Mamdani implication (MIN truncation)
            implied_curve = np.minimum(premise_weight, self.output_mfs[consequent_term])

            rule_evaluations.append({
                "id": rule_id,
                "cibil_term": cibil_term,
                "income_term": income_term,
                "consequent": consequent_term,
                "description": desc,
                "weight": premise_weight,
                "implied_curve": implied_curve
            })

        return rule_evaluations

    # =================================================================
    # 5. AGGREGATION & DEFUZZIFICATION (CENTROID METHOD)
    # =================================================================
    def defuzzify_centroid(self, aggregated_curve: np.ndarray) -> float:
        """
        Calculates crisp output using Centroid (Center of Gravity) defuzzification:
        
                  ∫ y * mu(y) dy      ∑ (y_i * mu(y_i))
        Centroid = --------------  ≈  -----------------
                   ∫ mu(y) dy          ∑ mu(y_i)
        """
        y = self.approval_universe
        numerator = np.sum(y * aggregated_curve)
        denominator = np.sum(aggregated_curve)

        if denominator == 0 or np.isnan(denominator):
            return 50.0  # Fallback neutral point if no rules fire
            
        return float(numerator / denominator)

    # =================================================================
    # 6. COMPLETE END-TO-END INFERENCE PIPELINE
    # =================================================================
    def infer(self, cibil: float, income: float) -> Dict[str, Any]:
        """
        Runs complete Fuzzy Inference:
        Fuzzify -> Mamdani Rule Evaluation -> MAX Aggregation -> Centroid Defuzzification
        """
        # Step 1: Fuzzification
        cibil_deg, income_deg = self.fuzzify(cibil, income)

        # Step 2: Mamdani Rule Evaluation (MIN operator & MIN implication)
        rule_results = self.evaluate_rules(cibil_deg, income_deg)

        # Step 3: Fuzzy Aggregation (S-Norm: MAX operator across all active implied curves)
        aggregated_curve = np.zeros_like(self.approval_universe)
        for r in rule_results:
            if r["weight"] > 0:
                aggregated_curve = np.maximum(aggregated_curve, r["implied_curve"])

        # Also aggregate by individual linguistic terms for plotting and transparency
        term_curves = {
            "Reject": np.zeros_like(self.approval_universe),
            "Hold": np.zeros_like(self.approval_universe),
            "Approve": np.zeros_like(self.approval_universe),
        }
        for r in rule_results:
            conseq = r["consequent"]
            term_curves[conseq] = np.maximum(term_curves[conseq], r["implied_curve"])

        # Step 4: Centroid Defuzzification
        crisp_score = self.defuzzify_centroid(aggregated_curve)

        # Determine categorical decision and risk grade
        if crisp_score >= 68.0:
            decision = "APPROVED"
            decision_code = "APPROVE"
            decision_color = "#00E676"  # Vibrant Emerald Green
            risk_tier = "Tier 1: Prime / Low Credit Risk"
            risk_badge = "LOW RISK"
            recommendation = "Applicant qualifies for standard and preferential interest rates with expedited disbursement."
        elif crisp_score >= 42.0:
            decision = "ON HOLD / MANUAL REVIEW"
            decision_code = "HOLD"
            decision_color = "#FF9100"  # Vibrant Amber/Orange
            risk_tier = "Tier 2: Moderate Credit Risk"
            risk_badge = "MODERATE RISK"
            recommendation = "Requires manual underwriting, income verification documents, and co-applicant/collateral assessment."
        else:
            decision = "REJECTED"
            decision_code = "REJECT"
            decision_color = "#FF1744"  # Vibrant Crimson Red
            risk_tier = "Tier 3: Subprime / High Default Risk"
            risk_badge = "HIGH RISK"
            recommendation = "Application does not meet the minimum credit-to-income threshold. Recommend credit rebuilding."

        # Active rules for transparency breakdown
        active_rules = [r for r in rule_results if r["weight"] > 0]
        active_rules.sort(key=lambda x: x["weight"], reverse=True)

        return {
            "cibil_input": cibil,
            "income_input": income,
            "cibil_degrees": cibil_deg,
            "income_degrees": income_deg,
            "rule_results": rule_results,
            "active_rules": active_rules,
            "approval_universe": self.approval_universe,
            "aggregated_curve": aggregated_curve,
            "term_curves": term_curves,
            "crisp_score": round(crisp_score, 2),
            "decision": decision,
            "decision_code": decision_code,
            "decision_color": decision_color,
            "risk_tier": risk_tier,
            "risk_badge": risk_badge,
            "recommendation": recommendation,
        }
