"""
fuzzy.py
────────
Uncertainty handling via Fuzzy Logic (Mamdani-style rule base).

WHY FUZZY LOGIC?
────────────────
Real disaster environments are NOT binary:
  "Is this road blocked?" → Not yes/no, but "40% likely blocked"
  "Is this area dangerous?" → Not safe/unsafe, but "moderately risky"

Fuzzy Logic handles this by using MEMBERSHIP FUNCTIONS that map a crisp
value (0.0–1.0) to a degree of truth for linguistic labels like LOW, MEDIUM,
HIGH.  Then IF-THEN RULES combine these degrees to produce a crisp output
via DEFUZZIFICATION (weighted average of rule outputs).

This is why the instructor prefers Fuzzy over simple thresholds — it
produces human-interpretable, graded decisions.

FUNCTIONS
─────────
fuzzy_route_risk(blockage_prob, hazard_spread, road_reliability)
  → risk_level ∈ [0, 1]
  Used by the agent to decide between fast (risky) and safe routes.
  If fuzzy risk > 0.6 → override towards the safe route.

fuzzy_priority_score(severity, distance, survival_prob)
  → priority ∈ [0, 1]
  Combines victim urgency factors into one score that ranks who to rescue first.
  This score DIRECTLY DETERMINES rescue order (feeds into agent decisions).
"""

from environment import SEVERITY_SCORE, GRID_SIZE


# ─────────────────────────────────────────────────────────────
#  MEMBERSHIP FUNCTIONS  (triangular / trapezoidal)
# ─────────────────────────────────────────────────────────────

def _mf_low(x: float) -> float:
    """Membership in LOW: peaks at 0, zero by 0.4."""
    return max(0.0, 1.0 - x / 0.4)

def _mf_medium(x: float) -> float:
    """Membership in MEDIUM: peaks at 0.5, zero at 0.2 and 0.8."""
    return max(0.0, 1.0 - abs(x - 0.5) / 0.3)

def _mf_high(x: float) -> float:
    """Membership in HIGH: zero below 0.6, peaks at 1.0."""
    return max(0.0, (x - 0.6) / 0.4)


# ─────────────────────────────────────────────────────────────
#  FUZZY ROUTE RISK
# ─────────────────────────────────────────────────────────────

def fuzzy_route_risk(blockage_prob: float,
                     hazard_spread: float,
                     road_reliability: float) -> float:
    """
    Estimate the overall risk level of a route.

    Inputs (all normalised to [0, 1]):
      blockage_prob     — probability that the route has a road block
      hazard_spread     — how widely fire/aftershock hazards have spread
      road_reliability  — how trustworthy current road status data is
                          (low reliability → higher uncertainty → higher risk)

    Output:
      risk_level ∈ [0, 1]
        0.0 → completely safe
        1.0 → extremely dangerous

    Rule base (Mamdani):
      IF blockage HIGH   AND hazard HIGH   → risk = 1.00  (avoid at all costs)
      IF blockage HIGH   AND hazard MEDIUM → risk = 0.80
      IF blockage MEDIUM AND hazard HIGH   → risk = 0.75
      IF blockage MEDIUM AND hazard MEDIUM → risk = 0.50
      IF blockage LOW    AND hazard LOW    → risk = 0.10  (nearly safe)
      IF reliability LOW AND blockage HIGH → risk = 0.90  (can't trust the map)
      IF reliability HIGH AND blockage LOW → risk = 0.15  (map is accurate & safe)

    Defuzzification: weighted average (centroid) of rule outputs.
    """
    bp_low  = _mf_low(blockage_prob);   bp_med = _mf_medium(blockage_prob)
    bp_high = _mf_high(blockage_prob)

    hs_low  = _mf_low(hazard_spread);   hs_med = _mf_medium(hazard_spread)
    hs_high = _mf_high(hazard_spread)

    rr_low  = _mf_low(road_reliability)
    rr_high = _mf_high(road_reliability)

    rules = [
        (min(bp_high, hs_high), 1.00),
        (min(bp_high, hs_med),  0.80),
        (min(bp_med,  hs_high), 0.75),
        (min(bp_med,  hs_med),  0.50),
        (min(bp_low,  hs_low),  0.10),
        (min(rr_low,  bp_high), 0.90),
        (min(rr_high, bp_low),  0.15),
    ]

    numerator   = sum(strength * output for strength, output in rules)
    denominator = sum(strength for strength, _ in rules)

    risk = (numerator / denominator) if denominator > 0 else 0.5
    return round(risk, 3)


# ─────────────────────────────────────────────────────────────
#  FUZZY PRIORITY SCORE
# ─────────────────────────────────────────────────────────────

def fuzzy_priority_score(severity: str,
                         distance: int,
                         survival_prob: float) -> float:
    """
    Compute a rescue priority score for one victim.

    Inputs:
      severity      : "critical" / "moderate" / "minor"
      distance      : Manhattan distance from base to victim
      survival_prob : ML-predicted survival probability ∈ [0, 1]

    Output:
      priority ∈ [0, 1]  — higher = rescue sooner

    Weighting rationale (tunable):
      50% severity      — medical urgency is the dominant factor
      30% survival_prob — ML prediction directly influences priority
      20% proximity     — closer victims are slightly preferred
                          (faster rescue → better outcome)

    The 30% weight on survival_prob is WHY the ML module must feed into
    priority — without it, the score ignores predicted outcome entirely.
    """
    max_dist   = GRID_SIZE * 2          # normalisation constant
    sev_score  = SEVERITY_SCORE[severity] / 3.0
    dist_score = max(0.0, 1.0 - distance / max_dist)

    priority = (0.5 * sev_score) + (0.3 * survival_prob) + (0.2 * dist_score)
    return round(priority, 3)


# ─────────────────────────────────────────────────────────────
#  ENVIRONMENTAL RISK SNAPSHOT  (convenience wrapper)
# ─────────────────────────────────────────────────────────────

def assess_environment(blockage_prob=0.3,
                       hazard_spread=0.4,
                       road_reliability=0.6) -> dict:
    """
    Return a full environmental risk assessment dict.
    Default values represent a mid-disaster scenario.
    """
    risk = fuzzy_route_risk(blockage_prob, hazard_spread, road_reliability)
    level = "HIGH" if risk > 0.65 else ("MEDIUM" if risk > 0.35 else "LOW")
    return {
        "risk_score":  risk,
        "risk_level":  level,
        "inputs": {
            "blockage_prob":    blockage_prob,
            "hazard_spread":    hazard_spread,
            "road_reliability": road_reliability,
        },
    }
