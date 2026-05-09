"""
environment.py
──────────────
World model: grid constants, initial map layout, victim definitions,
resource definitions, and helper utilities shared across all modules.
"""

# ── Grid dimensions ───────────────────────────────────────────
GRID_SIZE = 10

# ── Cell type constants ───────────────────────────────────────
NORMAL    = 0   # passable road
BLOCKED   = 1   # rubble / impassable
HIGH_RISK = 2   # fire / structural collapse zone
MEDICAL   = 3   # safe medical centre (destination)
BASE      = 4   # rescue base (origin)

# ── Colours used by the GUI ───────────────────────────────────
CELL_COLORS = {
    NORMAL:    "#2d4a2d",
    BLOCKED:   "#1a0a0a",
    HIGH_RISK: "#5a1a00",
    MEDICAL:   "#003366",
    BASE:      "#4a3800",
}

SEVERITY_COLORS = {
    "critical": "#ff2222",
    "moderate": "#ff9900",
    "minor":    "#44ff44",
}

# ── Map layout ────────────────────────────────────────────────
# 0=normal 1=blocked 2=high-risk
INITIAL_GRID = [
    [0, 0, 0, 2, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 2, 0, 1, 0, 0, 0, 0],
    [0, 1, 0, 2, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 0],
    [0, 0, 1, 1, 0, 0, 0, 0, 2, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 2, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 2, 2, 0, 0, 0, 0, 0, 0, 0],
    [0, 2, 2, 0, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]

# ── Fixed locations ───────────────────────────────────────────
BASE_POS        = (9, 0)
MEDICAL_CENTERS = [(0, 9), (5, 9)]

# ── Initial victims ───────────────────────────────────────────
INITIAL_VICTIMS = [
    {"id": 1, "pos": (1, 3), "severity": "critical", "rescued": False,
     "assigned": None, "survival_prob": None},
    {"id": 2, "pos": (4, 7), "severity": "critical", "rescued": False,
     "assigned": None, "survival_prob": None},
    {"id": 3, "pos": (6, 2), "severity": "moderate", "rescued": False,
     "assigned": None, "survival_prob": None},
    {"id": 4, "pos": (3, 8), "severity": "moderate", "rescued": False,
     "assigned": None, "survival_prob": None},
    {"id": 5, "pos": (8, 1), "severity": "minor",    "rescued": False,
     "assigned": None, "survival_prob": None},
]

# ── Resource pool ─────────────────────────────────────────────
INITIAL_RESOURCES = {"ambulances": 2, "teams": 1, "kits": 10}

# ── Severity numeric mapping ──────────────────────────────────
SEVERITY_SCORE = {"critical": 3, "moderate": 2, "minor": 1}


# ── Utilities ─────────────────────────────────────────────────

def fresh_grid():
    """Return a deep copy of the initial grid so mutations don't bleed across runs."""
    return [row[:] for row in INITIAL_GRID]


def fresh_victims():
    """Return deep copies of all victim dicts."""
    return [dict(v) for v in INITIAL_VICTIMS]


def manhattan(a, b):
    """Manhattan distance between two (row, col) positions."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def get_neighbors(grid, pos, allow_risk=True):
    """
    Return all traversable neighbours of *pos* on *grid*.

    allow_risk=False → HIGH_RISK cells are treated as walls.
    BLOCKED cells are always walls.
    """
    r, c = pos
    result = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
            cell = grid[nr][nc]
            if cell == BLOCKED:
                continue
            if cell == HIGH_RISK and not allow_risk:
                continue
            result.append((nr, nc))
    return result


def risk_steps_in_path(grid, path):
    """Count how many cells of *path* are HIGH_RISK zones."""
    return sum(1 for r, c in path if grid[r][c] == HIGH_RISK)


def path_cost(grid, path):
    """
    Weighted cost of a path:
      normal cell   → 1
      high-risk     → 5  (discouraged but traversable)
      blocked       → 100 (should never appear in a valid path)
    """
    total = 0
    for r, c in path:
        cell = grid[r][c]
        if cell == HIGH_RISK:
            total += 5
        elif cell == BLOCKED:
            total += 100
        else:
            total += 1
    return total
