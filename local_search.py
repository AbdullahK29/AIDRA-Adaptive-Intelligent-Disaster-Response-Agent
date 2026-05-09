"""
local_search.py
───────────────
Local search optimisation algorithms.

Unlike BFS/DFS/A* which BUILD a path from scratch, local search algorithms
START with an already-found path (seeded by A*) and try to IMPROVE it by
making small modifications called "perturbations".

Why this matters for the assignment:
  The CCP requires at least one local search method as an alternative
  optimisation strategy to compare against classical search.

Both functions return (path, iterations_used) — same contract as search.py.
"""

import random
import math
from search import astar
from environment import get_neighbors, path_cost


# ─────────────────────────────────────────────────────────────
#  PERTURBATION OPERATOR  (shared by both algorithms)
# ─────────────────────────────────────────────────────────────

def _perturb(grid, path):
    """
    Create a neighbour of *path* by applying ONE of two moves:

    Move A — Detour (insert):
      Pick a random interior node and insert one of its grid-neighbours
      as an extra step.  This can open an escape route around a costly cell.

    Move B — Shortcut (remove):
      If two non-adjacent path nodes are themselves grid-neighbours,
      remove the intermediate node to shorten the path.

    The resulting path is always valid (every consecutive pair of cells
    is a grid-neighbour) because we only insert/remove reachable cells.
    """
    if len(path) < 3:
        return path[:]

    move = random.choice(["detour", "shortcut"])

    if move == "detour":
        # Insert a neighbour of a random interior node
        idx       = random.randint(1, len(path) - 2)
        nbs       = get_neighbors(grid, path[idx], allow_risk=True)
        extra_nbs = [n for n in nbs if n not in (path[idx - 1], path[idx + 1])]
        if not extra_nbs:
            return path[:]
        detour   = random.choice(extra_nbs)
        new_path = path[:idx + 1] + [detour] + path[idx + 1:]
        return new_path

    else:  # shortcut
        # Try to remove a node if its predecessor can reach its successor directly
        idx  = random.randint(1, len(path) - 2)
        prev = path[idx - 1]
        nxt  = path[idx + 1]
        if nxt in get_neighbors(grid, prev, allow_risk=True):
            return path[:idx] + path[idx + 1:]
        return path[:]


# ─────────────────────────────────────────────────────────────
#  HILL CLIMBING
# ─────────────────────────────────────────────────────────────

def hill_climbing(grid, start, goal, iterations=300):
    """
    Greedy local search.

    Algorithm:
      1. Start with an A* path as the initial solution.
      2. Repeat *iterations* times:
           a. Create a perturbed neighbour path.
           b. If it costs LESS → accept it (move "uphill" in quality).
           c. If not → discard it (strictly greedy, no worse moves accepted).

    Weakness (Local Optima):
      Hill Climbing can get stuck — once no immediate neighbour is better,
      it stops even if a globally better path exists elsewhere.
      Simulated Annealing fixes this by sometimes accepting worse moves.

    Returns (best_path, iterations_run).
    """
    current_path, _ = astar(grid, start, goal, allow_risk=True)
    if current_path is None:
        return None, 0

    current_cost = path_cost(grid, current_path)
    best_path    = current_path[:]
    best_cost    = current_cost

    for i in range(iterations):
        candidate      = _perturb(grid, current_path)
        candidate_cost = path_cost(grid, candidate)

        # Accept only improvements (greedy / "always uphill")
        if candidate_cost < current_cost:
            current_path = candidate
            current_cost = candidate_cost

        # Track global best (in case we ever step away temporarily)
        if current_cost < best_cost:
            best_path = current_path[:]
            best_cost = current_cost

    return best_path, iterations


# ─────────────────────────────────────────────────────────────
#  SIMULATED ANNEALING
# ─────────────────────────────────────────────────────────────

def simulated_annealing(grid, start, goal,
                        T=15.0, cooling=0.97, iterations=400):
    """
    Probabilistic local search inspired by the annealing process in metallurgy.

    Algorithm:
      1. Start with an A* path.
      2. Repeat *iterations* times:
           a. Create a perturbed neighbour path.
           b. If it is better → always accept it.
           c. If it is WORSE  → accept it with probability exp(-Δ / T).
              This probability decreases as T cools down.
      3. Return the best path ever found.

    Why the random acceptance of worse moves?
      It lets the algorithm ESCAPE LOCAL OPTIMA that trap Hill Climbing.
      Early on (high T) the algorithm explores broadly.
      Late (low T → near 0) it converges and only accepts improvements.

    Parameters:
      T        – initial temperature (higher = more random exploration)
      cooling  – multiplier applied to T each iteration (< 1 → cools down)
      iterations – total moves attempted

    Returns (best_path, iterations_run).
    """
    current_path, _ = astar(grid, start, goal, allow_risk=True)
    if current_path is None:
        return None, 0

    current_cost = path_cost(grid, current_path)
    best_path    = current_path[:]
    best_cost    = current_cost

    for i in range(iterations):
        candidate      = _perturb(grid, current_path)
        candidate_cost = path_cost(grid, candidate)
        delta          = candidate_cost - current_cost

        # Accept if better, OR with decreasing probability if worse
        if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-6)):
            current_path = candidate
            current_cost = candidate_cost

        # Always track the global best
        if current_cost < best_cost:
            best_path = current_path[:]
            best_cost = current_cost

        T *= cooling   # Cool down

    return best_path, iterations
