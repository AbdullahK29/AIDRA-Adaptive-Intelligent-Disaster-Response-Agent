"""
search.py
─────────
Classic search algorithms used for route planning on the grid map.

Every function returns (path, nodes_expanded):
  path            – list of (row, col) from start → goal, or None if unreachable
  nodes_expanded  – integer; how many nodes the algorithm examined

This return signature is identical across all algorithms so the agent can
swap them in/out and compare them on exactly the same metric.
"""

import heapq
from collections import deque
from environment import get_neighbors, manhattan, GRID_SIZE


# ─────────────────────────────────────────────────────────────
#  SHARED UTILITY
# ─────────────────────────────────────────────────────────────

def _reconstruct(came_from, start, goal):
    """Walk the came_from dict backwards to build a start→goal path."""
    path, node = [], goal
    while node is not None:
        path.append(node)
        node = came_from[node]
    path.reverse()
    return path


# ─────────────────────────────────────────────────────────────
#  BFS  —  Breadth-First Search
# ─────────────────────────────────────────────────────────────

def bfs(grid, start, goal, allow_risk=True):
    """
    Explores nodes layer by layer (distance 1, then 2, then 3…).
    Guarantees the SHORTEST PATH in terms of number of steps.
    Does NOT consider risk cost — every passable cell costs the same.

    Strengths : always finds the fewest-step path.
    Weaknesses: expands many nodes on large maps; ignores risk zones.
    """
    frontier  = deque([start])
    came_from = {start: None}
    expanded  = 0

    while frontier:
        node = frontier.popleft()
        expanded += 1

        if node == goal:
            return _reconstruct(came_from, start, goal), expanded

        for nb in get_neighbors(grid, node, allow_risk):
            if nb not in came_from:
                came_from[nb] = node
                frontier.append(nb)

    return None, expanded   # goal unreachable


# ─────────────────────────────────────────────────────────────
#  DFS  —  Depth-First Search
# ─────────────────────────────────────────────────────────────

def dfs(grid, start, goal, allow_risk=True):
    """
    Dives as deep as possible before backtracking.
    Will find SOME path, but it is often long and convoluted.

    Strengths : very memory-efficient (only keeps one path in stack).
    Weaknesses: path quality is poor; not guaranteed optimal.
    The assignment uses DFS as a negative baseline to contrast with A*.
    """
    stack    = [(start, [start])]
    visited  = {start}
    expanded = 0

    while stack:
        node, path = stack.pop()
        expanded += 1

        if node == goal:
            return path, expanded

        for nb in get_neighbors(grid, node, allow_risk):
            if nb not in visited:
                visited.add(nb)
                stack.append((nb, path + [nb]))

    return None, expanded


# ─────────────────────────────────────────────────────────────
#  GREEDY BEST-FIRST SEARCH
# ─────────────────────────────────────────────────────────────

def greedy(grid, start, goal, allow_risk=True):
    """
    Always expands the node that LOOKS closest to the goal
    (using Manhattan distance as the heuristic).

    Strengths : fast — usually reaches the goal with few expansions.
    Weaknesses: not optimal; can get trapped or take suboptimal detours
                because it ignores the actual cost already paid.
    """
    frontier  = [(manhattan(start, goal), start)]
    came_from = {start: None}
    expanded  = 0

    while frontier:
        _, node = heapq.heappop(frontier)
        expanded += 1

        if node == goal:
            return _reconstruct(came_from, start, goal), expanded

        for nb in get_neighbors(grid, node, allow_risk):
            if nb not in came_from:
                came_from[nb] = node
                heapq.heappush(frontier, (manhattan(nb, goal), nb))

    return None, expanded


# ─────────────────────────────────────────────────────────────
#  A*  —  A-Star Search
# ─────────────────────────────────────────────────────────────

def astar(grid, start, goal, allow_risk=True):
    """
    Combines actual path cost g(n) with heuristic estimate h(n).
    f(n) = g(n) + h(n)

    Move costs:
      normal cell   → 1
      high-risk     → 3  (traversable but expensive → naturally avoids hazards)

    Guaranteed optimal when heuristic is admissible (never over-estimates).
    Manhattan distance is admissible on a grid.

    Strengths : optimal AND efficient — expands far fewer nodes than BFS.
    Weaknesses: more complex to implement; needs a good heuristic.
    This is the RECOMMENDED algorithm for the AIDRA system.
    """
    g_score   = {start: 0}
    f_score   = {start: manhattan(start, goal)}
    frontier  = [(f_score[start], start)]
    came_from = {start: None}
    expanded  = 0

    while frontier:
        _, node = heapq.heappop(frontier)
        expanded += 1

        if node == goal:
            return _reconstruct(came_from, start, goal), expanded

        for nb in get_neighbors(grid, node, allow_risk):
            cell      = grid[nb[0]][nb[1]]
            move_cost = 3 if cell == 2 else 1       # HIGH_RISK = 2
            tentative = g_score[node] + move_cost

            if nb not in g_score or tentative < g_score[nb]:
                g_score[nb]   = tentative
                f_score[nb]   = tentative + manhattan(nb, goal)
                came_from[nb] = node
                heapq.heappush(frontier, (f_score[nb], nb))

    return None, expanded
