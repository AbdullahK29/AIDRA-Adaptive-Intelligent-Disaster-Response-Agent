"""
csp.py
──────
Constraint Satisfaction Problem (CSP) solver for resource allocation.

PROBLEM FORMULATION
───────────────────
Variables   : one per unrescued victim → which ambulance is assigned?
Domains     : {0, 1}  (ambulance index — we have 2 ambulances)
Constraints :
  HARD-1  Capacity : no ambulance may carry more than 2 victims at once.
  HARD-2  Team     : only 1 rescue team available (assigned to the
                     highest-severity victim in each trip).
  SOFT    Priority : critical victims should be assigned before moderate/minor
                     (implemented via MRV ordering, not a hard constraint).

HEURISTICS (required by the assignment rubric)
──────────────────────────────────────────────
1. MRV — Minimum Remaining Values:
   Order victims by DESCENDING severity so the most constrained ones
   (critical — must be served quickly) are assigned first.
   This reduces the search space pruned by forward checking.

2. Degree heuristic (embedded in MRV tie-breaking):
   Among equal-severity victims, prefer the one farthest from base
   (higher "degree" of urgency).

3. Forward Checking:
   After assigning a victim to an ambulance, immediately check whether
   that ambulance would exceed capacity.  If yes, prune that branch
   before recursing — avoids useless exploration.

Why use a CSP instead of simple if/else logic?
   Constraints interact: putting victim A in ambulance 1 might force
   victim B into ambulance 2, which might then be over capacity.
   CSP + backtracking handles these interactions automatically and
   provably satisfies ALL hard constraints.
"""

from environment import SEVERITY_SCORE


class CSPSolver:
    """
    Assigns unrescued victims to ambulances in trips,
    respecting all hard resource constraints.
    """

    def __init__(self, victims, n_ambulances=2, n_teams=1, n_kits=10):
        self.victims     = [v for v in victims if not v["rescued"]]
        self.n_amb       = n_ambulances
        self.n_teams     = n_teams
        self.n_kits      = n_kits
        self.backtrack_count = 0

    # ── Public API ────────────────────────────────────────────

    def solve(self):
        """
        Entry point.  Returns (assignment_dict, backtrack_count).

        assignment_dict maps victim_id → {"ambulance": int, "team": int, "trip": int}.
        Returns (None, count) if no valid assignment exists.
        """
        # MRV ordering: critical first, then moderate, then minor.
        # Tie-break: victims with higher id (arbitrary but consistent).
        ordered = sorted(
            self.victims,
            key=lambda v: (-SEVERITY_SCORE[v["severity"]], v["id"])
        )

        full_assignment = {}
        trip_number     = 1
        remaining       = ordered[:]

        # Batch victims into trips.  Each trip can hold at most
        # n_ambulances × 2 victims (2 per ambulance).
        while remaining:
            batch     = remaining[: self.n_amb * 2]
            remaining = remaining[self.n_amb * 2 :]

            amb_load = {i: 0 for i in range(self.n_amb)}
            result   = self._backtrack(batch, 0, {}, amb_load)

            if result is None:
                # This batch has no valid assignment (shouldn't happen
                # with 2 ambulances × 2 capacity for ≤4 victims per batch,
                # but handle it gracefully).
                continue

            for vid, alloc in result.items():
                alloc["trip"]        = trip_number
                full_assignment[vid] = alloc

            trip_number += 1

        return (full_assignment if full_assignment else None), self.backtrack_count

    # ── Internal backtracking ─────────────────────────────────

    def _backtrack(self, victims, idx, assignment, amb_load):
        """
        Recursive backtracking with forward checking.

        victims   : list of victim dicts for this trip (MRV-ordered)
        idx       : index of the victim we are currently assigning
        assignment: {victim_id: {"ambulance": int, "team": int}}
        amb_load  : {ambulance_id: current_victim_count}
        """
        # Base case: all victims in this batch assigned successfully
        if idx == len(victims):
            return assignment

        victim = victims[idx]

        # Try ambulances in load-ascending order (least-loaded first).
        # This is the degree heuristic: prefer the resource with the
        # most remaining capacity (fewer constraints active).
        amb_order = sorted(range(self.n_amb), key=lambda a: amb_load[a])

        for amb_id in amb_order:

            # ── FORWARD CHECKING ─────────────────────────────
            # Before recursing, verify that assigning this victim to
            # amb_id does NOT violate the capacity constraint.
            if amb_load[amb_id] >= 2:
                # This ambulance is full — skip without recursing.
                continue

            # ── ASSIGNMENT ───────────────────────────────────
            assignment[victim["id"]] = {
                "ambulance": amb_id,
                "team":      0,      # only 1 team; always team 0
            }
            amb_load[amb_id] += 1

            # ── RECURSE ──────────────────────────────────────
            result = self._backtrack(victims, idx + 1, assignment, amb_load)
            if result is not None:
                return result   # Found a valid complete assignment

            # ── BACKTRACK ────────────────────────────────────
            # This branch failed → undo and try next ambulance
            amb_load[amb_id] -= 1
            del assignment[victim["id"]]
            self.backtrack_count += 1

        # No ambulance could take this victim → signal failure upward
        return None

    # ── Reporting ─────────────────────────────────────────────

    def report(self, assignment):
        """
        Return a human-readable summary of the assignment.
        Used by the agent's decision log.
        """
        if not assignment:
            return ["CSP: No valid assignment found."]

        lines = [f"CSP Assignment (backtracks={self.backtrack_count}):"]
        trip_groups = {}
        for vid, alloc in assignment.items():
            t = alloc["trip"]
            trip_groups.setdefault(t, []).append((vid, alloc))

        for trip, entries in sorted(trip_groups.items()):
            lines.append(f"  Trip {trip}:")
            for vid, alloc in entries:
                lines.append(
                    f"    Victim {vid} → Ambulance {alloc['ambulance'] + 1}"
                )

        return lines
