"""
Safe-Interval Path Planning (SIPP) for one train over the shared reservation
table, with flatland 4.2.6 exact timing semantics:

- time t = post-step elapsed_steps at which a state/position is realized
- placement (off-map -> on-map) possible at t_place >= max(ed,1)+1; the
  placement step does not accumulate distance; cell exit follows
  ceil((1-f)/v) accumulation steps
- a halt inside a cell extends its occupancy; one halt event costs
  HALT penalty (the env charges collision_factor * v on MOVING->STOPPED)
- entering a goal node finishes the trip; goal cell occupied [t, t+1)
- swap conflicts are forbidden via the vacate_to table; same-step
  chain-following is allowed (headway=0) unless headway > 0

Costs are in "reward points" ~ steps: cost = arrival_time - earliest_possible
+ halt_weight * n_halts. We minimize, A* heuristic = exact nonstop
remaining travel time.
"""
import heapq
from bisect import bisect_right, insort
from fractions import Fraction
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np

from .netgraph import NetGraph

INF = float("inf")
UNREACH = 65535


class CellVisit(NamedTuple):
    node: int          # (r,c,d) encoded; -1 for off-map origin hold
    t_enter: int       # occupancy start (inclusive)
    t_exit: int        # occupancy end (exclusive) = entry time of next cell
    halt: bool         # whether a halt event happens inside this cell


class Reservations:
    """Shared reservation table over resources."""

    def __init__(self, horizon: int):
        self.horizon = horizon
        # res -> sorted list of (start, end, agent)
        self.intervals: Dict[int, List[Tuple[int, int, int]]] = {}
        # (t, res) -> (to_res, agent): an agent vacates res at t moving into to_res
        self.vacate_to: Dict[Tuple[int, int], Tuple[int, int]] = {}

    # --- mutation ------------------------------------------------------
    def add_path(self, agent: int, graph: NetGraph, visits: List[CellVisit]):
        prev_res = None
        for cv in visits:
            res = graph.resource_of_node(cv.node)
            insort(self.intervals.setdefault(res, []), (cv.t_enter, cv.t_exit, agent))
            if prev_res is not None:
                self.vacate_to[(cv.t_enter, prev_res)] = (res, agent)
            prev_res = res

    def remove_agent_path(self, agent: int, graph: NetGraph, visits: List[CellVisit]):
        prev_res = None
        for cv in visits:
            res = graph.resource_of_node(cv.node)
            lst = self.intervals.get(res)
            if lst is not None:
                try:
                    lst.remove((cv.t_enter, cv.t_exit, agent))
                except ValueError:
                    pass
            if prev_res is not None:
                cur = self.vacate_to.get((cv.t_enter, prev_res))
                if cur is not None and cur[1] == agent:
                    del self.vacate_to[(cv.t_enter, prev_res)]
            prev_res = res

    def block(self, agent: int, res: int, t_from: int, t_to: int):
        """Hard block (e.g. malfunctioning train sitting in a cell)."""
        insort(self.intervals.setdefault(res, []), (t_from, t_to, agent))

    # --- queries ---------------------------------------------------------
    def safe_intervals(self, res: int, t_from: int) -> List[Tuple[int, int]]:
        """Maximal free intervals of res intersecting [t_from, horizon)."""
        lst = self.intervals.get(res)
        if not lst:
            return [(t_from, self.horizon)] if t_from < self.horizon else []
        out = []
        lo = t_from
        # first interval that could overlap or follow t_from
        i = bisect_right(lst, (t_from, 1 << 60, 1 << 60)) - 1
        if i >= 0 and lst[i][1] > t_from:
            lo = lst[i][1]
        i += 1
        n = len(lst)
        while i < n:
            s, e, _ = lst[i]
            if e <= lo:
                i += 1
                continue
            if s > lo:
                out.append((lo, min(s, self.horizon)))
            lo = max(lo, e)
            if lo >= self.horizon:
                return out
            i += 1
        if lo < self.horizon:
            out.append((lo, self.horizon))
        return out

    def owner_at(self, res: int, t: int) -> Optional[int]:
        """Agent holding res at time t, or None if free."""
        lst = self.intervals.get(res)
        if not lst:
            return None
        i = bisect_right(lst, (t, 1 << 60, 1 << 60))
        if i > 0 and lst[i - 1][1] > t:
            return lst[i - 1][2]
        return None

    def free_until(self, res: int, t: int) -> int:
        """End of the free interval containing t (exclusive), or t if occupied."""
        lst = self.intervals.get(res)
        if not lst:
            return self.horizon
        i = bisect_right(lst, (t, self.horizon + 1, 1 << 30))
        # check interval starting before-or-at t
        if i > 0 and lst[i - 1][1] > t:
            return t  # occupied
        if i < len(lst):
            return lst[i][0]
        return self.horizon

    def conflicts_with(self, agent: int, graph: NetGraph, visits: List[CellVisit]) -> bool:
        for cv in visits:
            res = graph.resource_of_node(cv.node)
            for s, e, a in self.intervals.get(res, ()):
                if a == agent:
                    continue
                if s < cv.t_exit and cv.t_enter < e:
                    return True
        return False

    def conflicting_owners(self, agent: int, graph: NetGraph, visits: List[CellVisit]) -> set:
        """Agents whose reservations overlap the given visits."""
        owners = set()
        for cv in visits:
            res = graph.resource_of_node(cv.node)
            for s, e, a in self.intervals.get(res, ()):
                if a != agent and s < cv.t_exit and cv.t_enter < e:
                    owners.add(a)
        return owners

    def earliest_conflicting_owner(self, agent: int, graph: NetGraph, cv: CellVisit):
        """Owner of the earliest reservation overlapping the visit, or None."""
        res = graph.resource_of_node(cv.node)
        for s, e, a in self.intervals.get(res, ()):
            if a != agent and a >= 0 and s < cv.t_exit and cv.t_enter < e:
                return a
        return None


class SippPlanner:
    def __init__(self, graph: NetGraph, reservations: Reservations, horizon: int,
                 halt_weight: float = 250.0, headway: int = 0, max_expansions: int = 200000):
        self.g = graph
        self.res = reservations
        self.horizon = horizon
        self.halt_weight = halt_weight
        self.headway = headway
        self.max_expansions = max_expansions

    def plan(
        self,
        agent: int,
        start_node: int,
        goal_nodes: frozenset,
        speed: Fraction,
        earliest_start: int,
        start_frac: Fraction = Fraction(0),
        start_on_map: bool = False,
        start_halted: bool = False,
        latest_useful: Optional[int] = None,
        halt_weight: Optional[float] = None,
        stop_bonus: Optional[Dict[int, Tuple[int, int, int]]] = None,
        max_expansions: Optional[int] = None,
    ) -> Optional[List[CellVisit]]:
        """
        Plan a timed path. If start_on_map, the train is at start_node with
        accumulated fraction start_frac at time earliest_start (already
        occupying it). Otherwise it is off-map and may be placed at
        t >= earliest_start (occupancy of origin starts at placement time).

        stop_bonus: node -> (group, latest_arrival, earliest_departure) for
        this agent's unserved intermediate stops. Halting at such a node
        serves the stop: cost delta = -50 (skip penalty avoided)
        + 0.5*lateness + 0.5*earliness. The env records the serve whether we
        like it or not, so the delta is applied unconditionally on halts there.

        Returns list of CellVisit, or None if no path found.
        """
        g = self.g
        if halt_weight is None:
            halt_weight = self.halt_weight
        dist = g.dist_to_goals(goal_nodes)
        if dist[start_node] == UNREACH and start_node not in goal_nodes:
            return None
        p, q = speed.numerator, speed.denominator
        horizon = self.horizon
        hw = self.headway

        def h(node: int, frac_num: int) -> float:
            # exact nonstop time: ceil((cells*q - frac_num)/p)
            d = dist[node]
            if d == UNREACH:
                return INF
            need = d * q - frac_num
            if need <= 0:
                return 0.0
            return (need + p - 1) // p

        # search state: (node, frac_num, interval_start) — g-value = entry time t
        # cost = t + halt_weight * halts (we also track halts for tie-breaks)
        start_states = []
        if start_on_map:
            start_states.append((earliest_start, int(start_frac * q), start_node, start_halted))
        else:
            res0 = g.resource_of_node(start_node)
            for s, e in self.res.safe_intervals(res0, earliest_start):
                t0 = max(s + hw if s > 0 else s, earliest_start)
                if t0 < e:
                    start_states.append((t0, 0, start_node, False))

        if not start_states:
            return None

        # priority queue entries: (f, cost, t, node, frac_num, halts, parent_key)
        openq = []
        best: Dict[Tuple[int, int, int], float] = {}
        parents: Dict[Tuple[int, int, int], Tuple[Optional[Tuple], int, bool]] = {}

        free_halt_keys = set()  # states whose in-cell wait is free (already stopped)
        overstay_keys = set()   # start states: we PHYSICALLY occupy the cell, so
        # others' reservations of it are stale fiction — we may stay as long as
        # needed (otherwise queued trains behind a breakdown can never replan).
        for (t0, fr0, node0, halted0) in start_states:
            hh = h(node0, fr0)
            if hh == INF:
                continue
            key = (node0, fr0, t0)
            cost0 = float(t0)
            best[key] = cost0
            parents[key] = (None, t0, halted0)
            if halted0:
                free_halt_keys.add(key)
            if start_on_map:
                overstay_keys.add(key)
            heapq.heappush(openq, (cost0 + hh, cost0, t0, node0, fr0, 0, key))

        goal_key = None
        expansions = 0
        while openq:
            f, cost, t, node, fr, halts, key = heapq.heappop(openq)
            if best.get(key, INF) < cost - 1e-9:
                continue
            expansions += 1
            if expansions > (max_expansions or self.max_expansions):
                return None
            if node in goal_nodes:
                goal_key = key
                break
            res_u = g.resource_of_node(node)
            if key in overstay_keys:
                free_til = horizon
            else:
                free_til = self.res.free_until(res_u, t)
                if free_til <= t:
                    continue  # should not happen: we entered an occupied cell
            # nonstop exit time
            k = (q - fr + p - 1) // p  # ceil((q-fr)/p) steps to exit
            exit_nonstop = t + k
            if exit_nonstop > horizon:
                continue
            new_fr = fr + k * p - q  # fraction numerator on entering successor
            halt_w = 0.0 if key in free_halt_keys else halt_weight
            for v_node in g.successors(node):
                if dist[v_node] == UNREACH and v_node not in goal_nodes:
                    continue
                res_v = g.resource_of_node(v_node)
                # candidate entry times into v: earliest in each safe interval of res_v
                for s, e in self.res.safe_intervals(res_v, exit_nonstop):
                    if s > free_til:  # cannot wait in u beyond free_til
                        break
                    tau = max(exit_nonstop, s + hw if s > 0 else s)
                    # swap protection: occupant of res_v vacating into res_u at tau
                    while tau < e:
                        vt = self.res.vacate_to.get((tau, res_v))
                        if vt is not None and vt[0] == res_u:
                            tau += 1
                            continue
                        break
                    if tau >= e or tau > free_til or tau > horizon:
                        continue
                    halted = tau > exit_nonstop
                    step_cost = (tau - t) + (halt_w if halted else 0.0)
                    if halted and stop_bonus is not None:
                        sb = stop_bonus.get(node)
                        if sb is not None:
                            _, la_w, ed_w = sb
                            step_cost += -50.0 \
                                + 0.5 * max(0, t - la_w) \
                                + 0.5 * max(0, ed_w - tau)
                    nkey = (v_node, new_fr, tau)
                    ncost = cost + step_cost
                    if ncost < best.get(nkey, INF) - 1e-9:
                        hh = h(v_node, new_fr)
                        if hh == INF:
                            continue
                        best[nkey] = ncost
                        parents[nkey] = (key, tau, halted)
                        heapq.heappush(openq, (ncost + hh, ncost, tau, v_node, new_fr, halts + (1 if halted else 0), nkey))
        if goal_key is None:
            return None

        # reconstruct: chain of (key=(node,fr,t), halted_on_the_way)
        chain = []
        key = goal_key
        while key is not None:
            pkey, t_enter, halted = parents[key]
            chain.append((key[0], t_enter, halted))
            key = pkey
        chain.reverse()

        visits: List[CellVisit] = []
        for i, (node, t_enter, halted) in enumerate(chain):
            t_exit = chain[i + 1][1] if i + 1 < len(chain) else t_enter + 1
            # 'halted' flag belongs to the PREVIOUS cell (extended stay before
            # entering this one); recompute per cell below
            visits.append(CellVisit(node, t_enter, t_exit, False))
        # mark halts: cell i has a halt if next entry later than nonstop exit
        out: List[CellVisit] = []
        for i, cv in enumerate(visits):
            halt = False
            if i + 1 < len(visits):
                halt = chain[i + 1][2]
            out.append(CellVisit(cv.node, cv.t_enter, cv.t_exit, halt))
        return out
