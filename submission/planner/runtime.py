"""
Plan-and-execute runtime for the ECML2026 competition.

On the first act() call it builds the network graph and plans all trains by
priority (SIPP over a shared reservation table). Each step it compares the
env state against the plans, replans deviating/malfunctioning trains, and
emits per-agent actions.

Cost model driving decisions (flatland 4.2.6 / ECML2026Rewards):
- MOVING->STOPPED transition: -250 * speed (even when braking cleanly!)
- intermediate stop not served: -50 (serving needs a halt: usually worse)
- late at target: -1/step; cancellation: -5 * shortest path time
- so: minimize halt events, skip stops, hold trains off-map for timing.
"""
import os
from fractions import Fraction
from typing import Dict, List, Optional

from .netgraph import NetGraph, DO_NOTHING, MOVE_FORWARD, STOP_MOVING
from .sipp import CellVisit, Reservations, SippPlanner

# TrainState values (avoid importing flatland)
ST_WAITING, ST_READY, ST_MALF_OFF_MAP, ST_MOVING, ST_STOPPED, ST_MALFUNCTION, ST_DONE = range(7)

OFF_MAP_STATES = (ST_WAITING, ST_READY, ST_MALF_OFF_MAP)


class AgentPlan:
    __slots__ = ("visits", "actions", "idx", "cancelled", "retry_at", "fails")

    def __init__(self):
        self.visits: Optional[List[CellVisit]] = None
        self.actions: Dict[int, int] = {}
        self.idx = 0
        self.cancelled = False
        self.retry_at = -1
        self.fails = 0


class PlannerRuntime:
    def __init__(self, halt_factor: float = 250.0, headway: int = 0,
                 max_replans_per_step: int = 16, max_expansions: int = 300000,
                 cascade: bool = False, relaxed_fallback: bool = False):
        self.halt_factor = halt_factor
        self.headway = int(os.environ.get("PLANNER_HEADWAY", headway))
        self.max_replans_per_step = max_replans_per_step
        self.max_expansions = max_expansions
        self.cascade = os.environ.get("PLANNER_CASCADE", str(int(cascade))) == "1"
        self.relaxed_fallback = os.environ.get("PLANNER_RELAXED", str(int(relaxed_fallback))) == "1"
        # serve-stops OFF by default: opportunistic serving triggers
        # departure-delay malfunctions at stations and destabilizes dense
        # scenarios (ablation 2026-06-10); rebates don't cover the chaos.
        self.serve_stops = os.environ.get("PLANNER_SERVE_STOPS", "0") == "1"
        self.corridor_guard = os.environ.get("PLANNER_CORRIDOR_GUARD", "1") == "1"
        # Defaults = best measured stack (2026-06-10 matrix on example
        # curriculum + storm envs). shift/eager repair are net-negative:
        # they preserve doomed routes where a full SIPP replan reroutes.
        self.deadlock_every = int(os.environ.get("PLANNER_DEADLOCK_EVERY", "5"))
        self.owner_check = os.environ.get("PLANNER_OWNER_CHECK", "1") == "1"
        self.eager_repair = os.environ.get("PLANNER_EAGER_REPAIR", "0") == "1"
        self.shift_repair = os.environ.get("PLANNER_SHIFT_REPAIR", "0") == "1"
        self.flat_backoff = os.environ.get("PLANNER_FLAT_BACKOFF", "0") == "1"
        # runtime governor: degrade planning effort when nearing the
        # 30-min per-scenario evaluation cap (leave room for env stepping)
        self.time_budget_s = float(os.environ.get("PLANNER_TIME_BUDGET_S", "1200"))
        self._time_spent = 0.0
        # escape moves measured net-negative (more halts, parked trains block
        # routes) — keep available for experimentation only
        self.escape_moves = os.environ.get("PLANNER_ESCAPE_MOVES", "0") == "1"
        self._initialized = False
        self.graph: Optional[NetGraph] = None
        self.res: Optional[Reservations] = None
        self.plans: Dict[int, AgentPlan] = {}
        self.dirty: set = set()
        self._goal_nodes: Dict[int, frozenset] = {}
        self._speeds: Dict[int, Fraction] = {}
        self._malf_known: Dict[int, int] = {}  # agent -> known malfunction end time
        self._stub_blocks: Dict[int, List[CellVisit]] = {}  # plan-less on-map blocks
        self._la: Dict[int, int] = {}  # latest arrival per agent (priority)
        # agent -> {node: (group, la, ed)} for intermediate stops; serving by
        # halting at the node rebates the -50 skip penalty (window-aware)
        self._stop_nodes: Dict[int, Dict[int, tuple]] = {}
        self._served: Dict[int, set] = {}  # agent -> set of served group idxs
        self._dead: set = set()            # permanently deadlocked agents
        self._last_escape: Dict[int, int] = {}

    # ------------------------------------------------------------------
    def _setup(self, env):
        self.graph = NetGraph(env)
        horizon = env._max_episode_steps + 1
        self.res = Reservations(horizon)
        self.sipp = SippPlanner(self.graph, self.res, horizon,
                                headway=self.headway, max_expansions=self.max_expansions)
        g = self.graph
        for agent in env.agents:
            h = agent.handle
            self.plans[h] = AgentPlan()
            self._speeds[h] = Fraction(agent.speed_counter.max_speed)
            self._la[h] = agent.latest_arrival if agent.latest_arrival is not None else 1 << 30
            self._goal_nodes[h] = frozenset(
                g.node_of(pos, int(d)) for pos, d in agent.targets
            )
            stops: Dict[int, tuple] = {}
            wps = agent.waypoints or []
            wla = agent.waypoints_latest_arrival or []
            wed = agent.waypoints_earliest_departure or []
            for gi in range(1, max(len(wps) - 1, 0)):
                la = wla[gi] if gi < len(wla) and wla[gi] is not None else 10 ** 9
                ed = wed[gi] if gi < len(wed) and wed[gi] is not None else 0
                for wp in wps[gi]:
                    if wp.direction is None:
                        continue
                    stops[g.node_of(wp.position, int(wp.direction))] = (gi, la, ed)
            self._stop_nodes[h] = stops
            self._served[h] = set()

        # initial planning in priority order: tightest slack first
        def slack(agent):
            start = g.node_of(agent.initial_configuration[0], int(agent.initial_configuration[1]))
            dist = g.dist_to_goals(self._goal_nodes[agent.handle])[start]
            v = self._speeds[agent.handle]
            travel = (dist * v.denominator + v.numerator - 1) // v.numerator if dist != 65535 else 10 ** 9
            return (agent.latest_arrival or 10 ** 9) - agent.earliest_departure - travel

        order = sorted(env.agents, key=slack)
        for agent in order:
            self._plan_agent(env, agent.handle)
        self._initialized = True

    # ------------------------------------------------------------------
    def _try_shift_repair(self, env, h: int) -> bool:
        """If the agent is merely delayed on its planned route, shift the
        remaining schedule by the delay instead of re-searching."""
        g = self.graph
        agent = env.agents[h]
        ap = self.plans[h]
        # only for trains whose halt is already paid (stopped/malfunction):
        # shifting a MOVING train injects a fresh -250 boundary stop where a
        # full replan could reroute for free.
        if not ap.visits or agent.state not in (ST_STOPPED, ST_MALFUNCTION):
            return False
        if agent.position is None:
            return False
        node = g.node_of(agent.position, int(agent.direction))
        # find current node within the next few planned visits
        idx = None
        for i in range(max(ap.idx - 1, 0), min(ap.idx + 3, len(ap.visits))):
            if ap.visits[i].node == node:
                idx = i
                break
        if idx is None:
            return False
        t = env._elapsed_steps
        mf = agent.malfunction_handler.malfunction_down_counter
        cv = ap.visits[idx]
        speed = self._speeds[h]
        p, q = speed.numerator, speed.denominator
        fr_num = int(Fraction(agent.speed_counter.distance) * q)
        remaining = (q - fr_num + p - 1) // p  # steps to exit once moving
        t_exit_new = t + mf + remaining
        delta = t_exit_new - cv.t_exit
        if delta <= 0:
            return False  # not late; some other kind of deviation
        rest = ap.visits[idx:]
        shifted = [CellVisit(v.node, (v.t_enter + delta) if i_ > 0 else min(v.t_enter, t),
                             v.t_exit + delta, v.halt) for i_, v in enumerate(rest)]
        if shifted[-1].t_exit >= self.res.horizon:
            return False
        self.res.remove_agent_path(h, g, ap.visits)
        if self.res.conflicts_with(h, g, shifted):
            self.res.add_path(h, g, ap.visits)  # restore, fall back to SIPP
            return False
        ap.visits = shifted
        ap.idx = 0
        ap.fails = 0
        self.res.add_path(h, g, shifted)
        self._build_actions(h, self._speeds[h], False,
                            start_halted=agent.state in (ST_STOPPED, ST_MALFUNCTION),
                            start_frac=Fraction(agent.speed_counter.distance))
        return True

    # ------------------------------------------------------------------
    def _plan_agent(self, env, h: int) -> bool:
        """(Re)plan agent h from its current state. Replaces reservations."""
        g = self.graph
        agent = env.agents[h]
        ap = self.plans[h]
        if h in self._dead:
            return False
        if self.shift_repair and self._try_shift_repair(env, h):
            return True
        if ap.visits:
            self.res.remove_agent_path(h, g, ap.visits)
            ap.visits = None
            ap.actions = {}
            ap.idx = 0
        stub = self._stub_blocks.pop(h, None)
        if stub is not None:
            self.res.remove_agent_path(h, g, stub)

        if agent.state == ST_DONE:
            return True

        t = env._elapsed_steps
        speed = self._speeds[h]
        goal = self._goal_nodes[h]
        halt_weight = self.halt_factor * float(speed)
        _eb = int(os.environ.get("PLANNER_EXP_BUDGET", "300000"))
        exp_budget = None if not self._initialized else _eb
        stop_bonus = None
        if self.serve_stops:
            served = self._served.get(h, ())
            stop_bonus = {n: v for n, v in self._stop_nodes.get(h, {}).items()
                          if v[0] not in served} or None

        start_frac = None
        if agent.state in OFF_MAP_STATES:
            start_node = g.node_of(agent.initial_configuration[0], int(agent.initial_configuration[1]))
            earliest = max(agent.earliest_departure, 1) + 1
            if agent.state == ST_MALF_OFF_MAP:
                earliest = max(earliest, t + agent.malfunction_handler.malfunction_down_counter + 1)
            earliest = max(earliest, t + 1)
            visits = self.sipp.plan(
                h, start_node, goal, speed, earliest,
                start_on_map=False, latest_useful=agent.latest_arrival,
                halt_weight=halt_weight, stop_bonus=stop_bonus,
                max_expansions=exp_budget,
            )
        else:
            start_node = g.node_of(agent.position, int(agent.direction))
            if start_node in goal:
                return True
            fr = start_frac = Fraction(agent.speed_counter.distance)
            earliest = t
            halted = agent.state in (ST_STOPPED, ST_MALFUNCTION)
            if agent.state == ST_MALFUNCTION:
                earliest = t + agent.malfunction_handler.malfunction_down_counter
            visits = self.sipp.plan(
                h, start_node, goal, speed, earliest,
                start_frac=fr, start_on_map=True, start_halted=halted,
                latest_useful=agent.latest_arrival, halt_weight=halt_weight,
                stop_bonus=stop_bonus, max_expansions=exp_budget,
            )
            if visits is not None and visits[0].t_enter > t:
                # train occupies its cell from NOW (e.g. malfunction until
                # `earliest`); extend the first visit backwards.
                visits[0] = CellVisit(visits[0].node, t, visits[0].t_exit, visits[0].halt)

        if visits is None and self.relaxed_fallback and agent.state not in OFF_MAP_STATES:
            # full table is too crowded: plan against static blocks only
            # (stopped/malfunctioning trains and stubs); accept the risk of
            # en-route conflicts, which the executor resolves by replanning.
            relaxed = self._static_reservations(env, exclude=h)
            saved = self.sipp.res
            self.sipp.res = relaxed
            try:
                visits = self.sipp.plan(
                    h, start_node, goal, speed, earliest,
                    start_frac=start_frac or Fraction(0), start_on_map=True,
                    start_halted=agent.state in (ST_STOPPED, ST_MALFUNCTION),
                    halt_weight=halt_weight, stop_bonus=stop_bonus,
                )
            finally:
                self.sipp.res = saved

        if (visits is None and self.escape_moves and agent.state not in OFF_MAP_STATES
                and ap.fails >= 1 and t - self._last_escape.get(h, -1 << 30) > 50):
            # jam dissolution: can't reach the goal, so step out of the way.
            visits = self._plan_escape(env, h)
            if visits is not None:
                self._last_escape[h] = t
                ap.visits = visits
                ap.cancelled = False
                ap.idx = 0
                self.res.add_path(h, g, visits)
                self._build_actions(h, speed, False,
                                    start_halted=agent.state in (ST_STOPPED, ST_MALFUNCTION),
                                    start_frac=start_frac)
                # brake on arrival at the parking cell, then retry the goal
                ap.actions[visits[-1].t_enter] = STOP_MOVING
                ap.retry_at = visits[-1].t_enter + 3
                return True

        if visits is None:
            ap.cancelled = agent.state in OFF_MAP_STATES
            ap.fails += 1
            ap.retry_at = t + (10 if self.flat_backoff else min(60, 5 * (2 ** min(ap.fails, 4))))
            if agent.state not in OFF_MAP_STATES and agent.position is not None:
                # plan-less on-map train: block its current cell TO HORIZON so
                # others route around it instead of queueing head-on next to
                # it waiting for a short stub to expire (wedge trap). The
                # block is replaced by real reservations once a replan
                # succeeds.
                node = g.node_of(agent.position, int(agent.direction))
                stub = [CellVisit(node, t, self.res.horizon, False)]
                ap.visits = None
                ap.actions = {}
                self._stub_blocks[h] = stub
                self.res.add_path(h, g, stub)
            return False
        ap.visits = visits
        ap.cancelled = False
        ap.idx = 0
        ap.fails = 0
        ap.retry_at = -1
        # wake agents whose plans now conflict with this new path
        if self.cascade:
            for other in self.res.conflicting_owners(h, g, visits):
                self.dirty.add(other)
        elif agent.state not in OFF_MAP_STATES:
            # overstay rights: our start-cell stay overrides others' stale
            # claims of it — wake all claim owners so they re-time around the
            # truth. (Waking only the earliest claimant tested worse: the
            # queue learns the truth too slowly and wedges form.)
            for other in self.res.conflicting_owners(h, g, visits[:1]):
                self.dirty.add(other)
        self.res.add_path(h, g, visits)
        self._build_actions(h, speed, agent.state in OFF_MAP_STATES,
                            start_halted=agent.state in (ST_STOPPED, ST_MALFUNCTION),
                            start_frac=start_frac)
        return True

    # ------------------------------------------------------------------
    def _plan_escape(self, env, h: int) -> Optional[List[CellVisit]]:
        """Short get-out-of-the-way plan for a stuck train: BFS over cells
        free right now (and unreserved over our traversal window) to a cell
        with breathing room, then park there. Dissolves jams edge-inward."""
        g = self.graph
        agent = env.agents[h]
        if agent.position is None or agent.malfunction_handler.malfunction_down_counter > 0:
            return None
        t = env._elapsed_steps
        speed = self._speeds[h]
        p, q = speed.numerator, speed.denominator
        steps_per_cell = (q + p - 1) // p

        occupied = set()
        for other in env.agents:
            if other.handle != h and other.position is not None and other.state != ST_DONE:
                occupied.add(other.position[0] * g.width + other.position[1])

        start = g.node_of(agent.position, int(agent.direction))
        # BFS over free cells
        from collections import deque
        prev = {start: None}
        dq = deque([(start, 0)])
        target = None
        best = None
        while dq:
            u, depth = dq.popleft()
            if depth > 0:
                cell = u // 4
                # breathing room = few occupied neighbours and not a corridor
                # cell right next to the jam
                free_succ = sum(1 for v in g.successors(u) if v // 4 not in occupied)
                score = (free_succ, depth)
                if best is None or score > best[0]:
                    best = (score, u)
                if depth >= 12:
                    break
            if depth >= 12:
                continue
            for v in g.successors(u):
                if v in prev or (v // 4) in occupied:
                    continue
                prev[v] = u
                dq.append((v, depth + 1))
        if best is None:
            return None
        target = best[1]
        # reconstruct node path
        path = []
        u = target
        while u is not None:
            path.append(u)
            u = prev[u]
        path.reverse()
        if len(path) < 2:
            return None
        # timed visits: nonstop pace, park at the end until horizon
        visits: List[CellVisit] = []
        fr = int(Fraction(agent.speed_counter.distance) * q)
        t_enter = t
        for i, node in enumerate(path):
            k = (q - fr + p - 1) // p
            fr = fr + k * p - q
            t_exit = t_enter + k if i + 1 < len(path) else self.res.horizon
            visits.append(CellVisit(node, t_enter, t_exit, i + 1 == len(path)))
            t_enter = t_exit
        # validate against reservations (others' paths/stubs)
        if self.res.conflicts_with(h, g, visits):
            return None
        return visits
        """Reservation table holding only non-negotiable occupancies: trains
        currently stopped/malfunctioning (for their known durations) and
        plan-less stubs."""
        g = self.graph
        t = env._elapsed_steps
        res = Reservations(self.res.horizon)
        for agent in env.agents:
            h = agent.handle
            if h == exclude or agent.state == ST_DONE or agent.position is None:
                continue
            if agent.state in (ST_STOPPED, ST_MALFUNCTION):
                dur = max(agent.malfunction_handler.malfunction_down_counter, 2)
                node = g.node_of(agent.position, int(agent.direction))
                res.block(h, g.resource_of_node(node), t, min(t + dur + 1, res.horizon))
            else:
                node = g.node_of(agent.position, int(agent.direction))
                res.block(h, g.resource_of_node(node), t, t + 1)
        return res

    # ------------------------------------------------------------------
    def _build_actions(self, h: int, speed: Fraction, from_off_map: bool,
                       start_halted: bool = False, start_frac: Fraction = None):
        """Translate timed visits into env_time -> action schedule."""
        g = self.graph
        ap = self.plans[h]
        visits = ap.visits
        acts: Dict[int, int] = {}
        p, q = speed.numerator, speed.denominator

        if from_off_map:
            # placement attempt(s): MOVE_FORWARD from t_enter-1 (harmless if blocked)
            acts[visits[0].t_enter - 1] = MOVE_FORWARD

        # fraction numerator on entry of each visit
        fr_num = int((start_frac or Fraction(0)) * q) if not from_off_map else 0
        for i, cv in enumerate(visits):
            if i + 1 >= len(visits):
                break  # goal cell: nothing to do after entering
            nxt = visits[i + 1]
            turn = g.action_for(cv.node, nxt.node)
            k = (q - fr_num + p - 1) // p  # nonstop steps to exit this cell
            exit_nonstop = cv.t_enter + k
            if i == 0 and start_halted and nxt.t_enter > exit_nonstop:
                # already stopped: stay stopped first (free), then go nonstop
                # so we do not restart and re-brake (-250).
                go = nxt.t_enter - k
                for e in range(max(cv.t_enter, 0), go):
                    acts[e] = DO_NOTHING
                for e in range(go, nxt.t_enter):
                    acts[e] = turn
            elif nxt.t_enter > exit_nonstop:
                # halt at cell-exit boundary, resume to enter next at nxt.t_enter
                for e in range(cv.t_enter, cv.t_enter + k - 1):
                    acts[e] = turn
                acts[cv.t_enter + k - 1] = STOP_MOVING
                for e in range(cv.t_enter + k, nxt.t_enter - 1):
                    acts[e] = DO_NOTHING
                acts[nxt.t_enter - 1] = turn
            else:
                for e in range(cv.t_enter, cv.t_enter + k - 1):
                    acts[e] = turn
                acts[cv.t_enter + k - 1] = turn
            fr_num = fr_num + k * p - q
        ap.actions = acts

    # ------------------------------------------------------------------
    def _expected_node(self, ap: AgentPlan, t: int):
        """(node, is_last) expected at time t, advancing the visit pointer."""
        visits = ap.visits
        while ap.idx + 1 < len(visits) and visits[ap.idx + 1].t_enter <= t:
            ap.idx += 1
        cv = visits[ap.idx]
        return cv, ap.idx == len(visits) - 1

    # ------------------------------------------------------------------
    def _detect_deadlocks(self, env):
        """Mark only PROVABLY dead trains: mutual pairs where each train's
        only successor cell is the other's cell (flatland forbids swaps).
        Larger cycles can rotate in flatland, so they are NOT dead."""
        g = self.graph
        occ: Dict[int, int] = {}
        nodes: Dict[int, int] = {}
        for agent in env.agents:
            if agent.position is not None and agent.state != ST_DONE:
                cell = agent.position[0] * g.width + agent.position[1]
                if (agent.position[0], agent.position[1]) in g.level_free:
                    continue  # crossings: treat as passable
                occ[cell] = agent.handle
                nodes[agent.handle] = g.node_of(agent.position, int(agent.direction))

        t = env._elapsed_steps
        for h, u in nodes.items():
            if h in self._dead:
                continue
            cells = {v // 4 for v in g.successors(u)}
            if len(cells) != 1:
                continue
            c = next(iter(cells))
            other = occ.get(c)
            if other is None or other in self._dead:
                continue
            o_cells = {v // 4 for v in g.successors(nodes[other])}
            my_cell = nodes[h] // 4
            if o_cells == {my_cell}:
                for hh in (h, other):
                    self._dead.add(hh)
                    ap = self.plans[hh]
                    if ap.visits:
                        self.res.remove_agent_path(hh, g, ap.visits)
                        ap.visits = None
                        ap.actions = {}
                    stub = self._stub_blocks.pop(hh, None)
                    if stub is not None:
                        self.res.remove_agent_path(hh, g, stub)
                    self.res.block(hh, g.resource_of_node(nodes[hh]), t, self.res.horizon)
                    self.dirty.discard(hh)

    # ------------------------------------------------------------------
    def _build_corridor_occupancy(self, env):
        """Per-step index: corridor segment -> [(handle, sense)]."""
        g = self.graph
        occ: Dict[int, List] = {}
        for agent in env.agents:
            if agent.position is None or agent.state == ST_DONE:
                continue
            cell = agent.position[0] * g.width + agent.position[1]
            seg = g.corridor_id[cell]
            if seg < 0:
                continue
            node = g.node_of(agent.position, int(agent.direction))
            occ.setdefault(seg, []).append((agent.handle, g.corridor_sense[node]))
        self._corridor_occ = occ

    def _corridor_danger(self, env, h: int, enter_node: int) -> bool:
        """True if enter_node lies in a single-track corridor currently
        containing a hopeless train (dead or plan-less). Trains WITH plans
        are covered by reservations + the owner check; holding for them
        costs a certain -250 where a crash is only probabilistic."""
        g = self.graph
        seg = g.corridor_id[enter_node // 4]
        if seg < 0:
            return False
        occupants = self._corridor_occ.get(seg)
        if not occupants:
            return False
        for oh, osense in occupants:
            if oh == h:
                continue
            if oh in self._dead or not self.plans[oh].visits:
                return True
        return False

    # ------------------------------------------------------------------
    def act(self, env) -> Dict[int, int]:
        import time as _time
        _t0 = _time.perf_counter()
        try:
            return self._act(env)
        finally:
            self._time_spent += _time.perf_counter() - _t0

    def _act(self, env) -> Dict[int, int]:
        if not self._initialized:
            self._setup(env)

        t = env._elapsed_steps  # actions chosen now realize at t+1
        g = self.graph
        if self._time_spent > self.time_budget_s and self.max_replans_per_step > 4:
            # nearing the evaluation cap: degrade gracefully
            self.max_replans_per_step = 4
            os.environ["PLANNER_EXP_BUDGET"] = "30000"
        if t % self.deadlock_every == 0:
            self._detect_deadlocks(env)
        if self.corridor_guard:
            self._build_corridor_occupancy(env)

        # --- detect deviations & malfunctions --------------------------
        for agent in env.agents:
            h = agent.handle
            ap = self.plans[h]
            st = agent.state
            if st == ST_DONE or h in self._dead:
                continue
            # track served intermediate stops (STOPPED at a stop node)
            if st in (ST_STOPPED, ST_MALFUNCTION) and agent.position is not None:
                sb = self._stop_nodes[h].get(g.node_of(agent.position, int(agent.direction)))
                if sb is not None:
                    self._served[h].add(sb[0])
            # malfunction bookkeeping: replan when new malfunction appears/extends
            mf = agent.malfunction_handler.malfunction_down_counter
            if mf > 0:
                known = self._malf_known.get(h, -1)
                end = t + mf
                if end > known:
                    self._malf_known[h] = end
                    self.dirty.add(h)
                    # proactively wake trains routed through the blocked cell
                    if st in (ST_MOVING, ST_STOPPED, ST_MALFUNCTION) and agent.position is not None:
                        blocked = g.resource_of_node(g.node_of(agent.position, int(agent.direction)))
                        for other in env.agents:
                            oh = other.handle
                            if oh == h or other.state == ST_DONE:
                                continue
                            ov = self.plans[oh].visits
                            if not ov:
                                continue
                            for cv in ov:
                                if cv.t_enter > end:
                                    break
                                if cv.t_exit > t and g.resource_of_node(cv.node) == blocked:
                                    self.dirty.add(oh)
                                    break
                continue

            if ap.retry_at >= 0 and t >= ap.retry_at:
                self.dirty.add(h)  # parked escape or failed plan: retry goal
            if not ap.visits:
                continue

            if st in OFF_MAP_STATES:
                if t >= ap.visits[0].t_enter:
                    self.dirty.add(h)  # placement is late
            else:
                cv, _ = self._expected_node(ap, t)
                actual = g.node_of(agent.position, int(agent.direction))
                if actual != cv.node or t >= cv.t_exit:
                    self.dirty.add(h)

        # --- repair plans -------------------------------------------------
        # cheap shift-repairs for ALL dirty agents first (keeps the
        # reservation table truthful about delayed trains), then budgeted
        # full replans for the rest, most urgent first.
        if self.dirty and self.eager_repair:
            repaired = {h for h in self.dirty if h not in self._dead
                        and self._try_shift_repair(env, h)}
            self.dirty -= repaired
        budget = self.max_replans_per_step
        if self.dirty:
            for h in sorted(self.dirty, key=lambda x: self._la.get(x, 1 << 30)):
                if budget <= 0:
                    break
                self._plan_agent(env, h)
                self.dirty.discard(h)
                budget -= 1

        # --- emit actions ------------------------------------------------
        actions: Dict[int, int] = {}
        for agent in env.agents:
            h = agent.handle
            ap = self.plans[h]
            st = agent.state
            if h in self._dead:
                actions[h] = DO_NOTHING
                continue
            if st == ST_DONE or not ap.visits:
                # plan-less moving trains are stopped so they match their
                # stub block instead of drifting into trouble
                actions[h] = STOP_MOVING if st == ST_MOVING else DO_NOTHING
                continue
            a = ap.actions.get(t)
            if a is None:
                if st in OFF_MAP_STATES and t >= ap.visits[0].t_enter - 1:
                    a = MOVE_FORWARD  # keep attempting placement (free)
                else:
                    a = DO_NOTHING
            # entry guards: when the plan says we enter a new cell at t+1,
            # double-check it is safe even if our schedule has gone stale.
            enter_node = None
            if st in OFF_MAP_STATES:
                if a == MOVE_FORWARD and ap.visits[0].t_enter <= t + 1:
                    enter_node = ap.visits[0].node
                    cur_seg = -2
            elif st in (ST_MOVING, ST_STOPPED) and ap.idx + 1 < len(ap.visits):
                nxt = ap.visits[ap.idx + 1]
                if nxt.t_enter == t + 1 and a not in (DO_NOTHING, STOP_MOVING):
                    enter_node = nxt.node
                    cur_seg = g.corridor_id[ap.visits[ap.idx].node // 4]
            if enter_node is not None:
                hold = False
                # 1) reservation check: hold only when the slot is claimed by
                # an OPPOSING train (head-on risk). Same-direction conflicts
                # resolve as ordinary queueing, no worse than holding.
                owner = self.res.owner_at(g.resource_of_node(enter_node), t + 1) if self.owner_check else None
                if owner is not None and owner != h and owner >= 0:
                    o_agent = env.agents[owner]
                    if o_agent.position is not None and o_agent.state != ST_DONE:
                        o_dir = int(o_agent.direction)
                        if (o_dir - (enter_node % 4)) % 4 == 2:
                            hold = True
                # 2) corridor opposition guard: hold at the mouth until the
                # opposing/hopeless train clears; no phantom blocks (they
                # taxed every other route through the corridor).
                if not hold and self.corridor_guard:
                    seg = g.corridor_id[enter_node // 4]
                    if seg >= 0 and seg != cur_seg and self._corridor_danger(env, h, enter_node):
                        hold = True
                if hold:
                    self.dirty.add(h)
                    a = STOP_MOVING if st == ST_MOVING else DO_NOTHING
            actions[h] = a
        return actions
