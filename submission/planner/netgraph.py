"""
Static rail network model extracted from a RailEnv.

Nodes are (row, col, direction) encoded as ((r * W) + c) * 4 + d.
Resources are cells, except level-free crossings which split into two
resources by axis (so two trains may cross on different axes).
"""
from collections import deque
from typing import Dict, List, Set, Tuple

import numpy as np

# RailEnvActions values (avoid importing flatland here to keep module testable)
DO_NOTHING, MOVE_LEFT, MOVE_FORWARD, MOVE_RIGHT, STOP_MOVING = 0, 1, 2, 3, 4

# (next_dir - cur_dir) % 4 -> action
TURN_TO_ACTION = {3: MOVE_LEFT, 0: MOVE_FORWARD, 1: MOVE_RIGHT, 2: MOVE_FORWARD}  # 2: dead-end reversal


class NetGraph:
    def __init__(self, env):
        rail = env.rail
        self.height = env.height
        self.width = env.width
        H, W = self.height, self.width
        self.n_nodes = H * W * 4

        # successors as CSR arrays: for node u, succ in [ptr[u], ptr[u+1])
        succ_lists: List[List[int]] = [[] for _ in range(self.n_nodes)]
        rail_cells = []
        for r in range(H):
            for c in range(W):
                if rail.get_full_transitions(r, c) == 0:
                    continue
                rail_cells.append((r, c))
                for d in range(4):
                    tr = rail.get_transitions(((r, c), d))
                    if tr == (0, 0, 0, 0):
                        continue
                    u = ((r * W) + c) * 4 + d
                    for nd in range(4):
                        if tr[nd]:
                            nr, nc = self._new_pos(r, c, nd)
                            if 0 <= nr < H and 0 <= nc < W:
                                succ_lists[u].append(((nr * W) + nc) * 4 + nd)
        self.rail_cells = rail_cells

        ptr = np.zeros(self.n_nodes + 1, dtype=np.int64)
        for u in range(self.n_nodes):
            ptr[u + 1] = ptr[u] + len(succ_lists[u])
        flat = np.zeros(ptr[-1], dtype=np.int32)
        for u in range(self.n_nodes):
            flat[ptr[u]:ptr[u + 1]] = succ_lists[u]
        self.succ_ptr = ptr
        self.succ = flat

        # predecessors (for backward BFS heuristics)
        pred_lists: List[List[int]] = [[] for _ in range(self.n_nodes)]
        for u in range(self.n_nodes):
            for v in succ_lists[u]:
                pred_lists[v].append(u)
        pptr = np.zeros(self.n_nodes + 1, dtype=np.int64)
        for u in range(self.n_nodes):
            pptr[u + 1] = pptr[u] + len(pred_lists[u])
        pflat = np.zeros(pptr[-1], dtype=np.int32)
        for u in range(self.n_nodes):
            pflat[pptr[u]:pptr[u + 1]] = pred_lists[u]
        self.pred_ptr = pptr
        self.pred = pflat

        # resources: cell index, level-free cells split by axis
        # resource id = cell_id * 2 + (axis) where axis = d % 2 for level-free, else 0
        level_free: Set[Tuple[int, int]] = set()
        rm = getattr(env, "resource_map", None)
        if rm is not None:
            lf = getattr(rm, "level_free_positions", None)
            if lf:
                level_free = set(map(tuple, lf))
        self.level_free = level_free
        self._dist_cache: Dict[frozenset, np.ndarray] = {}

        self._build_corridors(rail, succ_lists)

    def _build_corridors(self, rail, succ_lists):
        """Identify single-track corridor segments: maximal chains of 'plain'
        cells (undirected degree <= 2, no routing choice in any direction).
        For each directed node inside a corridor, record (segment id, sense):
        opposing trains have different sense in the same segment."""
        H, W = self.height, self.width
        plain = {}
        for (r, c) in self.rail_cells:
            neighbors = set()
            choice = False
            for d in range(4):
                tr = rail.get_transitions(((r, c), d))
                n = sum(tr)
                if n > 1:
                    choice = True
                for nd in range(4):
                    if tr[nd]:
                        neighbors.add(self._new_pos(r, c, nd))
            if not choice and len(neighbors) <= 2 and (r, c) not in self.level_free:
                plain[(r, c)] = True

        # corridor id per cell via union of plain neighbors
        self.corridor_id = np.full(H * W, -1, dtype=np.int32)
        seg = 0
        seen = set()
        self.corridor_cells: List[List[int]] = []
        for cell in plain:
            if cell in seen:
                continue
            stack = [cell]
            comp = []
            seen.add(cell)
            while stack:
                cur = stack.pop()
                comp.append(cur)
                r, c = cur
                for d in range(4):
                    tr = rail.get_transitions(((r, c), d))
                    for nd in range(4):
                        if tr[nd]:
                            nb = self._new_pos(r, c, nd)
                            if nb in plain and nb not in seen:
                                seen.add(nb)
                                stack.append(nb)
            cids = []
            for (r, c) in comp:
                self.corridor_id[r * W + c] = seg
                cids.append(r * W + c)
            self.corridor_cells.append(cids)
            seg += 1
        self.n_corridors = seg

        # sense per directed node: follow unique transitions until leaving the
        # segment; sense = exit cell id (or -1 if trapped/dead-end loop)
        self.corridor_sense = np.full(self.n_nodes, -1, dtype=np.int64)
        for cell_list in self.corridor_cells:
            for cid in cell_list:
                r, c = cid // W, cid % W
                for d in range(4):
                    u = cid * 4 + d
                    if self.corridor_sense[u] != -1:
                        continue
                    if not succ_lists[u] and rail.get_transitions(((r, c), d)) == (0, 0, 0, 0):
                        continue
                    # trace
                    path = []
                    cur = u
                    sense = -1
                    for _ in range(4 * len(cell_list) + 8):
                        path.append(cur)
                        nxts = succ_lists[cur]
                        if not nxts:
                            break
                        nxt = nxts[0]
                        if self.corridor_id[nxt // 4] != self.corridor_id[cid]:
                            sense = nxt // 4  # exit cell
                            break
                        if self.corridor_sense[nxt] != -1:
                            sense = self.corridor_sense[nxt]
                            break
                        cur = nxt
                    for n_ in path:
                        if self.corridor_id[n_ // 4] == self.corridor_id[cid]:
                            self.corridor_sense[n_] = sense

    @staticmethod
    def _new_pos(r, c, d):
        if d == 0:
            return r - 1, c
        if d == 1:
            return r, c + 1
        if d == 2:
            return r + 1, c
        return r, c - 1

    # --- node helpers ---------------------------------------------------
    def node_of(self, pos, d) -> int:
        return ((pos[0] * self.width) + pos[1]) * 4 + d

    def pos_dir(self, node: int) -> Tuple[Tuple[int, int], int]:
        d = node % 4
        cell = node // 4
        return (cell // self.width, cell % self.width), d

    def resource_of_node(self, node: int) -> int:
        d = node % 4
        cell = node // 4
        r, c = cell // self.width, cell % self.width
        if (r, c) in self.level_free:
            return cell * 2 + (d % 2)
        return cell * 2

    def successors(self, u: int):
        return self.succ[self.succ_ptr[u]:self.succ_ptr[u + 1]]

    @staticmethod
    def action_for(u: int, v: int) -> int:
        return TURN_TO_ACTION[((v % 4) - (u % 4)) % 4]

    # --- heuristic: BFS cell-count distance to a set of goal nodes -------
    def dist_to_goals(self, goal_nodes: frozenset) -> np.ndarray:
        """dist[node] = minimum number of cell transitions from node to any
        goal node (uint16, 65535 = unreachable). Cached per goal set."""
        cached = self._dist_cache.get(goal_nodes)
        if cached is not None:
            return cached
        INF = np.uint16(65535)
        dist = np.full(self.n_nodes, INF, dtype=np.uint16)
        dq = deque()
        for g in goal_nodes:
            if dist[g] != 0:
                dist[g] = 0
                dq.append(g)
        pred, pptr = self.pred, self.pred_ptr
        while dq:
            v = dq.popleft()
            dv = dist[v] + 1
            for u in pred[pptr[v]:pptr[v + 1]]:
                if dv < dist[u]:
                    dist[u] = dv
                    dq.append(u)
        self._dist_cache[goal_nodes] = dist
        return dist
