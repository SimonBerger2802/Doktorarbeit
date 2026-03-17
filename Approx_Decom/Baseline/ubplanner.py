from __future__ import annotations

from dataclasses import dataclass
from math import ceil, sqrt, acos, pi, isclose
from typing import List, Optional, Tuple
import json
import time
 
from pyscipopt import Model as SCIPModel, quicksum
import pulp


Point2D = Tuple[float, float]
Cell = List[Point2D]


@dataclass
class PlannerInput:
    outer_boundary: List[Point2D]
    obstacles: List[List[Point2D]]
    agents: List[Point2D]


class UBPlanner:
    def __init__(self) -> None:
        self.m_file: str = ""
        self.m_res: float = 10.0
        self.m_limit: float = 1_000_000_000.0
        self.m_gap: float = 0.01
        self.m_lambda: float = 1.0
        self.m_gamma: float = 1.0
        self.m_kappa: float = 1_000_000_000.0
        self.m_pcn: float = 1.0

        self.m_depots: List[int] = []
        self.m_areas: List[List[Point2D]] = []
        self.m_nodes: List[Point2D] = []
        self.m_agents: List[Point2D] = []

        # Pro Agent: Liste von (node_index, successor_index)
        self.m_agent_paths: List[List[Tuple[int, int]]] = []

    # ------------------------------------------------------------------
    # Setter
    # ------------------------------------------------------------------
    def set_file(self, file: str) -> None:
        self.m_file = file

    def set_resolution(self, res: float) -> None:
        self.m_res = res

    def set_limit(self, limit: float) -> None:
        self.m_limit = limit

    def set_gap(self, gap: float) -> None:
        self.m_gap = gap

    def set_lambda(self, lambda_value: float) -> None:
        self.m_lambda = lambda_value

    def set_gamma(self, gamma: float) -> None:
        self.m_gamma = gamma

    def set_kappa(self, kappa: float) -> None:
        self.m_kappa = kappa

    def set_precision(self, pcn: float) -> None:
        self.m_pcn = pcn

    # ------------------------------------------------------------------
    # Öffentlicher Einstieg
    # ------------------------------------------------------------------
    def start_planner(self) -> None:
        planner_input = self.load_input_file(self.m_file)

        self.m_areas = [self._close_polygon(planner_input.outer_boundary)]
        self.m_areas.extend(self._close_polygon(poly) for poly in planner_input.obstacles)

        self.m_agents = list(planner_input.agents)
        self.m_nodes = []
        self.m_depots = [0 for _ in self.m_agents]
        self.m_agent_paths = [[] for _ in self.m_agents]

        total_start = time.perf_counter()

        self.decompose()

        print("Anzahl Nodes gesamt:", len(self.m_nodes))

        if not self.divide():
            raise RuntimeError("Unable to divide the area between agents!")

        for a in range(len(self.m_agent_paths)):
            assigned = [node for node, _ in self.m_agent_paths[a]]
            print(f"Agent {a} assigned nodes: {assigned}")

        for agent in range(len(self.m_agents)):
            agent_start = time.perf_counter()

            if not self.plan_agent_quadratic(agent): #-self.plan_agent(agent):
                raise RuntimeError(f"Unable to plan the coverage path for agent: {agent}")

            if not self.validate_path(agent):
                raise RuntimeError(f"Unable to validate the coverage path for agent: {agent}")

            elapsed = time.perf_counter() - agent_start
            print(f"Elapsed time for agent {agent} is {elapsed:.3f}")

            self.build_mission(agent)

        total_elapsed = time.perf_counter() - total_start
        print(
            "The planner has successfully planned the mission for each agent "
            f"in total time {total_elapsed:.3f}"
        )

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------
    def load_input_file(self, load_file: str) -> PlannerInput:
        if not load_file:
            raise ValueError("No input file was set.")

        with open(load_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "outer_boundary" not in data:
            raise ValueError("Input JSON must contain 'outer_boundary'.")
        if "agents" not in data:
            raise ValueError("Input JSON must contain 'agents'.")

        outer_boundary = [tuple(map(float, p)) for p in data["outer_boundary"]]
        obstacles = [
            [tuple(map(float, p)) for p in poly]
            for poly in data.get("obstacles", [])
        ]
        agents = [tuple(map(float, p)) for p in data["agents"]]

        return PlannerInput(
            outer_boundary=outer_boundary,
            obstacles=obstacles,
            agents=agents,
        )

    # ------------------------------------------------------------------
    # Geometrie-Helfer
    # ------------------------------------------------------------------
    @staticmethod
    def _close_polygon(poly: List[Point2D]) -> List[Point2D]:
        if not poly:
            return poly
        if poly[0] != poly[-1]:
            return poly + [poly[0]]
        return poly

    @staticmethod
    def _distance(p1: Point2D, p2: Point2D) -> float:
        return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    @staticmethod
    def _bounding_rect(poly: List[Point2D]) -> Tuple[float, float, float, float]:
        xs = [p[0] for p in poly[:-1]] if len(poly) > 1 and poly[0] == poly[-1] else [p[0] for p in poly]
        ys = [p[1] for p in poly[:-1]] if len(poly) > 1 and poly[0] == poly[-1] else [p[1] for p in poly]
        return min(xs), max(xs), min(ys), max(ys)

    @staticmethod
    def _point_on_segment(p: Point2D, a: Point2D, b: Point2D, eps: float = 1e-9) -> bool:
        cross = (p[1] - a[1]) * (b[0] - a[0]) - (p[0] - a[0]) * (b[1] - a[1])
        if abs(cross) > eps:
            return False
        dot = (p[0] - a[0]) * (b[0] - a[0]) + (p[1] - a[1]) * (b[1] - a[1])
        if dot < -eps:
            return False
        sq_len = (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2
        if dot - sq_len > eps:
            return False
        return True

    @staticmethod
    def _contains_point(poly: List[Point2D], p: Point2D) -> bool:
        """
        Odd-even point-in-polygon mit Rand als innen.
        """
        n = len(poly)
        if n < 3:
            return False

        for i in range(n - 1):
            if UBPlanner._point_on_segment(p, poly[i], poly[i + 1]):
                return True

        inside = False
        x, y = p

        for i in range(n - 1):
            x1, y1 = poly[i]
            x2, y2 = poly[i + 1]

            intersects = ((y1 > y) != (y2 > y))
            if intersects:
                x_intersect = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                if x_intersect >= x:
                    inside = not inside

        return inside

    @staticmethod
    def _orientation(a: Point2D, b: Point2D, c: Point2D, eps: float = 1e-9) -> int:
        val = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
        if abs(val) <= eps:
            return 0
        return 1 if val > 0 else 2

    @staticmethod
    def _segments_intersect(a1: Point2D, a2: Point2D, b1: Point2D, b2: Point2D, eps: float = 1e-9) -> bool:
        o1 = UBPlanner._orientation(a1, a2, b1, eps)
        o2 = UBPlanner._orientation(a1, a2, b2, eps)
        o3 = UBPlanner._orientation(b1, b2, a1, eps)
        o4 = UBPlanner._orientation(b1, b2, a2, eps)

        # 2. Bedingung für echte Kreuzung:
        # - Alle Orientierungen müssen NICHT 0 sein (keine Kollinearität/Überlappung/Endpunkt-Berührung)
        # - Die Punkte von B müssen auf unterschiedlichen Seiten von A liegen (o1 != o2)
        # - Die Punkte von A müssen auf unterschiedlichen Seiten von B liegen (o3 != o4)
        
        if o1 == 0 or o2 == 0 or o3 == 0 or o4 == 0:
            return False  # Sofort False bei jeglicher Kollinearität oder Berührung
            
        return (o1 != o2) and (o3 != o4)

    @staticmethod
    def _same_point(p1: Point2D, p2: Point2D, eps: float = 1e-9) -> bool:
        return isclose(p1[0], p2[0], abs_tol=eps) and isclose(p1[1], p2[1], abs_tol=eps)


    @staticmethod
    def _turn_angle(i: Point2D, j: Point2D, k: Point2D) -> float:
        r = UBPlanner._distance(i, j)
        e = UBPlanner._distance(j, k)
        s = UBPlanner._distance(k, i)

        if r <= 0.0 or e <= 0.0:
            return 0.0

        t = (r * r + e * e - s * s) / (2.0 * r * e)
        if t > 1.0:
            t = 1.0
        elif t < -1.0:
            t = -1.0

        return pi - acos(t)

    # ------------------------------------------------------------------
    # decompose
    # ------------------------------------------------------------------
    def decompose(self) -> bool:
        if not self.m_areas:
            raise ValueError("No area loaded.")

        outer = self.m_areas[0]
        x_min, x_max, y_min, y_max = self._bounding_rect(outer)

        width = x_max - x_min
        height = y_max - y_min

        xstep = int(ceil(width / self.m_res))
        ystep = int(ceil(height / self.m_res))

        self.m_nodes.clear()

        for j in range(ystep):
            for i in range(xstep):
                x0 = x_min + i * self.m_res
                x1 = x_min + (i + 1) * self.m_res
                y0 = y_min + j * self.m_res
                y1 = y_min + (j + 1) * self.m_res

                cell: Cell = [
                    (x0, y0),
                    (x0, y1),
                    (x1, y1),
                    (x1, y0),
                    (x0, y0),
                ]

                if self.evaluate(cell):
                    xm = x_min + (i + 0.5) * self.m_res
                    ym = y_min + (j + 0.5) * self.m_res
                    self.m_nodes.append((xm, ym))

        return True

    # ------------------------------------------------------------------
    # evaluate
    # ------------------------------------------------------------------
    def evaluate(self, cell: Cell) -> bool:
        if not self.m_areas:
            return False

        outer = self.m_areas[0]
        obstacles = self.m_areas[1:]

        for i in range(len(cell) - 1):
            if not self._contains_point(outer, cell[i]):
                return False

            for obs in obstacles:
                if self._contains_point(obs, cell[i]):
                    return False

            a1 = cell[i]
            a2 = cell[i + 1]

            for area in self.m_areas:
                for k in range(len(area) - 1):
                    b1 = area[k]
                    b2 = area[k + 1]
                    if self._segments_intersect(a1, a2, b1, b2):
                        return False

        return True

    # ------------------------------------------------------------------
    # divide  (linear)
    # ------------------------------------------------------------------
    def divide(self) -> bool:
        if not self.m_agents:
            raise ValueError("No agents loaded.")
        if not self.m_nodes:
            raise ValueError("No nodes available after decomposition.")

        num_agents = len(self.m_agents)
        num_nodes = len(self.m_nodes)

        dist_agent_node = [
            [
                self.m_pcn * self._distance(self.m_agents[a], self.m_nodes[i])
                for i in range(num_nodes)
            ]
            for a in range(num_agents)
        ]

        model = pulp.LpProblem("divide_area", pulp.LpMinimize)

        x = {
            (a, i): pulp.LpVariable(f"x_{a}_{i}", cat="Binary")
            for a in range(num_agents)
            for i in range(num_nodes)
        }
        z = pulp.LpVariable("z", lowBound=0.0, cat="Continuous")

        model += z

        for a in range(num_agents):
            model += (
                pulp.lpSum(dist_agent_node[a][i] * x[(a, i)] for i in range(num_nodes)) <= z,
                f"agent_max_dist_{a}",
            )

        for i in range(num_nodes):
            model += (
                pulp.lpSum(x[(a, i)] for a in range(num_agents)) == 1,
                f"assign_node_{i}",
            )

        solver = pulp.PULP_CBC_CMD(
            msg=False,
            timeLimit=self.m_limit if self.m_limit < 1e15 else None,
            gapRel=self.m_gap,
        )

        status = model.solve(solver)
        status_str = pulp.LpStatus[status]
        if status_str != "Optimal":
            return False

        obj_val = pulp.value(model.objective)
        if obj_val is not None:
            print(f"Minimume Cost = {obj_val / self.m_pcn}")

        self.m_agent_paths = [[] for _ in range(num_agents)]

        for a in range(num_agents):
            min_dist = self.m_kappa
            for i in range(num_nodes):
                val = pulp.value(x[(a, i)])
                if val is not None and val > 0.5:
                    self.m_agent_paths[a].append((i, i))
                    dist = self._distance(self.m_agents[a], self.m_nodes[i])
                    if dist < min_dist:
                        min_dist = dist
                        self.m_depots[a] = i

        return True

    # ------------------------------------------------------------------
    # plan_agent  (linearisiert)
    # ------------------------------------------------------------------
    """"
    def plan_agent(self, agent: int) -> bool:
        if agent < 0 or agent >= len(self.m_agent_paths):
            raise IndexError("Invalid agent index.")

        local_pairs = self.m_agent_paths[agent]
        n = len(local_pairs)

        if n == 0:
            return False
        if n == 1:
            node_idx = local_pairs[0][0]
            self.m_depots[agent] = node_idx
            self.m_agent_paths[agent][0] = (node_idx, node_idx)
            return True

        local_to_global = [pair[0] for pair in local_pairs]
        depot_global = self.m_depots[agent]

        max_dist = 1.5 * self.m_res

        dist_node_node: List[List[float]] = [[0.0] * n for _ in range(n)]
        turn_node_node_node: List[List[List[float]]] = [[[0.0] * n for _ in range(n)] for _ in range(n)]

        for i in range(n):
            for j in range(n):
                dist = self._distance(self.m_nodes[local_to_global[i]], self.m_nodes[local_to_global[j]])
                if dist == 0.0 or dist > max_dist:
                    dist_node_node[i][j] = self.m_kappa
                else:
                    dist_node_node[i][j] = self.m_pcn * dist

        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if (
                        dist_node_node[i][j] > self.m_pcn * max_dist
                        or dist_node_node[j][k] > self.m_pcn * max_dist
                    ):
                        turn_node_node_node[i][j][k] = self.m_kappa
                    else:
                        turn = self._turn_angle(
                            self.m_nodes[local_to_global[i]],
                            self.m_nodes[local_to_global[j]],
                            self.m_nodes[local_to_global[k]],
                        )
                        turn_node_node_node[i][j][k] = self.m_pcn * turn

        model = pulp.LpProblem(f"plan_agent_{agent}", pulp.LpMinimize)

        x = {
            (i, j): pulp.LpVariable(f"x_{i}_{j}", cat="Binary")
            for i in range(n)
            for j in range(n)
            if i != j
        }

        u = {
            i: pulp.LpVariable(f"u_{i}", lowBound=0.0, cat="Continuous")
            for i in range(n)
        }

        y = {
            (i, j, k): pulp.LpVariable(f"y_{i}_{j}_{k}", cat="Binary")
            for i in range(n)
            for j in range(n)
            for k in range(n)
            if i != j and k != j
        }

        total_dist = pulp.lpSum(
            dist_node_node[i][j] * x[(i, j)]
            for i in range(n)
            for j in range(n)
            if i != j
        )

        total_turn_terms = []
        for i in range(n):
            for j in range(n):
                if i == j or local_to_global[j] == depot_global:
                    continue
                for k in range(n):
                    if k == j:
                        continue
                    total_turn_terms.append(turn_node_node_node[i][j][k] * y[(i, j, k)])

        total_turn = pulp.lpSum(total_turn_terms)

        model += self.m_lambda * total_dist + self.m_gamma * total_turn

        for j in range(n):
            model += (
                pulp.lpSum(x[(i, j)] for i in range(n) if i != j) == 1,
                f"flow_in_{j}",
            )

        for i in range(n):
            model += (
                pulp.lpSum(x[(i, j)] for j in range(n) if j != i) == 1,
                f"flow_out_{i}",
            )

        for i in range(n):
            if local_to_global[i] == depot_global:
                continue
            for j in range(n):
                if local_to_global[j] == depot_global or j == i:
                    continue
                model += (
                    u[i] - u[j] + n * x[(i, j)] <= n - 1,
                    f"mtz_{i}_{j}",
                )

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                for k in range(n):
                    if k == j:
                        continue
                    model += y[(i, j, k)] <= x[(i, j)]
                    model += y[(i, j, k)] <= x[(j, k)]
                    model += y[(i, j, k)] >= x[(i, j)] + x[(j, k)] - 1

        solver = pulp.PULP_CBC_CMD(
            msg=False,
            timeLimit=self.m_limit if self.m_limit < 1e15 else None,
            gapRel=self.m_gap,
        )

        status = model.solve(solver)
        status_str = pulp.LpStatus[status]
        if status_str != "Optimal":
            return False

        obj_val = pulp.value(model.objective)
        if obj_val is None or (obj_val / self.m_pcn) >= self.m_kappa:
            return False

        print(f"Minimume Cost = {obj_val / self.m_pcn}")

        updated_pairs: List[Tuple[int, int]] = []
        for i in range(n):
            successor_global: Optional[int] = None
            for j in range(n):
                if i == j:
                    continue
                val = pulp.value(x[(i, j)])
                if val is not None and val > 0.5:
                    successor_global = local_to_global[j]
                    break

            if successor_global is None:
                return False

            updated_pairs.append((local_to_global[i], successor_global))

        self.m_agent_paths[agent] = updated_pairs
        return True
    """

    def plan_agent_quadratic(self, agent: int) -> bool:
        """
        Quadratische SCIP-Version von Subproblem 2 (MEPP) via Epigraph-Reformulierung.

        SCIP/PySCIPOpt unterstützt keine nichtlinearen Objectives direkt.
        Daher modellieren wir:

            min z
            s.t. z >= lambda * sum(d_ij x_ij) + gamma * sum(q_ijk x_ij x_jk)

        Das ist äquivalent zum ursprünglichen Minimierungsproblem.
        """
        if agent < 0 or agent >= len(self.m_agent_paths):
            raise IndexError("Invalid agent index.")

        local_pairs = self.m_agent_paths[agent]
        n = len(local_pairs)

        if n == 0:
            return False

        if n == 1:
            node_idx = local_pairs[0][0]
            self.m_depots[agent] = node_idx
            self.m_agent_paths[agent][0] = (node_idx, node_idx)
            print("Minimume Cost = 0.0")
            return True

        local_to_global = [pair[0] for pair in local_pairs]
        depot_global = self.m_depots[agent]

        max_dist = 1.5 * self.m_res

        dist_node_node: List[List[float]] = [[0.0] * n for _ in range(n)]
        turn_node_node_node: List[List[List[float]]] = [[[0.0] * n for _ in range(n)] for _ in range(n)]

        # Distanzmatrix
        for i in range(n):
            for j in range(n):
                dist = self._distance(
                    self.m_nodes[local_to_global[i]],
                    self.m_nodes[local_to_global[j]]
                )
                if dist == 0.0 or dist > max_dist:
                    dist_node_node[i][j] = self.m_kappa
                else:
                    dist_node_node[i][j] = self.m_pcn * dist

        # Turnmatrix
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if (
                        dist_node_node[i][j] > self.m_pcn * max_dist
                        or dist_node_node[j][k] > self.m_pcn * max_dist
                    ):
                        turn_node_node_node[i][j][k] = self.m_kappa
                    else:
                        turn = self._turn_angle(
                            self.m_nodes[local_to_global[i]],
                            self.m_nodes[local_to_global[j]],
                            self.m_nodes[local_to_global[k]],
                        )
                        turn_node_node_node[i][j][k] = self.m_pcn * turn

        model = SCIPModel(f"plan_agent_quadratic_{agent}")

        # Solver-Parameter
        model.hideOutput()   # zum Debuggen auskommentieren
        if self.m_limit < 1e15:
            model.setRealParam("limits/time", float(self.m_limit))
        if self.m_gap is not None:
            model.setRealParam("limits/gap", float(self.m_gap))

        # Binärvariablen x_ij
        x = {}
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                x[(i, j)] = model.addVar(vtype="B", name=f"x_{i}_{j}")

        # MTZ-Variablen
        u = {}
        for i in range(n):
            u[i] = model.addVar(vtype="C", lb=0.0, name=f"u_{i}")

        # Epigraph-Variable für die Objective
        z_obj = model.addVar(vtype="C", lb=0.0, name="z_obj")

        # Linearer Distanzterm
        total_dist = quicksum(
            dist_node_node[i][j] * x[(i, j)]
            for i in range(n)
            for j in range(n)
            if i != j
        )

        # Quadratischer Turnterm
        total_turn = quicksum(
            turn_node_node_node[i][j][k] * x[(i, j)] * x[(j, k)]
            for i in range(n)
            for j in range(n)
            if i != j and local_to_global[j] != depot_global
            for k in range(n)
            if k != j #and (j, k) in x
        )

        nl_obj_expr = self.m_lambda * total_dist + self.m_gamma * total_turn

        # Epigraph-Constraint statt nonlinear objective
        model.addCons(z_obj >= nl_obj_expr, name="epigraph_obj")

        # Jetzt lineares Objective
        model.setObjective(z_obj, "minimize")

        # Genau ein Eingang pro Knoten
        for j in range(n):
            model.addCons(
                quicksum(x[(i, j)] for i in range(n) if i != j) == 1,
                name=f"flow_in_{j}"
            )

        # Genau ein Ausgang pro Knoten
        for i in range(n):
            model.addCons(
                quicksum(x[(i, j)] for j in range(n) if j != i) == 1,
                name=f"flow_out_{i}"
            )

        # MTZ-Constraints
        for i in range(n):
            if local_to_global[i] == depot_global:
                continue
            for j in range(n):
                if local_to_global[j] == depot_global or j == i:
                    continue
                model.addCons(
                    u[i] - u[j] + n * x[(i, j)] <= n - 1,
                    name=f"mtz_{i}_{j}"
                )

        model.optimize()

        status = model.getStatus()
        if status not in {"optimal", "timelimit", "gaplimit", "bestsollimit"}:
            print(f"SCIP status: {status}")
            return False

        try:
            obj_val = model.getVal(z_obj)
        except Exception:
            return False

        if obj_val is None or (obj_val / self.m_pcn) >= self.m_kappa:
            return False

        print(f"Minimume Cost = {obj_val / self.m_pcn}")

        updated_pairs: List[Tuple[int, int]] = []
        for i in range(n):
            successor_global: Optional[int] = None
            for j in range(n):
                if i == j:
                    continue
                val = model.getVal(x[(i, j)])
                if val is not None and val > 0.5:
                    successor_global = local_to_global[j]
                    break

            if successor_global is None:
                return False

            updated_pairs.append((local_to_global[i], successor_global))

        self.m_agent_paths[agent] = updated_pairs
        return True

    # ------------------------------------------------------------------
    # validate_path
    # ------------------------------------------------------------------
    def validate_path(self, agent: int) -> bool:
        total_dist = 0.0
        total_turn = 0.0

        ang1 = 0
        ang2 = 0
        ang3 = 0

        depot = self.m_depots[agent]
        path_pairs = self.m_agent_paths[agent]
        successor_map = {src: dst for src, dst in path_pairs}

        i = depot
        j = successor_map.get(i, depot)
        k = depot

        max_dist = 1.5 * self.m_res
        visited_guard = 0
        max_iters = len(path_pairs) + 2

        while True:
            visited_guard += 1
            if visited_guard > max_iters:
                return False

            dist = self._distance(self.m_nodes[i], self.m_nodes[j])
            if dist > max_dist:
                print(f"[VALIDATION ERROR] Agent {agent}: invalid edge from {i} to {j}")
                print(f"  node {i}: {self.m_nodes[i]}")
                print(f"  node {j}: {self.m_nodes[j]}")
                print(f"  distance = {dist}, max_dist = {max_dist}")
                return False

            total_dist += dist

            if j == depot:
                break

            if j not in successor_map:
                return False

            k = successor_map[j]

            r = self._distance(self.m_nodes[i], self.m_nodes[j])
            e = self._distance(self.m_nodes[j], self.m_nodes[k])
            s = self._distance(self.m_nodes[k], self.m_nodes[i])

            if r <= 0.0 or e <= 0.0:
                return False

            t = (r * r + e * e - s * s) / (2.0 * r * e)
            if t > 1.0:
                t = 1.0
            elif t < -1.0:
                t = -1.0

            turn = pi - acos(t)
            total_turn += turn

            if (pi / 4.0 - pi / 8.0) < turn < (pi / 4.0 + pi / 8.0):
                ang1 += 1
            elif (pi / 2.0 - pi / 8.0) < turn < (pi / 2.0 + pi / 8.0):
                ang2 += 1
            elif (3.0 * pi / 4.0 - pi / 8.0) < turn < (3.0 * pi / 4.0 + pi / 8.0):
                ang3 += 1

            i = j
            j = k

        print(
            f"Total Distance: {total_dist} | "
            f"Number of 45' Turn: {ang1} | "
            f"Number of 90' Turn: {ang2} | "
            f"Number of 135' Turn: {ang3}"
        )
        print(f"Total Cost: {self.m_lambda * total_dist + self.m_gamma * total_turn}")

        return True

    # ------------------------------------------------------------------
    # build_mission
    # ------------------------------------------------------------------
    def build_mission(self, agent: int) -> bool:
        try:
            depot = self.m_depots[agent]
            successor_map = {src: dst for src, dst in self.m_agent_paths[agent]}

            ordered_nodes = [depot]
            current = depot
            guard = 0
            max_iters = len(self.m_agent_paths[agent]) + 2

            while True:
                guard += 1
                if guard > max_iters:
                    raise RuntimeError("Mission export failed due to cycle inconsistency.")

                nxt = successor_map[current]
                ordered_nodes.append(nxt)
                current = nxt
                if current == depot:
                    break

            mission = {
                "agent": agent,
                "depot_index": depot,
                "depot_point": list(self.m_nodes[depot]),
                "tour_node_indices": ordered_nodes,
                "tour_points": [list(self.m_nodes[idx]) for idx in ordered_nodes],
            }

            out_file = f"mission_{agent}.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(mission, f, indent=2)

            return True

        except Exception:
            return False

    # ------------------------------------------------------------------
    # visualize_tours
    # ------------------------------------------------------------------
    def visualize_tours(self, save_path: Optional[str] = None, show: bool = True) -> None:
        """
        Visualisiert die Touren aller Agenten in verschiedenen Farben.
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 8))

        if self.m_areas:
            outer = self.m_areas[0]
            ax.plot(
                [p[0] for p in outer],
                [p[1] for p in outer],
                linewidth=2,
                color="black",
                label="Area"
            )

        for obs in self.m_areas[1:]:
            ax.plot(
                [p[0] for p in obs],
                [p[1] for p in obs],
                linestyle="--",
                linewidth=1.5,
                color="black"
            )

        if not self.m_agent_paths:
            print("Keine Agentenpfade vorhanden.")
            if show:
                plt.show()
            return

        colors = plt.cm.tab10.colors

        for agent in range(len(self.m_agent_paths)):
            if not self.m_agent_paths[agent]:
                print(f"Agent {agent}: kein Pfad vorhanden.")
                continue

            depot = self.m_depots[agent]
            successor_map = {src: dst for src, dst in self.m_agent_paths[agent]}

            ordered_nodes = [depot]
            current = depot
            visited = {depot}

            while True:
                if current not in successor_map:
                    print(f"Agent {agent}: successor für Knoten {current} fehlt.")
                    break

                nxt = successor_map[current]
                ordered_nodes.append(nxt)

                if nxt == depot:
                    break

                if nxt in visited:
                    print(f"Agent {agent}: Zyklusfehler bei Knoten {nxt}.")
                    break

                visited.add(nxt)
                current = nxt

            tour_points = [self.m_nodes[idx] for idx in ordered_nodes]
            xs = [p[0] for p in tour_points]
            ys = [p[1] for p in tour_points]

            color = colors[agent % len(colors)]

            ax.plot(xs, ys, color=color, linewidth=2.5, label=f"Agent {agent}")
            ax.scatter(xs, ys, color=color, s=30)

            depot_point = self.m_nodes[depot]
            ax.scatter(
                [depot_point[0]],
                [depot_point[1]],
                color=color,
                s=120,
                marker="s",
                edgecolors="black",
                zorder=5
            )

        ax.set_title("Agent Tours")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        ax.legend()

        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)
   