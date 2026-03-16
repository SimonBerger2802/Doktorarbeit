from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from random import Random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

Node = int
Edge = Tuple[int, int]


def norm_edge(a: int, b: int) -> Edge:
    return (a, b) if a < b else (b, a)


@dataclass(frozen=True)
class Point:
    x: float
    y: float


class LinKernighan1973Section1:
    """
    Paper-oriented implementation of the *basic* Lin-Kernighan algorithm from
    Section 1 of Lin & Kernighan (1973).

    This code intentionally follows the seven-step algorithmic outline from the
    paper and *does not* include the later refinements from Section 2 onward.

    Important honesty note:
    ----------------------
    The paper gives the basic procedure at a high level. Some operational
    details are still left to the implementer, for example:
      - tie-breaking among equal-length choices,
      - exact bookkeeping representation,
      - practical limits on the number of y1 / y2 contenders,
      - how to operationalize the special alternate-x2 case.

    So this class is as close as practical to the Section-1 prescription, but
    no executable implementation can be claimed to be a perfectly literal copy
    of the paper because the paper is not pseudocode-complete.

    Defaults chosen from the paper text:
      - random starting tour(s)
      - limited backtracking only at levels 1 and 2
      - at most five contenders for y1 and y2 (paper notes this was their
        current practice in experiments)
    """

    def __init__(
        self,
        points: Sequence[Tuple[float, float] | Point],
        *,
        seed: Optional[int] = None,
        max_y1_choices: int = 5,
        max_y2_choices: int = 5,
        random_restarts: int = 1,
        start_tour: Optional[Sequence[int]] = None,
    ) -> None:
        self.points: List[Point] = [p if isinstance(p, Point) else Point(*p) for p in points]
        self.n = len(self.points)
        if self.n < 4:
            raise ValueError("At least 4 cities are required.")

        self.rng = Random(seed)
        self.dist = self._build_distance_matrix()
        self.max_y1_choices = max(1, max_y1_choices)
        self.max_y2_choices = max(1, max_y2_choices)
        self.random_restarts = max(1, random_restarts)

        if start_tour is None:
            self.tour = self._random_tour()
        else:
            self.tour = list(start_tour)
            self._validate_tour(self.tour)

    # ------------------------------------------------------------------
    # Geometry / tour basics
    # ------------------------------------------------------------------
    def _build_distance_matrix(self) -> List[List[float]]:
        d = [[0.0] * self.n for _ in range(self.n)]
        for i in range(self.n):
            for j in range(i + 1, self.n):
                dij = hypot(self.points[i].x - self.points[j].x, self.points[i].y - self.points[j].y)
                d[i][j] = d[j][i] = dij
        return d

    def _validate_tour(self, tour: Sequence[int]) -> None:
        if len(tour) != self.n:
            raise ValueError("Tour has wrong length.")
        if set(tour) != set(range(self.n)):
            raise ValueError("Tour must contain each city exactly once.")

    def _random_tour(self) -> List[int]:
        tour = list(range(self.n))
        self.rng.shuffle(tour)
        return tour

    def tour_length(self, tour: Optional[Sequence[int]] = None) -> float:
        t = self.tour if tour is None else tour
        return sum(self.dist[t[i]][t[(i + 1) % self.n]] for i in range(self.n))

    def _tour_edges(self, tour: Optional[Sequence[int]] = None) -> set[Edge]:
        t = self.tour if tour is None else tour
        return {norm_edge(t[i], t[(i + 1) % len(t)]) for i in range(len(t))}

    def _tour_neighbors(self, tour: Sequence[int]) -> Dict[Node, Tuple[Node, Node]]:
        n = len(tour)
        out: Dict[Node, Tuple[Node, Node]] = {}
        for i, node in enumerate(tour):
            out[node] = (tour[(i - 1) % n], tour[(i + 1) % n])
        return out

    def _build_adj(self, edges: Iterable[Edge]) -> Dict[Node, List[Node]]:
        adj = {i: [] for i in range(self.n)}
        for a, b in edges:
            adj[a].append(b)
            adj[b].append(a)
        return adj

    def _edges_to_tour(self, edges: set[Edge]) -> Optional[List[int]]:
        if len(edges) != self.n:
            return None
        adj = self._build_adj(edges)
        if any(len(adj[v]) != 2 for v in adj):
            return None

        start = 0
        prev: Optional[int] = None
        cur = start
        tour = [start]
        for _ in range(self.n - 1):
            a, b = adj[cur]
            nxt = a if a != prev else b
            if nxt == start:
                return None
            tour.append(nxt)
            prev, cur = cur, nxt
        if start not in adj[cur]:
            return None
        if len(set(tour)) != self.n:
            return None
        return tour

    def _edge_len(self, e: Edge) -> float:
        return self.dist[e[0]][e[1]]

    # ------------------------------------------------------------------
    # Exchange helpers
    # ------------------------------------------------------------------
    def _make_tour_from_exchange(
        self,
        base_edges: set[Edge],
        broken: set[Edge],
        joined: set[Edge],
        close_edge: Optional[Edge] = None,
    ) -> Optional[List[int]]:
        if not broken.issubset(base_edges):
            return None
        edges = set(base_edges)
        edges.difference_update(broken)
        edges.update(joined)
        if close_edge is not None:
            edges.add(close_edge)
        return self._edges_to_tour(edges)

    def _g(self, x: Edge, y: Edge) -> float:
        return self._edge_len(x) - self._edge_len(y)

    # ------------------------------------------------------------------
    # Step 4(a): feasible x_i choices
    # ------------------------------------------------------------------
    def _feasible_x_choices(
        self,
        base_edges: set[Edge],
        neighbors: Dict[Node, Tuple[Node, Node]],
        broken: set[Edge],
        joined: set[Edge],
        t_odd: Node,
        t1: Node,
    ) -> List[Tuple[Node, Edge]]:
        """
        Return x_i = (t_{2i-1}, t_{2i}) satisfying Step 4(a):
        joining t_{2i} to t1 closes to a tour.
        """
        out: List[Tuple[Node, Edge]] = []
        for t_even in neighbors[t_odd]:
            x_i = norm_edge(t_odd, t_even)
            if x_i in broken or x_i in joined:
                continue
            close_edge = norm_edge(t_even, t1)
            trial = self._make_tour_from_exchange(base_edges, broken | {x_i}, joined, close_edge)
            if trial is not None:
                out.append((t_even, x_i))
        return out

    def _all_x_choices_ignore_feasibility(
        self,
        neighbors: Dict[Node, Tuple[Node, Node]],
        broken: set[Edge],
        joined: set[Edge],
        t_odd: Node,
    ) -> List[Tuple[Node, Edge]]:
        """
        Used only for Step 6(b): alternate x2 may temporarily violate feasibility.
        """
        out: List[Tuple[Node, Edge]] = []
        for t_even in neighbors[t_odd]:
            x_i = norm_edge(t_odd, t_even)
            if x_i in broken or x_i in joined:
                continue
            out.append((t_even, x_i))
        return out

    # ------------------------------------------------------------------
    # Step 4(e): chosen y_i must permit x_{i+1}
    # ------------------------------------------------------------------
    def _permits_next_x(
        self,
        base_edges: set[Edge],
        neighbors: Dict[Node, Tuple[Node, Node]],
        broken: set[Edge],
        joined: set[Edge],
        endpoint: Node,
        t1: Node,
        *,
        allow_nonfeasible_alternate_x2: bool = False,
    ) -> bool:
        # Normal feasible continuation.
        if self._feasible_x_choices(base_edges, neighbors, broken, joined, endpoint, t1):
            return True
        # Special Section-1 permission: only at i = 2, allow alternate x2 even if not feasible.
        if allow_nonfeasible_alternate_x2:
            for _, x_alt in self._all_x_choices_ignore_feasibility(neighbors, broken, joined, endpoint):
                if x_alt not in self._tour_edges(self.tour):
                    continue
                return True
        return False

    # ------------------------------------------------------------------
    # Step 4(b,c,d,e): enumerate y_i in increasing length
    # ------------------------------------------------------------------
    def _y_choices(
        self,
        base_edges: set[Edge],
        neighbors: Dict[Node, Tuple[Node, Node]],
        broken: set[Edge],
        joined: set[Edge],
        t_even: Node,
        x_i: Edge,
        G_prev: float,
        t1: Node,
        *,
        permit_nonfeasible_next_x: bool = False,
        forbid_join_to_t1: bool = False,
    ) -> List[Tuple[Node, Edge, float]]:
        out: List[Tuple[Node, Edge, float]] = []
        for t_next in range(self.n):
            if t_next == t_even:
                continue
            if forbid_join_to_t1 and t_next == t1:
                continue
            y_i = norm_edge(t_even, t_next)
            if y_i in base_edges:
                continue
            if y_i in joined or y_i in broken:
                continue

            G_i = G_prev + self._g(x_i, y_i)
            if G_i <= 0.0:  # Step 4(d)
                continue

            joined2 = joined | {y_i}
            if not self._permits_next_x(
                base_edges,
                neighbors,
                broken,
                joined2,
                t_next,
                t1,
                allow_nonfeasible_alternate_x2=permit_nonfeasible_next_x,
            ):
                continue
            out.append((t_next, y_i, G_i))

        out.sort(key=lambda item: self._edge_len(item[1]))
        return out

    # ------------------------------------------------------------------
    # Section-1 special alternate x2 case (Fig. 4)
    # ------------------------------------------------------------------
    def _alternate_x2_branch(
        self,
        base_edges: set[Edge],
        neighbors: Dict[Node, Tuple[Node, Node]],
        t1: Node,
        x1: Edge,
        t3: Node,
        y1: Edge,
    ) -> Tuple[float, Optional[List[int]]]:
        """
        Implements Step 6(b) plus the special Fig. 4 discussion for the
        nonfeasible alternate x2 allowed only at i = 2.
        """
        broken = {x1}
        joined = {y1}
        G_prev = self._g(x1, y1)
        G_star = 0.0
        best_tour: Optional[List[int]] = None

        feasible_x2 = {x for _, x in self._feasible_x_choices(base_edges, neighbors, broken, joined, t3, t1)}
        alt_x2_choices = [(t4, x2) for t4, x2 in self._all_x_choices_ignore_feasibility(neighbors, broken, joined, t3) if x2 not in feasible_x2]
        if not alt_x2_choices:
            return 0.0, None

        # Paper discusses the alternate choice in singular form; there is effectively one alternate x2.
        t4, x2 = alt_x2_choices[0]
        broken2 = broken | {x2}

        # In Fig. 4(a), y2 must not join to t1.
        y2_choices = self._y_choices(
            base_edges,
            neighbors,
            broken2,
            joined,
            t4,
            x2,
            G_prev,
            t1,
            permit_nonfeasible_next_x=False,
            forbid_join_to_t1=True,
        )[: self.max_y2_choices]

        for t5, y2, G2 in y2_choices:
            joined2 = joined | {y2}

            # After this point, the paper says we return to the normal course of Step 4.
            # We therefore continue from node t5 with normal Step-4 logic.
            g_star, tour_star = self._continue_normal_step4(
                base_edges,
                neighbors,
                t1,
                broken2,
                joined2,
                G2,
                t5,
                start_i=3,
            )
            if g_star > G_star:
                G_star = g_star
                best_tour = tour_star

        return G_star, best_tour

    # ------------------------------------------------------------------
    # Normal Step 4/5 continuation
    # ------------------------------------------------------------------
    def _continue_normal_step4(
        self,
        base_edges: set[Edge],
        neighbors: Dict[Node, Tuple[Node, Node]],
        t1: Node,
        broken: set[Edge],
        joined: set[Edge],
        G_prev: float,
        t_odd: Node,
        *,
        start_i: int,
        forced_y_rank_at_i2: Optional[int] = None,
    ) -> Tuple[float, Optional[List[int]]]:
        G_star = 0.0
        best_tour: Optional[List[int]] = None
        i = start_i

        while True:
            x_choices = self._feasible_x_choices(base_edges, neighbors, broken, joined, t_odd, t1)
            if not x_choices:
                break

            # In the normal sequential case the paper states x_i is uniquely determined.
            t_even, x_i = x_choices[0]
            broken2 = broken | {x_i}

            # Step 4(f): close-up before constructing y_i.
            y_star = norm_edge(t_even, t1)
            close_gain = G_prev + self._g(x_i, y_star)
            close_tour = self._make_tour_from_exchange(base_edges, broken2, joined, y_star)
            if close_tour is not None and close_gain > G_star:
                G_star = close_gain
                best_tour = close_tour

            y_choices = self._y_choices(
                base_edges,
                neighbors,
                broken2,
                joined,
                t_even,
                x_i,
                G_prev,
                t1,
            )
            if not y_choices:
                break

            if i == 2 and forced_y_rank_at_i2 is not None:
                if forced_y_rank_at_i2 >= len(y_choices):
                    break
                t_next, y_i, G_i = y_choices[forced_y_rank_at_i2]
            else:
                t_next, y_i, G_i = y_choices[0]

            # Step 5 stopping rule.
            if G_i <= G_star:
                break

            broken = broken2
            joined = joined | {y_i}
            G_prev = G_i
            t_odd = t_next
            i += 1

        return G_star, best_tour

    # ------------------------------------------------------------------
    # Fixed x1,y1 branch: Step 4/5 plus Step 6(a)
    # ------------------------------------------------------------------
    def _run_fixed_x1_y1(
        self,
        base_tour: List[int],
        base_edges: set[Edge],
        neighbors: Dict[Node, Tuple[Node, Node]],
        t1: Node,
        x1: Edge,
        t3: Node,
        y1: Edge,
        *,
        y2_rank: Optional[int],
    ) -> Tuple[float, Optional[List[int]]]:
        _ = base_tour
        broken = {x1}
        joined = {y1}
        G1 = self._g(x1, y1)
        return self._continue_normal_step4(
            base_edges,
            neighbors,
            t1,
            broken,
            joined,
            G1,
            t3,
            start_i=2,
            forced_y_rank_at_i2=y2_rank,
        )

    # ------------------------------------------------------------------
    # Step 2..7 for a fixed t1
    # ------------------------------------------------------------------
    def _try_t1(self, t1_index: int) -> bool:
        base_tour = list(self.tour)
        base_edges = self._tour_edges(base_tour)
        neighbors = self._tour_neighbors(base_tour)
        t1 = base_tour[t1_index]

        # Step 2 / Step 6(d): try both adjacent x1 choices.
        t2_options = [neighbors[t1][1], neighbors[t1][0]]
        for t2 in t2_options:
            x1 = norm_edge(t1, t2)

            # Step 3 / Step 6(c): y1 choices in increasing length, with g1 > 0.
            y1_choices: List[Tuple[Node, Edge, float]] = []
            for t3 in range(self.n):
                if t3 in (t1, t2):
                    continue
                y1 = norm_edge(t2, t3)
                if y1 in base_edges or y1 == x1:
                    continue
                g1 = self._g(x1, y1)
                if g1 <= 0.0:
                    continue
                y1_choices.append((t3, y1, g1))
            y1_choices.sort(key=lambda item: self._edge_len(item[1]))
            y1_choices = y1_choices[: self.max_y1_choices]
            if not y1_choices:
                continue

            for t3, y1, _ in y1_choices:
                # Step 6(a): y2 contenders in increasing length.
                for y2_rank in range(self.max_y2_choices):
                    G_star, tprime = self._run_fixed_x1_y1(
                        base_tour,
                        base_edges,
                        neighbors,
                        t1,
                        x1,
                        t3,
                        y1,
                        y2_rank=y2_rank,
                    )
                    if tprime is not None and G_star > 0.0:
                        self.tour = tprime
                        return True
                    if G_star <= 0.0 and tprime is None:
                        break

                # Step 6(b): alternate x2.
                G_star_alt, tprime_alt = self._alternate_x2_branch(base_edges, neighbors, t1, x1, t3, y1)
                if tprime_alt is not None and G_star_alt > 0.0:
                    self.tour = tprime_alt
                    return True

        # Step 6(e): caller moves to next t1.
        return False

    # ------------------------------------------------------------------
    # Public solve loop
    # ------------------------------------------------------------------
    def solve(self, max_passes_per_restart: int = 10_000) -> List[int]:
        best_global_tour = list(self.tour)
        best_global_length = self.tour_length(best_global_tour)

        for restart in range(self.random_restarts):
            if restart > 0:
                self.tour = self._random_tour()  # Step 1 again.

            passes = 0
            while passes < max_passes_per_restart:
                passes += 1
                improved = False

                # Step 7: all n values of t1 examined without profit.
                for t1_index in range(self.n):
                    if self._try_t1(t1_index):
                        improved = True
                        break  # Step 5: restart from Step 2 with T'.

                if not improved:
                    break

            cur_len = self.tour_length()
            if cur_len < best_global_length:
                best_global_length = cur_len
                best_global_tour = list(self.tour)

        self.tour = best_global_tour
        return self.tour


import random

if __name__ == "__main__":

    random.seed(42)

    pts = [(random.uniform(0, 200), random.uniform(0, 200)) for _ in range(200)]

    lk = LinKernighan1973Section1(pts, seed=42, random_restarts=3)

    print("Initial tour:", lk.tour)
    print("Initial length:", round(lk.tour_length(), 4))

    lk.solve()

    print("Improved tour:", lk.tour)
    print("Improved length:", round(lk.tour_length(), 4))
