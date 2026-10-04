"""Autonomous viewpoint selection in a *known simulated* 3-D room.

This is a simulator, not a flight controller or a reconstruction benchmark.
Both navigation and predicted observation gain use simulator ground truth: a
6 x 5 x 3 metre room and one box. Observations are sampled visible surfaces,
with perspective field of view, limited range, and box occlusion. They are
not SLAM estimates. No video socket, aircraft API, or control transport is used.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math

import numpy as np


ROOM_SIZE = np.array([6.0, 5.0, 3.0])
BOX_MIN = np.array([2.55, 1.85, 0.0])
BOX_MAX = np.array([3.45, 3.15, 1.65])


@dataclass(frozen=True)
class SurveySimulationConfig:
    """Bounded simulator settings; all lengths here are simulated metres."""

    surface_spacing: float = 0.4
    body_clearance: float = 0.25
    max_step_distance: float = 0.35
    observation_range: float = 4.5
    horizontal_fov_degrees: float = 75.0
    vertical_fov_degrees: float = 55.0
    target_coverage: float = 0.95
    max_steps: int = 1200

    def __post_init__(self) -> None:
        ranges = {
            "surface_spacing": (0.25, 0.8),
            "body_clearance": (0.05, 0.45),
            "max_step_distance": (0.05, 0.75),
            "observation_range": (0.1, 12.0),
            "horizontal_fov_degrees": (10.0, 150.0),
            "vertical_fov_degrees": (10.0, 150.0),
            "target_coverage": (0.01, 1.0),
        }
        for name, (low, high) in ranges.items():
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if not low <= value <= high:
                raise ValueError(f"{name} must be between {low} and {high}")
        if isinstance(self.max_steps, bool) or not isinstance(self.max_steps, int):
            raise ValueError("max_steps must be an integer")
        if not 1 <= self.max_steps <= 5000:
            raise ValueError("max_steps must be between 1 and 5000")


def _box_intersections(
    origin: np.ndarray, destinations: np.ndarray, low: np.ndarray, high: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return parametric entry/exit of finite rays using the slab method."""
    directions = destinations - origin
    enter = np.full(len(destinations), -np.inf)
    leave = np.full(len(destinations), np.inf)
    for axis in range(3):
        parallel = np.abs(directions[:, axis]) < 1e-12
        a = np.zeros(len(destinations))
        b = np.zeros(len(destinations))
        np.divide(low[axis] - origin[axis], directions[:, axis], out=a, where=~parallel)
        np.divide(high[axis] - origin[axis], directions[:, axis], out=b, where=~parallel)
        slab_enter = np.minimum(a, b)
        slab_leave = np.maximum(a, b)
        slab_enter[parallel] = -np.inf
        slab_leave[parallel] = np.inf
        if origin[axis] < low[axis] or origin[axis] > high[axis]:
            slab_enter[parallel] = np.inf
            slab_leave[parallel] = -np.inf
        enter = np.maximum(enter, slab_enter)
        leave = np.minimum(leave, slab_leave)
    return enter, leave


class RoomSurveySimulation:
    """Choose informative 3-D viewpoints and move through known free space.

    ``step`` advances at most ``max_step_distance`` and returns a detached,
    JSON-compatible snapshot. All history, surface samples, and candidate
    counts are bounded. ``complete`` means the requested fraction of sampled
    simulator surfaces was observed; it does not assert complete reconstruction
    of a real room. Exhausted candidates and step budgets are explicit states.
    """

    def __init__(self, seed: int = 0, config: SurveySimulationConfig | None = None):
        self.config = config or SurveySimulationConfig()
        self._points, self._normals = self._sample_surfaces()
        self._seen = np.zeros(len(self._points), dtype=bool)
        self._position = np.array([0.75, 0.75, 1.4])
        self._forward = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
        self._nodes = self._make_nodes()
        self._edges = self._make_edges()
        self._views: list[tuple[int, np.ndarray, np.ndarray]] = []
        for node_id, position in enumerate(self._nodes):
            for pitch in (-30.0, 0.0, 30.0):
                for yaw in range(0, 360, 45):
                    p, y = math.radians(pitch), math.radians(yaw)
                    forward = np.array([math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), math.sin(p)])
                    visible = self._visible_indices(position, forward)
                    self._views.append((node_id, forward, visible))
        self._view_order = np.random.default_rng(seed).permutation(len(self._views))
        self._visited: set[int] = set()
        self._active_view: int | None = None
        self._route: list[np.ndarray] = []
        self._keyframes: list[dict] = []
        self._path = [self._position.tolist()]
        self._steps = 0
        self._state = "running"
        self._message = "Simulation ready: geometry and distances are known to the simulator."

    def _sample_surfaces(self) -> tuple[np.ndarray, np.ndarray]:
        points: list[np.ndarray] = []
        normals: list[np.ndarray] = []

        def face(low: np.ndarray, high: np.ndarray, axis: int, side: int, normal_sign: int) -> None:
            other = [i for i in range(3) if i != axis]
            # Interior face samples avoid ambiguous coincident corner normals.
            grids = [np.linspace(low[i] + 0.05, high[i] - 0.05,
                                 max(2, math.ceil((high[i] - low[i]) / self.config.surface_spacing)))
                     for i in other]
            for u in grids[0]:
                for v in grids[1]:
                    point = np.zeros(3)
                    point[axis] = high[axis] if side else low[axis]
                    point[other] = [u, v]
                    normal = np.zeros(3)
                    normal[axis] = normal_sign
                    points.append(point)
                    normals.append(normal)

        for axis in range(3):
            face(np.zeros(3), ROOM_SIZE, axis, 0, 1)
            face(np.zeros(3), ROOM_SIZE, axis, 1, -1)
        # The floor underneath the opaque box is not part of the observable scene.
        keep = [not (p[2] == 0 and np.all(p[:2] >= BOX_MIN[:2]) and np.all(p[:2] <= BOX_MAX[:2]))
                for p in points]
        points = [p for p, use in zip(points, keep) if use]
        normals = [n for n, use in zip(normals, keep) if use]
        for axis in (0, 1):
            face(BOX_MIN, BOX_MAX, axis, 0, -1)
            face(BOX_MIN, BOX_MAX, axis, 1, 1)
        face(BOX_MIN, BOX_MAX, 2, 1, 1)
        return np.asarray(points), np.asarray(normals)

    def _position_is_clear(self, position: np.ndarray) -> bool:
        margin = self.config.body_clearance
        if np.any(position < margin) or np.any(position > ROOM_SIZE - margin):
            return False
        return not bool(np.all(position >= BOX_MIN - margin) and np.all(position <= BOX_MAX + margin))

    def _segment_is_clear(self, a: np.ndarray, b: np.ndarray) -> bool:
        if not self._position_is_clear(a) or not self._position_is_clear(b):
            return False
        margin = self.config.body_clearance
        enter, leave = _box_intersections(a, b[None, :], BOX_MIN - margin, BOX_MAX + margin)
        intersects = (enter[0] <= leave[0]) and leave[0] >= 0 and enter[0] <= 1
        return not intersects

    def _make_nodes(self) -> np.ndarray:
        return np.array([
            [x, y, z]
            for x in (0.75, 3.0, 5.25)
            for y in (0.75, 2.5, 4.25)
            for z in (0.65, 1.4, 2.3)
            if self._position_is_clear(np.array([x, y, z]))
        ])

    def _make_edges(self) -> list[list[tuple[int, float]]]:
        edges: list[list[tuple[int, float]]] = [[] for _ in self._nodes]
        for i, a in enumerate(self._nodes):
            for j in range(i + 1, len(self._nodes)):
                b = self._nodes[j]
                if self._segment_is_clear(a, b):
                    distance = float(np.linalg.norm(b - a))
                    edges[i].append((j, distance))
                    edges[j].append((i, distance))
        return edges

    def _visible_indices(self, position: np.ndarray, forward: np.ndarray) -> np.ndarray:
        relative = self._points - position
        distance = np.linalg.norm(relative, axis=1)
        depth = relative @ forward
        right = np.cross(forward, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        horizontal = math.tan(math.radians(self.config.horizontal_fov_degrees / 2))
        vertical = math.tan(math.radians(self.config.vertical_fov_degrees / 2))
        mask = ((depth > 0.05) & (distance <= self.config.observation_range)
                & (np.abs(relative @ right) <= depth * horizontal)
                & (np.abs(relative @ up) <= depth * vertical)
                & (np.einsum("ij,ij->i", self._normals, -relative) > 1e-8))
        candidates = np.flatnonzero(mask)
        enter, leave = _box_intersections(position, self._points[candidates], BOX_MIN, BOX_MAX)
        # A first hit exactly at an observed box surface is visible, not an occlusion.
        blocked = (enter <= leave) & (leave > 1e-8) & (enter < 1.0 - 1e-8)
        return candidates[~blocked]

    def _observe(self) -> None:
        self._seen[self._visible_indices(self._position, self._forward)] = True

    def _shortest_routes(self) -> tuple[np.ndarray, dict[int, int | None]]:
        distances = np.full(len(self._nodes), np.inf)
        parents: dict[int, int | None] = {}
        queue: list[tuple[float, int]] = []
        for i, node in enumerate(self._nodes):
            if self._segment_is_clear(self._position, node):
                distances[i] = np.linalg.norm(node - self._position)
                parents[i] = None
                heapq.heappush(queue, (float(distances[i]), i))
        while queue:
            distance, node = heapq.heappop(queue)
            if distance > distances[node] + 1e-12:
                continue
            for neighbor, edge_distance in self._edges[node]:
                alternative = distance + edge_distance
                if alternative < distances[neighbor] - 1e-12:
                    distances[neighbor] = alternative
                    parents[neighbor] = node
                    heapq.heappush(queue, (alternative, neighbor))
        return distances, parents

    def _select_target(self) -> bool:
        distances, parents = self._shortest_routes()
        best: int | None = None
        best_score = -1.0
        for raw_id in self._view_order:
            view_id = int(raw_id)
            if view_id in self._visited:
                continue
            node, direction, visible = self._views[view_id]
            if not np.isfinite(distances[node]):
                continue
            gain = int(np.count_nonzero(~self._seen[visible]))
            if gain == 0:
                continue
            turn = math.acos(float(np.clip(direction @ self._forward, -1, 1)))
            score = gain / (1.0 + 0.25 * distances[node] + 0.15 * turn)
            if score > best_score:
                best, best_score = view_id, score
        if best is None:
            self._state = "no_reachable_view"
            self._message = "Simulation stopped: no reachable candidate view adds unseen surface samples."
            return False
        self._active_view = best
        node = self._views[best][0]
        route = [node]
        while parents[route[-1]] is not None:
            route.append(parents[route[-1]])
        self._route = [self._nodes[i].copy() for i in reversed(route)
                       if np.linalg.norm(self._nodes[i] - self._position) > 1e-9]
        self._message = "Simulation: moving toward the reachable view with greatest gain per travel cost."
        return True

    def _finish_view(self) -> None:
        assert self._active_view is not None
        self._visited.add(self._active_view)
        self._keyframes.append({"id": len(self._keyframes), "position": self._position.tolist(),
                                "forward": self._forward.tolist()})
        self._active_view = None

    def step(self) -> dict:
        if self._state != "running":
            return self.snapshot()
        self._steps += 1
        self._observe()
        if float(np.mean(self._seen)) >= self.config.target_coverage:
            self._state = "complete"
            self._message = f"Simulation complete: observed {float(np.mean(self._seen)):.1%} of sampled simulator surfaces."
            return self.snapshot()
        if self._active_view is None and not self._select_target():
            return self.snapshot()
        assert self._active_view is not None
        self._forward = self._views[self._active_view][1].copy()
        if self._route:
            destination = self._route[0]
            delta = destination - self._position
            distance = float(np.linalg.norm(delta))
            next_position = self._position + delta * min(1.0, self.config.max_step_distance / distance)
            if not self._segment_is_clear(self._position, next_position):
                self._state = "no_reachable_view"
                self._message = "Simulation stopped: the planned movement failed its clearance check."
                return self.snapshot()
            self._position = next_position
            self._path.append(self._position.tolist())
            if distance <= self.config.max_step_distance + 1e-9:
                self._route.pop(0)
        self._observe()
        if not self._route:
            self._finish_view()
        coverage = float(np.mean(self._seen))
        if coverage >= self.config.target_coverage:
            self._state = "complete"
            self._message = f"Simulation complete: observed {coverage:.1%} of sampled simulator surfaces."
        elif self._steps >= self.config.max_steps:
            self._state = "stopped"
            self._message = "Simulation stopped at the step budget before reaching the requested coverage."
        return self.snapshot()

    def stop(self) -> dict:
        if self._state == "running":
            self._state = "stopped"
            self._message = "Simulation stopped by the user."
        return self.snapshot()

    def snapshot(self) -> dict:
        target = (None if self._active_view is None or self._state != "running"
                  else self._nodes[self._views[self._active_view][0]].tolist())
        return {
            "source": "simulation",
            "units": "meters (simulated)",
            "points": [{"id": int(i), "position": self._points[i].tolist()} for i in np.flatnonzero(self._seen)],
            "keyframes": [{"id": k["id"], "position": list(k["position"]), "forward": list(k["forward"])}
                          for k in self._keyframes],
            "camera_position": self._position.tolist(),
            "camera_forward": self._forward.tolist(),
            "target_position": target,
            "path": [list(p) for p in self._path],
            "state": self._state,
            "coverage": float(np.mean(self._seen)),
            "steps": self._steps,
            "message": self._message,
            "flight_commands_sent": 0,
            "surface_samples_total": len(self._points),
            "target_coverage": self.config.target_coverage,
            "planning_assumption": "Known simulator geometry supplies collision checks and predicted visibility.",
            "room_size": ROOM_SIZE.tolist(),
            "obstacle": {"minimum": BOX_MIN.tolist(), "maximum": BOX_MAX.tolist()},
            "body_clearance": self.config.body_clearance,
        }
