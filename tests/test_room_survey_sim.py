"""Geometric and behavioral checks of the simulation, without aircraft APIs."""

import json

import numpy as np
import pytest

from follow_neo.room_survey_sim import RoomSurveySimulation, SurveySimulationConfig


def finish(sim):
    previous_coverage = 0.0
    for _ in range(sim.config.max_steps + 1):
        snapshot = sim.step()
        assert snapshot["coverage"] >= previous_coverage
        previous_coverage = snapshot["coverage"]
        if snapshot["state"] != "running":
            return snapshot
    pytest.fail("simulation did not terminate within its stated step budget")


def test_default_survey_collects_three_dimensional_surfaces_progressively():
    sim = RoomSurveySimulation()
    initial = sim.snapshot()
    assert initial["points"] == []
    first = sim.step()
    assert 0 < first["coverage"] < 0.5
    assert first["state"] == "running"
    result = finish(sim)
    assert result["state"] == "complete"
    assert result["coverage"] >= sim.config.target_coverage
    assert len(result["points"]) > len(first["points"])
    points = np.array([p["position"] for p in result["points"]])
    assert np.all(np.ptp(points, axis=0) > [5.0, 4.0, 2.0])
    # The camera also moves vertically; this is not a planar tour with 3-D dots.
    path = np.array(result["path"])
    assert np.ptp(path[:, 2]) >= 0.7
    assert len(result["keyframes"]) >= 3


def test_all_swept_movements_keep_room_and_obstacle_clearance():
    sim = RoomSurveySimulation()
    result = finish(sim)
    path = np.asarray(result["path"])
    clearance = sim.config.body_clearance
    lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    assert np.all(lengths <= sim.config.max_step_distance + 1e-10)
    # Independently sample each swept segment densely instead of trusting the
    # simulator's slab checker or checking endpoints alone.
    expanded_min = np.array([2.55, 1.85, 0.0]) - clearance
    expanded_max = np.array([3.45, 3.15, 1.65]) + clearance
    for start, end in zip(path[:-1], path[1:]):
        samples = start + np.linspace(0, 1, 201)[:, None] * (end - start)
        assert np.all(samples >= clearance - 1e-10)
        assert np.all(samples <= np.array([6.0, 5.0, 3.0]) - clearance + 1e-10)
        in_box = np.all(samples >= expanded_min, axis=1) & np.all(samples <= expanded_max, axis=1)
        assert not np.any(in_box)


def test_segment_checker_rejects_crossing_even_when_endpoints_are_free():
    sim = RoomSurveySimulation()
    assert not sim._segment_is_clear(np.array([1.0, 2.5, 1.0]), np.array([5.0, 2.5, 1.0]))
    assert sim._segment_is_clear(np.array([1.0, 2.5, 2.3]), np.array([5.0, 2.5, 2.3]))
    assert not sim._segment_is_clear(np.array([0.1, 0.5, 1.0]), np.array([1.0, 0.5, 1.0]))


def test_visibility_rejects_back_fov_out_of_range_and_box_occlusion():
    sim = RoomSurveySimulation()
    # Deliberate samples on front box face, back wall behind the opaque box,
    # outside horizontal FOV, and behind the camera.
    sim._points = np.array([[3.0, 1.85, 1.0], [3.0, 5.0, 1.0],
                            [5.8, 1.2, 1.0], [3.0, 0.0, 1.0]])
    sim._normals = np.array([[0., -1., 0.], [0., -1., 0.],
                             [-1., 0., 0.], [0., 1., 0.]])
    visible = sim._visible_indices(np.array([3., 0.75, 1.]), np.array([0., 1., 0.]))
    assert visible.tolist() == [0]
    sim._points = np.array([[3.0, 1.85, 1.0]])
    sim._normals = np.array([[0., -1., 0.]])
    sim.config = SurveySimulationConfig(observation_range=0.5)
    assert len(sim._visible_indices(np.array([3., 0.75, 1.]), np.array([0., 1., 0.]))) == 0


def test_simulation_is_deterministic_json_safe_and_snapshots_are_detached():
    a, b = RoomSurveySimulation(seed=18), RoomSurveySimulation(seed=18)
    for _ in range(15):
        assert a.step() == b.step()
    snapshot = a.snapshot()
    assert json.loads(json.dumps(snapshot, allow_nan=False)) == snapshot
    snapshot["points"][0]["position"][0] = 1000
    snapshot["path"][0][0] = 1000
    snapshot["camera_position"][0] = 1000
    if snapshot["keyframes"]:
        snapshot["keyframes"][0]["position"][0] = 1000
    assert a.snapshot() == b.snapshot()


def test_unobservable_scene_stops_honestly_without_claiming_complete():
    sim = RoomSurveySimulation(config=SurveySimulationConfig(observation_range=0.1))
    result = sim.step()
    assert result["state"] == "no_reachable_view"
    assert result["coverage"] == 0
    assert result["points"] == []
    assert "no reachable" in result["message"]
    assert result["flight_commands_sent"] == 0


def test_budget_and_user_stop_prevent_any_further_motion():
    sim = RoomSurveySimulation(config=SurveySimulationConfig(max_steps=2, target_coverage=1.0))
    result = finish(sim)
    assert result["state"] == "stopped"
    assert result["steps"] == 2
    assert "step budget" in result["message"]
    assert sim.step() == result
    other = RoomSurveySimulation()
    other.step()
    stopped = other.stop()
    assert stopped["state"] == "stopped"
    assert other.step() == stopped


def test_simulation_never_represents_cloud_as_real_measurements():
    result = finish(RoomSurveySimulation())
    assert result["source"] == "simulation"
    assert result["units"] == "meters (simulated)"
    assert result["flight_commands_sent"] == 0
    assert "Known simulator geometry" in result["planning_assumption"]
    assert "sampled simulator surfaces" in result["message"]
    assert len(result["points"]) <= result["surface_samples_total"] < 5000
    assert result["target_position"] is None


@pytest.mark.parametrize("kwargs", [
    {"surface_spacing": 0.01}, {"body_clearance": float("nan")},
    {"max_step_distance": float("inf")}, {"target_coverage": 1.1},
    {"max_steps": 0}, {"max_steps": 1.5}, {"max_steps": True},
    {"horizontal_fov_degrees": 180},
])
def test_invalid_or_unbounded_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        SurveySimulationConfig(**kwargs)
