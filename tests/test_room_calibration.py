"""Independent pinhole/distortion projections exercise measured calibration."""
import json
import math
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pytest

from follow_neo.room_calibration import (
    calibrate_directory, calibrate_observations, write_board_svg,
)


def _rotation(x, y, z):
    cx, cy, cz, sx, sy, sz = math.cos(x), math.cos(y), math.cos(z), math.sin(x), math.sin(y), math.sin(z)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


def observations(*, frontal=False, noise=0.10):
    rng = np.random.default_rng(1729)
    xyz = np.zeros((54, 3))
    xyz[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * .025
    K = np.array([[900., 0, 640.], [0, 910., 360.], [0, 0, 1.]])
    views = []
    for i in range(16):
        angles = (0, 0, 0) if frontal else (-.40 + .24 * (i % 4), -.4 + .25 * (i // 4), -.15 + .1 * (i % 3))
        rotation = _rotation(*angles)
        center = np.array([-.13 + .085 * (i % 4), -.085 + .055 * (i // 4), .58 + .07 * (i % 3)])
        camera = (xyz - xyz.mean(axis=0)) @ rotation.T + center
        x, y = (camera[:, :2] / camera[:, 2, None]).T
        r2 = x * x + y * y
        radial = 1 - .12 * r2 + .025 * r2**2 - .004 * r2**3
        xd = x * radial + 2 * .001 * x * y - .0005 * (r2 + 2 * x * x)
        yd = y * radial + .001 * (r2 + 2 * y * y) - 2 * .0005 * x * y
        pixel = np.column_stack([900 * xd + 640, 910 * yd + 360])
        pixel += rng.normal(0, noise, size=pixel.shape)
        views.append(pixel)
    return K, views


def test_known_camera_and_distortion_recovered(tmp_path):
    known, views = observations()
    target = tmp_path / "calibration.json"
    report = calibrate_observations(views, 1280, 720, output_path=target)
    assert report["success"], report["warnings"]
    actual = np.asarray(report["K"])
    np.testing.assert_allclose(actual.diagonal()[:2], known.diagonal()[:2], rtol=.02)
    np.testing.assert_allclose(actual[:2, 2], known[:2, 2], atol=7)
    assert abs(report["distortion"][0] - (-.12)) < .03
    assert .08 < report["rms_px"] < .20
    assert report["views"] == 16
    assert len(report["per_view_rmse_px"]) == 16
    assert report["quality"]["max_normal_separation_deg"] > 30
    saved = json.loads(target.read_text(encoding="utf-8"))
    assert saved == report
    assert saved["provenance"]["metric_map_scale_established"] is False
    assert list(tmp_path.glob("*.tmp")) == []


def test_calibration_compatible_with_slam_loader(tmp_path):
    calibration = pytest.importorskip("rbd_slam.calibration")
    _, views = observations()
    path = tmp_path / "camera.json"
    report = calibrate_observations(views, 1280, 720, output_path=path)
    assert report["success"]
    loaded = calibration.load_calibration(path)
    assert (loaded.width, loaded.height) == (1280, 720)
    np.testing.assert_allclose(loaded.K, report["K"])


def test_front_parallel_translated_views_fail_geometry(tmp_path):
    _, views = observations(frontal=True)
    path = tmp_path / "camera.json"
    report = calibrate_observations(views, 1280, 720, output_path=path)
    assert not report["success"]
    assert report["views"] == 16
    assert any("perspective" in warning or "normal" in warning for warning in report["warnings"])
    assert not path.exists()


def test_duplicate_views_and_reversed_corner_order_do_not_count():
    _, views = observations()
    report = calibrate_observations([views[0], views[0][::-1]] * 10, 1280, 720)
    assert not report["success"]
    assert report["views"] == 1
    assert len(report["rejectedfiles"]) == 19


def test_too_few_views_never_writes(tmp_path):
    _, views = observations()
    path = tmp_path / "camera.json"
    report = calibrate_observations(views[:9], 1280, 720, output_path=path)
    assert not report["success"]
    assert not path.exists()


def test_bad_reprojection_rejects_output(tmp_path):
    _, views = observations(noise=2.5)
    path = tmp_path / "camera.json"
    report = calibrate_observations(views, 1280, 720, output_path=path)
    assert not report["success"]
    assert report["rms_px"] > 1
    assert not path.exists()


def test_existing_file_protected_and_explicit_overwrite(tmp_path):
    _, views = observations()
    path = tmp_path / "camera.json"
    path.write_text("original", encoding="utf-8")
    with pytest.raises(FileExistsError):
        calibrate_observations(views, 1280, 720, output_path=path)
    assert path.read_text() == "original"
    failed = calibrate_observations(views[:2], 1280, 720, output_path=path, overwrite=True)
    assert not failed["success"]
    assert path.read_text() == "original"
    passed = calibrate_observations(views, 1280, 720, output_path=path, overwrite=True)
    assert passed["success"]
    assert json.loads(path.read_text())["success"]


def _board():
    image = np.full((480, 640), 255, np.uint8)
    for row in range(7):
        for col in range(10):
            if (row + col) % 2 == 0:
                image[80 + 40 * row:120 + 40 * row, 100 + 40 * col:140 + 40 * col] = 0
    return image


def test_actual_detector_corrupt_files_and_duplicates(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    for i in range(11):
        assert cv2.imwrite(str(images / f"board{i:02}.png"), _board())
    (images / "corrupt.png").write_bytes(b"not an image")
    path = tmp_path / "camera.json"
    report = calibrate_directory(images, path)
    assert not report["success"]
    assert report["views"] == 1
    assert any(item["reason"] == "unreadable image" for item in report["rejectedfiles"])
    assert len(report["rejectedfiles"]) == 11
    assert not path.exists()


def test_resolution_mismatch_rejects_directory(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    cv2.imwrite(str(images / "one.png"), _board())
    cv2.imwrite(str(images / "two.png"), np.ones((240, 320), np.uint8))
    path = tmp_path / "camera.json"
    report = calibrate_directory(images, path)
    assert not report["success"]
    assert any("resolution" in warning for warning in report["warnings"])
    assert not path.exists()


def test_empty_directory_and_missing_directory(tmp_path):
    report = calibrate_directory(tmp_path, tmp_path / "camera.json")
    assert not report["success"]
    assert report["views"] == 0
    with pytest.raises(FileNotFoundError):
        calibrate_directory(tmp_path / "missing", tmp_path / "camera.json")


@pytest.mark.parametrize("width,height", [(0, 720), (1280., 720), (1280, True)])
def test_invalid_dimensions(width, height):
    with pytest.raises(ValueError):
        calibrate_observations([], width, height)


def test_printable_board_geometry_and_existing_file(tmp_path):
    path = write_board_svg(tmp_path / "board.svg")
    root = ET.fromstring(path.read_text())
    assert root.attrib["width"] == "300mm"
    assert root.attrib["height"] == "250mm"
    squares = [item for item in root if item.attrib.get("fill") == "black"]
    assert len(squares) == 35
    assert max(float(item.attrib["x"]) for item in squares) == 250
    assert max(float(item.attrib["y"]) for item in squares) == 175
    assert all(item.attrib["width"] == "25" and item.attrib["height"] == "25" for item in squares)
    assert "Print 100%" in path.read_text()
    with pytest.raises(FileExistsError):
        write_board_svg(path)
