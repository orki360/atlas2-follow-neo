# Calibrating the original video frames

This helper calls OpenCV's chessboard detector and camera-calibration solver.
It is not a Python reimplementation of that solver. It never connects to a
camera, video socket, or aircraft.

1. Create a board with `write_board_svg("board.svg")`. The default is **9 x 6
   inner corners**, therefore **10 x 7 squares**, each 25 mm wide. The complete
   page is 300 x 250 mm, so use A3 paper. Print at 100% / actual size, disable
   fit-to-page, and verify the printed square size with a ruler. Keep the sheet
   flat on a rigid backing. A different actual square size must be supplied to
   the calibration call.
2. Capture approximately 20 sharp original stream frames with the whole board
   visible. Keep exactly the video mode, resolution, zoom, crop, and focus that
   mapping will use. Do not use resized previews or frames with overlays.
3. Move the board across the image and vary distance and tilt around **both**
   axes. Simply translating a front-facing board is insufficient. Repeated
   pictures of an unchanged board do not count as independent observations.
4. Put PNG/JPEG/BMP/TIFF images in one directory and call:

   ```python
   from follow_neo.room_calibration import calibrate_directory
   report = calibrate_directory("calibration_frames", "camera.json")
   print(report["success"], report["warnings"])
   ```

At least 10 distinct complete board detections are required. All readable
images must have the same resolution. The helper screens observed perspective
variation and recovered board-normal diversity, requires overall reprojection
RMS <= 1 pixel and each view RMSE <= 2 pixels, and rejects nonfinite or implausible
parameters. These are screening criteria, not proof of accuracy in a room.
Inspect the per-view errors and retake blurry, warped, or poorly distributed
images; a smaller residual does not by itself prove a more accurate camera.

A JSON file is written atomically only on success. Existing files are protected
unless `overwrite=True` is passed explicitly. It contains `K`, `distortion`,
`width`, and `height` compatible with `rbd_slam.calibration.load_calibration`,
plus quality and provenance metadata. The original image files are unchanged.

Changing resolution, image cropping, zoom, or camera processing requires checking
or repeating calibration. Calibration estimates camera intrinsics and distortion;
the physical board dimensions do **not** establish metric scale for the subsequent
monocular room map. Additional scale information and camera-to-aircraft geometry
are needed before translating map coordinates into aircraft motion.

Reference: [OpenCV camera calibration tutorial](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html).
