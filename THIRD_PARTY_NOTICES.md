# Source attribution

Version 11.0.0 bundles BoT-SORT from NirAharon/BoT-SORT, commit
251985436d6712aaf682aaaf5f71edb4987224bd, under its MIT license (copyright
2022 Nir Aharon). The full license and local modification notes are in
follow_neo/vendor/botsort/LICENSE and README.md. SciPy and lap retain the
licenses distributed with their wheels. No ReID weights are included.

The model best.onnx is the exact user-supplied file. Its metadata names Ultralytics and includes `AGPL-3.0 License (https://ultralytics.com/license)`; the ONNX bytes and metadata are unchanged.

The detector adapter is adapted from the user-supplied ATLAS V2 source. Tracker and controller algorithms are ported from the user-supplied NEO Dance C++ project. PROVENANCE.json records the source hashes. No ownership or license of these source components is reassigned by this update.

NumPy, OpenCV, PyAV, ONNX Runtime and Pillow are installed from their Python distributions and retain their respective licenses. Tkinter is supplied by the Python installation. The Java application and OpenDJI.py are not redistributed or modified by this package.
