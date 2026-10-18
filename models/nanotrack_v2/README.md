# NanoTrack V2 / OpenCV CPU

These are the official NanoTrack V2 backbone and head, licensed under the
included Apache-2.0 LICENSE. They are separate from the YOLO model selector.

Source: https://github.com/HonglinChu/SiamTrackers/tree/master/NanoTrack/models/nanotrackv2

SHA-256 (verified on loading):

* nanotrack_backbone_sim.onnx: 530bdd0cd00f19afab79a863e71ba71e3312395a5dc9151af675082bdaaa2fc4
* nanotrack_head_sim.onnx: 0d8c0637be849f092cc7236cae02e55c8b9455ebe37ba50601d6115db4247cd9

The pinned opencv-python 4.12.0.88 already exposes TrackerNano. No contrib,
CUDA, dependency replacement or new Python environment is required.
Nano's score is not a calibrated NEO-presence probability. Its update success
flag does not detect target absence. Never use either as flight authorization.
