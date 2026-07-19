# QA validation performed before packaging

Validation environment: Linux, Python 3.13, CPU inference. The recommended user
environment remains Python 3.11 on Windows.

## Static checks

- All Python files compiled successfully with `compileall`.
- `tests/smoke_test.py` passed.
- Dataset image/label pairs were counted for all three splits.
- No file exceeds GitHub's 100 MB single-file limit.

## Checkpoint checks

The bundled `models/best.pt` loaded successfully with Ultralytics and reported:

```text
architecture: YOLOv8s
parameters: 11,136,761
classes: {0: helmet, 1: no_helmet, 2: rider}
```

## End-to-end integration check

A temporary 12-frame MP4 was generated from an included test image containing
rider and no-helmet detections. The real checkpoint and ByteTrack pipeline were
run with low test thresholds.

Observed result:

```text
processed frames: 12
unique rider tracks: 4
confirmed no-helmet tracks: 3
SQLite rows: 3
CSV rows: 3
saved evidence images: 3
annotated video: created and readable
```

The temporary input and generated runtime outputs were deleted before packaging.
This test confirms software integration, not real-world accuracy.
