# Model card — `models/best.pt`

## Identity

- Task: object detection.
- Architecture: **YOLOv8s**.
- Parameters: **11,136,761**.
- Classes and fixed IDs:
  - `0`: `helmet`
  - `1`: `no_helmet`
  - `2`: `rider`
- Checkpoint size: approximately 22 MB.

The architecture was verified from the checkpoint metadata: depth multiplier
`0.33` and width multiplier `0.50`, which correspond to YOLOv8s. The repository
must not describe this checkpoint as YOLOv8n.

## Recorded training metadata

The checkpoint records a second-stage run with the following values:

- image size: `896`;
- epochs: `80`;
- batch: `4`;
- optimizer: `AdamW`;
- patience: `20`;
- pretrained initialization: enabled.

The included notebook and scripts preserve the available training provenance,
but the original Kaggle environment and all exact package versions were not
fully recorded. Therefore, retraining is reproducible in method, not guaranteed
bit-for-bit.

## Detector test metrics

See `metrics/test_metrics_summary.json` and `metrics/plots/`.

- Precision: `0.8937`
- Recall: `0.8661`
- mAP@50: `0.9302`
- mAP@50:95: `0.5714`

These values are detector box metrics only. They do not validate ByteTrack,
rider-head association, violation-event accuracy or OCR.

## Known limitations

- The test split was sampled from the original training split rather than split
  by source video or scene.
- The `no_helmet` class has the lowest class mAP@50:95 (`0.5155`).
- The checkpoint has no license-plate class.
- The project is a classroom decision-support prototype and requires human
  review before any real-world action.
