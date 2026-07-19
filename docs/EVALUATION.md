# Evaluation

## Detector metrics

Use `training/evaluate_on_test_split.py` and report:

- precision;
- recall;
- mAP@50;
- mAP@50:95;
- class-level AP;
- confusion matrix and PR curves.

The recorded detector metrics are in `metrics/test_metrics_summary.json`.

## End-to-end metrics to measure on the defense video

- number of rider tracks;
- confirmed no-helmet events;
- duplicate events per rider;
- false event count after manual review;
- processing FPS;
- event latency in frames;
- OCR readable rate only when OCR is enabled.

Detector mAP must not be presented as complete system accuracy. The included
test split is internal and was not separated by source video, so it may be
optimistic.
