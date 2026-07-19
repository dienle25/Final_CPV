# Helmet Detection Project Notes

## Quick audit of the current dataset

- Dataset format is already compatible with YOLO: `train/images`, `train/labels`, `valid/images`, `valid/labels`.
- There are **942 images** total: **800 train** and **142 valid**.
- `data/data.yaml` declares **5 classes**, but the class names are currently only `['0', '1', '2', '3', '4']`.
- The declared `test` split is missing on disk, so evaluation is currently only train/valid.
- All images were already resized to **640x640 (stretch)** before export.
- One class is almost certainly a **license plate** class, and one large-box class is almost certainly a **motorcycle/rider** class.
- From sample inspection, the remaining classes appear helmet/head-related, but the exact mapping was lost in the export and should be verified manually before final training.

## Risks you should mention in the course report

- The class names were lost during export, so relabeling the names in `data.yaml` is required before the final experiment.
- The split likely contains frames or screenshots from the same source videos in both train and valid, so validation mAP may look better than real-world performance.
- Many images include watermarks, screenshots, and traffic scenes from mixed domains, which can create domain shift if your demo video is different.
- The dataset includes classes that are not directly needed for the final task of "driver wears helmet or not", so a naive 5-class detector is not yet the cleanest final solution.

## Recommended training direction

### 1. Start with a YOLO baseline

Use the current dataset as-is to get a first measurable result quickly:

```bash
pip install ultralytics pillow
python scripts/audit_yolo_dataset.py
python train_yolo_baseline.py --data data/data.yaml --model yolov8s.pt --epochs 100 --device 0
```

This baseline is useful for your report because it gives:

- training loss curves
- precision / recall / mAP
- sample predictions
- a concrete first experiment

### 2. Verify and rename the classes

Run the audit script and inspect the preview images in `audit_samples/`.

After you identify the real meaning of each numeric class, rename `names` in `data/data.yaml`.

Example idea only:

```yaml
names: ['plate', 'no_helmet', 'helmet', 'head_or_passenger', 'motorcycle']
```

Do not use the example above blindly. Confirm it visually first.

### 3. For the final course project, simplify the task

Your project goal is not "detect everything in the dataset". It is:

**detect riders in traffic video and decide whether the rider is wearing a helmet**

So the cleaner final pipeline is:

1. Detect motorcycle/rider boxes.
2. Detect helmet / no-helmet head boxes.
3. Match the head box to the rider box by overlap or by checking whether the head center lies in the upper area of the rider box.
4. Mark each rider as `helmet` or `no_helmet`.
5. For video, add tracking such as ByteTrack so the decision is stable across frames.

### 4. Stronger project version after the baseline

After you confirm the class mapping, create a reduced dataset focused on only the classes relevant to the final task:

- keep: rider / motorcycle
- keep: helmet
- keep: no_helmet
- drop: license plate
- drop: any irrelevant helper class

This usually improves clarity, training stability, and presentation quality in the final demo.

## Suggested experiment plan

### Baseline

- Model: `yolov8n.pt` if GPU is weak, `yolov8s.pt` if GPU is decent
- Epochs: `80-120`
- Image size: `640`
- Batch: `16` or `auto`
- Metric to report: `mAP50`, `mAP50-95`, precision, recall

### Improved run

- Rename classes correctly
- Build a cleaner split by source video if possible
- Retrain on only the relevant classes
- Test on your own traffic video, not only the Roboflow valid split

## Files added for you

- `scripts/audit_yolo_dataset.py`: prints dataset stats and exports annotated sample images
- `train_yolo_baseline.py`: runs a practical YOLO baseline with reasonable defaults
