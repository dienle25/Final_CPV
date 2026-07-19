# Data

The merged repository includes the remapped three-class YOLO dataset at:

```text
data/helmet_3class/
├── train/   # 670 images
├── valid/   # 142 images
├── test/    # 120 images
└── data.yaml
```

Class order is fixed and must not be changed:

```text
0 helmet
1 no_helmet
2 rider
```

Box counts in the included copy:

| Split | Helmet | No helmet | Rider | Total boxes |
|---|---:|---:|---:|---:|
| Train | 1,516 | 668 | 1,494 | 3,678 |
| Valid | 274 | 179 | 300 | 753 |
| Test | 271 | 123 | 256 | 650 |

The test split was created by sampling approximately 15% of the original train
split. It is useful for an internal demo but is not a source-video-independent
real-world test set.

Place the prepared defense video at `data/demo.mp4`. Generated videos should not
be committed to Git.
