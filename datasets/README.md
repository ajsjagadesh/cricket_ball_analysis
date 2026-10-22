# Cricket-ball dataset workflow

This workspace has the Kaggle [Cricket Ball Dataset for YOLO](https://www.kaggle.com/datasets/kushagra3204/cricket-ball-dataset-for-yolo)
downloaded and extracted locally. Kaggle lists it as CC0/Public Domain. The local zip and
extracted image data are ignored by version control. The archive contains 1,912 images:
1,778 train, 63 validation, and 71 test images. Two training images have no matching
label file; review them as possible negative examples before training.

Use `datasets/cricket_ball_data.yaml` with the training commands below. The checkpoint
`models/cricket-ball-best.pt` is a separate MIT-listed YOLOv8 model card trained on this
dataset. Its reported source and license are recorded in [models/README.md](../models/README.md);
the model has not been independently evaluated on this project's video footage.

The earlier FPVShotDetection candidate was not downloaded: its public repository contains
Pascal VOC XML annotations and frame metadata but no corresponding image files, so it
cannot be used as a standalone training dataset.

## YOLO one-class layout

Annotate the ball as class `ball` (class index `0`) with tight boxes around the visible
ball. Include tiny and motion-blurred balls, but mark frames where the ball cannot
reliably be located as having no ball object. Do not label highlights, bright pixels, or
other objects as balls.

Split by complete delivery, match, or recording session—not adjacent frames. Frames from
one delivery must not appear in both training and validation/test sets.

The locally extracted layout is:

```text
datasets/cricket_ball_dataset/cricket_ball_data/
  train/images/
  train/labels/
  valid/images/
  valid/labels/
  test/images/
  test/labels/
```

Each label file uses YOLO format, one object per line:

```text
class_id x_center_normalized y_center_normalized width_normalized height_normalized
```

Example `data.yaml` (replace paths with your dataset paths):

```yaml
path: datasets/ball
train: images/train
val: images/val
test: images/test
names:
  0: ball
```

Review labels at full resolution. Record consistently whether a partially occluded ball
is labeled; avoid oversized boxes that include the bat or hands. Reserve complete unseen
deliveries for validation and test evaluation.

## Train and evaluate

Install the optional dependency after checking the current Ultralytics and base-weight
licenses for your intended use:

```powershell
python -m pip install -r requirements-ball.txt
python -m cricket_vision.cli train-ball --data "datasets/cricket_ball_data.yaml" --base-model "models/cricket-ball-best.pt" --epochs 100 --imgsz 1280
```

Training is an explicit command and may download the requested base checkpoint if it is
not already installed. The trained run writes checkpoints under the training output
directory; pass the resulting `best.pt` path to analysis:

```powershell
yolo detect val model="runs/detect/train/weights/best.pt" data="datasets/ball/data.yaml" imgsz=1280
yolo predict model="runs/detect/train/weights/best.pt" source="videos/held_out_delivery.mp4" imgsz=1280
```

Report precision/recall and per-frame localization performance on held-out deliveries.
Inspect false positives in crowd/advertising regions and misses in tiny or blurred-ball
frames before using the checkpoint.

## Temporal bowling-action dataset

The temporal model uses labeled per-frame pose-derived features. Do not use the heuristic
labels produced by the application as ground truth. Review recorded deliveries and label
each selected frame using one of: `unknown`, `run-up`, `delivery_approaching`,
`release_candidate`, or `follow_through`. Use a stable `delivery_id` for every frame
belonging to one delivery. Preserve complete delivery groups for held-out testing.

1. Run pose analysis on a clip with a compatible pose model and manually selected bowler
   track. The analysis sidecar includes `bowling_action.features`.
2. Create a manual labels JSONL file, one record per reviewed frame:

```json
{"delivery_id":"match1-over3-ball2","frame_index":125,"label":"run-up"}
{"delivery_id":"match1-over3-ball2","frame_index":131,"label":"delivery_approaching"}
{"delivery_id":"match1-over3-ball2","frame_index":137,"label":"release_candidate"}
{"delivery_id":"match1-over3-ball2","frame_index":141,"label":"follow_through"}
```

3. Assemble and validate features. Frames that have no pose feature record cannot be
   included. Missing individual feature values are represented as zero by the assembler.

```powershell
python -m cricket_vision.cli prepare-action-data --frames "output/pose_analysis.jsonl" --labels "datasets/action_labels.jsonl" --output "datasets/action_sequences.jsonl"
python -m cricket_vision.cli validate-action-data --data "datasets/action_sequences.jsonl"
```

4. Install PyTorch if needed and train. The trainer splits by `delivery_id`; frames from
   the same delivery are never divided between internal training and validation.

```powershell
python -m pip install -r requirements-action.txt
python -m cricket_vision.cli train-action --data "datasets/action_sequences.jsonl" --output "models/bowling_action.pt" --epochs 40
```

Training writes a metrics JSON beside the checkpoint, including frame accuracy,
macro-F1, per-class results, confusion matrix, and pre-release anticipation rate/mean
lead time over the held-out delivery groups. These metrics are meaningful only if the
labels are independently reviewed and the split reflects deployment footage.

Use a separately held-out JSONL dataset for an external evaluation:

```powershell
python -m cricket_vision.cli evaluate-action --data "datasets/action_test_sequences.jsonl" --model "models/bowling_action.pt"
```

To show the trained model's per-frame state during inference, provide both the pose
checkpoint and action-model checkpoint:

```powershell
python -m cricket_vision.cli analyze --video "videos/match.mp4" --output "output/action_analysis.mp4" --pose-model "models/pose.pt" --bowler-track-id 3 --action-model "models/bowling_action.pt"
```

The current sample repository contains no action labels or trained action-model
checkpoint. Do not report accuracy, lead time, or generalization until the workflow has
been run on representative, held-out cricket deliveries.
