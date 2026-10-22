# Model checkpoints

The model files are downloaded locally and ignored by version control.

| File | Purpose | Source and terms |
| --- | --- | --- |
| `cricket-ball-best.pt` | One-class YOLOv8 cricket-ball detector (`ball`, class 0) | [Hugging Face model card](https://huggingface.co/ashishgimekar/cricket-ball-yolo), listed as MIT; trained on the Kaggle dataset documented in `datasets/README.md`. |
| `yolo11n-pose.pt` | General 17-keypoint human pose model; not trained to recognize cricket bowling actions | [Official Ultralytics release](https://github.com/ultralytics/assets/releases/tag/v8.3.0), distributed under AGPL-3.0 terms. |

The default CLI configuration points at both files. Pose inference still requires a
manually selected bowler track. The pose model provides generic COCO keypoints; it does
not provide validated release timing or bowling-action recognition.

The cricket-ball checkpoint is a task-specific candidate, not a verified best-in-class
model. Evaluate it on held-out footage from the intended camera, venue, lighting, and
ball colors before relying on its detections. The Ultralytics inference software has its
own AGPL-3.0 or Enterprise licensing terms; review the applicable terms for your use.

Do not commit private weights or large binary artifacts to version control. Checkpoint
SHA-256 values for the downloaded files:

- `cricket-ball-best.pt`: `2ce15e3ef8dd81f55de08dc0497fdd2d9cad4eb85d00d7d608095aaad6bee18e`
- `yolo11n-pose.pt`: `869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0`
