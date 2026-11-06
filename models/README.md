# External checkpoints

Obtain the following checkpoints from their original providers and place them at the paths in `configs/final.json`.

| Component | Expected path | Source |
| --- | --- | --- |
| Official person YOLO26x | `models/yolo26x.pt` | [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/) |
| Construction YOLO26x | `models/chd_yolo26x.pt` | [Construction Hazard Detection](https://huggingface.co/yihong1120/Construction-Hazard-Detection) |
| GroundingDINO Swin-B | `models/groundingdino_swinb_cogcoor.pth` | [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) |
| BERT text encoder | `models/bert-base-uncased/` | [BERT base uncased](https://huggingface.co/google-bert/bert-base-uncased) |
| Optional Qwen VLM | Supplied with `--model-path` | [Qwen2.5-VL-72B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct-AWQ) |

The official person model uses class 0. The construction model must identify missing hard hats as class 2 and machinery as class 8. GroundingDINO must use the Swin-B configuration corresponding to the stated checkpoint, not the Swin-T checkpoint.

Checkpoint hashes from the verified historical local asset registry appear in [docs/checkpoints.json](../docs/checkpoints.json). The two YOLO files have different contents even though their upstream/local filenames can both be `yolo26x.pt`. Avoid substituting one for the other.

The upstream BERT Git revision and Qwen checkpoint revision were not recorded. Provider updates may change available files; use the recorded hashes for detectors and record any differences. Weights are excluded by `.gitignore` and remain subject to their original licenses.
