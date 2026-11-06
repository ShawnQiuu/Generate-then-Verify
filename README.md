# Generate-then-Verify

Code and annotations for **Generate-then-Verify: A Training-Free Evidence-Grounded Framework for Trustworthy Vision-Language Reasoning in Safety-Critical Applications**.

The framework combines the output of **one vision-language model per image** with frozen visual detectors and deterministic rules for construction safety reporting. This repository contains the final `fixed_chd960_f1` configuration and the finalized **ConstructionSite-RB** rule-level annotations.

No detector, VLM, fusion model, or calibrator is trained by this framework. Rules and thresholds were selected using the 807-image development set; training-free does not imply label-free.

## Repository structure

```text
configs/final.json          Final detector, geometry and fusion settings
src/                       Detection, generation, fusion and evaluation code
prompts/safety_report.txt   Construction safety report prompt
data/annotations/          Final annotation CSV, JSONL and version hashes
data/image_ids.txt         807 selected source image IDs
data/splits/               Five fixed evaluation subsets
requirements/              Optional detector and VLM dependencies
docs/                      Method, metrics, environment and provenance
models/README.md           Checkpoint sources and expected local paths
run.sh                     Command-line entry point with experiment recording
```

## Installation

Python 3.10 is recommended. For evaluation of saved model outputs:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
./run.sh check
```

Alternatively, use `conda env create -f environment.yml` and activate `generate-then-verify`. The launcher uses the active `python`; the `PYTHON` environment variable can select another interpreter.

New detector inference requires the additional packages and checkpoints described in [docs/environment.md](docs/environment.md) and [models/README.md](models/README.md). VLM generation uses a separate environment because GroundingDINO and Qwen require different Transformers versions.

## ConstructionSite-RB annotations

The benchmark contains **807 images selected from [ConstructionSite10K](https://huggingface.co/datasets/LouisChen15/ConstructionSite)**. Images are obtained from the original dataset. This repository distributes the selected image IDs and the finalized image-level annotations.

| Rule | Violation | Positive images |
| --- | --- | ---: |
| Rule 1 | Personnel without hard hats | 205 |
| Rule 2 | Elevated work without visible harness/lanyard protection | 93 |
| Rule 3 | Unprotected deep pits, trenches or edges | 201 |
| Rule 4 | Personnel dangerously close to machinery | 96 |

There are 302 safe images and 505 images with at least one violation. Labels can overlap. See [data/README.md](data/README.md) for the CSV/JSONL schemas.

The authoritative file is [benchmark_labels_final.csv](data/annotations/benchmark_labels_final.csv); [benchmark_labels_final_metadata.jsonl](data/annotations/benchmark_labels_final_metadata.jsonl) is its canonical JSONL representation. Checksums are recorded in [version.json](data/annotations/version.json). Preserve all seven digits in image IDs, including leading zeros. The [image list](data/image_ids.txt) identifies the selected source images without requiring an image preparation script.

The JSONL `source_split` field refers to the original dataset partition. It does not identify an independent test set for this work. Generated detector/VLM boxes are not human ground truth; the released human annotations are rule-level labels.

## Final configuration

The CV stream uses an official YOLO26 person detector, construction-specific YOLO26 missing-hardhat/machinery detections, and GroundingDINO Swin-B protection-equipment detections. YOLO uses BGR input and size 960. GroundingDINO uses its own standard image transform. The configuration uses no image crops and no second VLM.

| Rule | Final decision |
| --- | --- |
| Rule 1 | A VLM claim with matched CV support, or at least one matched no-hardhat detection with confidence ≥ 0.20 |
| Rule 2 | Elevated-work context and a Rule 2 claim from the same VLM |
| Rule 3 | Pit/edge context from the VLM and a CV person without a nearby guardrail |
| Rule 4 | A VLM Rule 4 claim and qualifying person–machinery geometry |

Rule 1 associates the no-hardhat box center with the upper 40% of a person box, with at most one vote per person. Rule 2 does not require CV harness confirmation. Rule 3 does not require a separate VLM Rule 3 claim. Geometry uses two-dimensional image heuristics. All active thresholds are in [configs/final.json](configs/final.json).

## Running experiments

Model outputs and detector evidence are supplied by the user; full historical logs and run directories are not distributed in this code release. Logs may be a directory containing `<seven-digit-ID>.json` files or one JSONL file with `image_id` on each row. Each log contains `scene_context` and `anomalies`; anomaly rule categories are `Rule_1` through `Rule_4` or `Other`.

Evaluate saved single-VLM logs:

```bash
./run.sh eval --logs /path/to/qwen_logs \
  --run-name qwen_baseline --run-note "Standalone Qwen on final annotations"
```

Replay the final method using saved detector evidence:

```bash
./run.sh reproduce --logs /path/to/qwen_logs \
  --evidence /path/to/evidence.jsonl \
  --run-name qwen_final --run-note "Final configuration with saved CV evidence"
```

Recompute CV detections while reusing the chosen VLM logs:

```bash
./run.sh validate --images-dir /path/to/images --logs /path/to/qwen_logs \
  --run-name fresh_cv --run-note "Fresh CHD960 inference with saved Qwen logs"
```

The image directory should contain files such as `0000007.jpg`. `detect` performs detector extraction only; its `outputs/evidence.jsonl` can be passed to `reproduce --evidence`. [docs/method.md](docs/method.md) specifies the evidence schema and final paper ablations.

Optional new Qwen generation:

```bash
./run.sh generate --model-path /path/to/Qwen2.5-VL-72B-Instruct-AWQ \
  --images-dir /path/to/images \
  --run-name qwen_generation --run-note "Single-Qwen generation with the retained prompt"
```

Generation writes `outputs/logs.jsonl`. Pass that file to subsequent evaluation with `--logs`. The adapter uses greedy decoding, 1,024 new tokens and the retained pixel budget. Malformed JSON fails visibly, with raw text preserved; the adapter does not silently treat generation failure as a safe prediction. It is not guaranteed to reproduce historical model outputs byte for byte.

GPT saved-log evaluation uses `--model gpt --logs /path/to/gpt_logs` with the same frozen configuration. Each invocation uses only one VLM. No GPT API client or API credentials are included.

Use `--image_ids_file /path/to/ids.txt` for an explicit subset of the 807 benchmark IDs. Relative paths are resolved against the repository root. See `./run.sh --help` for all options.

## Evaluation and statistics

The released five subsets contain 162, 162, 161, 161 and 161 images. Every subset uses the same complete frozen configuration. The configuration was previously selected using **all 807 development labels**, so these subset summaries describe development-set performance. They are not leakage-free selection-validation cross-validation or independent testing. The additional 100-image test set is not included or used.

The evaluator saves pooled metrics, five-subset means and sample SDs, per-rule confusion counts and per-image predictions. Scores and rates in machine-readable outputs use a **0–1** scale. The Standard/Strict/Relaxed image-level definitions and hallucination metrics follow the original implementation; Strict accuracy is distinct from exact-set accuracy. See [docs/metrics.md](docs/metrics.md).

The six paper ablations are enabled with `reproduce --ablations`; all reuse the final thresholds. For the full Qwen statistical family, add `--ablations --statistics`. The optional analysis uses 20,000 paired image bootstrap samples within subsets, 100,000 whole-image paired permutations, exact McNemar tests and a joint 355-test Holm correction. The existing GPT main comparison uses a separate 51-test family. These analyses can take longer than ordinary evaluation. Confidence intervals are pointwise and conditional on fixed predictions.

For this code release, numerical equivalence was checked using the historical **non-bbox Qwen logs** and the finalized labels. The later bbox-prompt log correction is outside this release's validation scope. This distinction is retained in [docs/release_provenance.json](docs/release_provenance.json).

## Experiment records

Experiments run through `./run.sh` and create a new timestamped directory in `experiments/runs/`. Records include the configuration, parsed arguments, image IDs, input fingerprints, environment, source snapshot, command, console log, metrics and summary. Successful, failed and interrupted runs are retained. `check` and help are read-only. Generated run directories are excluded by `.gitignore`.

Saved-output replay, new detector inference and new VLM generation are different operations. Detector and generation timings are recorded separately. End-to-end latency, memory use and cost reduction have not been established by this release's packaging checks.

## Attribution

Please cite the accompanying manuscript and the original [ConstructionSite10K work](https://huggingface.co/datasets/LouisChen15/ConstructionSite/blob/main/README.md) when using this benchmark. The manuscript citation should be updated when its public bibliographic record is available.

Dependencies include [Ultralytics](https://github.com/ultralytics/ultralytics), [Construction Hazard Detection](https://github.com/yihong1120/Construction-Hazard-Detection), [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), and [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct-AWQ). Original images and model weights retain their respective terms.

## License

- **Code** (`src/`, `configs/`, `prompts/`, `run.sh`): MIT, see [LICENSE](LICENSE).
- **Annotations and image identifiers** (`data/annotations/`, `data/image_ids.txt`, `data/splits/`): CC BY-NC 4.0, following the license of ConstructionSite10K; see [data/annotations/LICENSE.md](data/annotations/LICENSE.md).

This repository does not redistribute images or model weights.
