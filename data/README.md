# ConstructionSite-RB

This folder provides finalized rule-level annotations and the source image list for 807 images selected from [ConstructionSite10K](https://huggingface.co/datasets/LouisChen15/ConstructionSite). It contains no images.

## Files

| File | Description |
| --- | --- |
| `annotations/benchmark_labels_final.csv` | Authoritative finalized annotations |
| `annotations/benchmark_labels_final_metadata.jsonl` | Canonical JSONL representation |
| `annotations/version.json` | Annotation version and SHA256 values |
| `image_ids.txt` | Selected image IDs, one seven-digit string per line |
| `splits/five_subsets.json` | Fixed descriptive evaluation subsets |

Two civil-engineering annotators independently labeled all images; disagreements were adjudicated by a third expert. Labels represent visible image-level rule violations. Bounding boxes produced by a model are not human annotations.

## CSV schema

The CSV is UTF-8 with BOM. Read IDs as strings, for example with `csv.DictReader` and `encoding='utf-8-sig'`.

| Column | Meaning |
| --- | --- |
| `image_id` | Source filename, such as `0000007.jpg` |
| `rule1`–`rule4` | Binary violation indicators |
| `is_safe` | 1 only when all four rule indicators are zero |
| `label_source` | Annotation agreement/adjudication provenance |

Rule 1 concerns missing hard hats; Rule 2 concerns elevated work without visible harness/lanyard protection; Rule 3 concerns unprotected pits, trenches and edges; Rule 4 concerns dangerous worker proximity to machinery. Labels can overlap. The final positive counts are 205/93/201/96 for Rules 1–4; 302 images have no rule violation.

## JSONL schema

Each line contains one object with `image_id`, `image_path`, `source_split`, `risk_categories`, `risk_boxes`, `is_risk_image`, and `annotation_source`. IDs omit the `.jpg` extension. Rule labels use `rule_1_violation` through `rule_4_violation`; a safe image has an empty list.

`risk_boxes` is empty because this release provides human rule-level labels, not a human bounding-box benchmark. `image_path` identifies the original filename. `source_split` refers to the upstream dataset partition and does not create an independent test partition for this project.

Acquire images directly from the original dataset if running new inference, preserving the selected filenames. This repository does not include an image download or preparation script.
