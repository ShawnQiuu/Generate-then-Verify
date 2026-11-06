# Experiment outputs

Use `./run.sh` to run experiments. Each run is placed in a new timestamped directory under `runs/`, using Shanghai time. Names follow `YYYYMMDD_HHMMSS__stage__model__name`. Existing directories are never reused.

Use `--run-name` and `--run-note` to describe the purpose. Use `--baseline-run` when a specific prior run is the comparison. Configuration, arguments, input fingerprints, image IDs, environment, source snapshot, command, console, metrics and summary are recorded. Failed and interrupted runs remain visible in `index.jsonl`.

Run outputs and the index are local products excluded from the GitHub code release. Generation/detection stages do not invent evaluation metrics. Replay uses existing evidence; validation repeats CV inference; generation produces new VLM output.
