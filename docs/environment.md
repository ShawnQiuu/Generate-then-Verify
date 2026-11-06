# Environment and reproducibility

## Saved-output evaluation

The evaluation dependencies are pinned in `requirements.txt`: NumPy 2.2.6 and SciPy 1.15.3, using Python 3.10.20. Evaluation does not import PyTorch, Ultralytics, GroundingDINO or Transformers. Supply one VLM's logs and, for fusion replay, detector evidence.

## Detector environment

The historical CV environment used Python 3.10.20, PyTorch 2.4.0+cu124, torchvision 0.19.0+cu124, Transformers 4.38.2 and Ultralytics 8.4.46. Supporting versions are listed in `requirements/cv.txt`.

```bash
python -m venv .venv-cv
source .venv-cv/bin/activate
python -m pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements/cv.txt
git clone https://github.com/IDEA-Research/GroundingDINO.git third_party/GroundingDINO
python -m pip install --no-build-isolation --no-deps -e third_party/GroundingDINO
```

Follow the [GroundingDINO installation instructions](https://github.com/IDEA-Research/GroundingDINO) for the CUDA compiler and custom extension. The upstream Git commit used for the historical extraction was not recorded. The public installation command is a setup route, not an exact historical source lock. The recorded source-file hashes are retained in `groundingdino_source_manifest.json` for comparison.

The checkpoint paths and BERT directory are repository-relative in `configs/final.json`. No weights or third-party source trees are bundled. Existing checkpoint hashes are listed in `checkpoints.json`; downloads should be compared with these identities before claiming exact detector reproduction.

## Optional Qwen generation

Use a separate environment following the [Qwen AWQ model card](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct-AWQ). `requirements/vlm.txt` lists the API compatibility requirements; it is not a validated historical generation lockfile. Install a compatible PyTorch/CUDA build and AWQ backend for the quantized checkpoint. The GroundingDINO Transformers 4.38.2 pin cannot load Qwen2.5-VL.

Historical VLM generation dependency versions and checkpoint revisions were not fully recorded. New generation therefore requires environment verification on the target hardware, and may differ from archived logs. The optional GPU adapters were not exercised during this code-packaging task. Only saved-output equivalence is claimed by the release validation.

Generation records its precision, attention implementation, pixel budget, token budget and timings. Fresh detector extraction records its own timings. Neither measurement by itself is end-to-end system cost.
