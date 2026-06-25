# Prototype 4 Lacrosse Pipeline

Prototype 4 is an integration layer for end-to-end lacrosse video processing. It connects sibling public repositories without vendoring their model code:

- `../vggt` from `facebookresearch/vggt`
- `../sam3` from `facebookresearch/sam3`
- `../sam-body4d` from `gaomingqi/sam-body4d`

The first runnable goal is a dry-run pipeline. It accepts a video path, samples frames when OpenCV is available, creates the expected output folders, writes placeholder stage manifests, and records exactly where VGGT, SAM 3, and SAM-Body4D will be called later. It does not run heavy inference, download checkpoints, install large dependencies, or train anything.

## Missing Sibling Repos

If a sibling repo is missing, clone it next to this project:

```bash
cd /afs/ece.cmu.edu/usr/zllenza/research/prototype4
git clone https://github.com/facebookresearch/vggt.git
git clone https://github.com/facebookresearch/sam3.git
git clone https://github.com/gaomingqi/sam-body4d.git
```

Do not copy those repositories into `prototype4_pipeline/`.

## Quick Start

```bash
cd /afs/ece.cmu.edu/usr/zllenza/research/prototype4/prototype4_pipeline
python3 scripts/run_dry_run.py --video /path/to/lacrosse_game.mp4 --dry-run
```

The base ECE Python may not include OpenCV. If video ingestion fails with a `cv2` import error, create a lightweight environment and install only the dry-run dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install opencv-python pyyaml
python3 scripts/run_dry_run.py --video /path/to/lacrosse_game.mp4 --dry-run
```

## VGGT Import Smoke Test

The VGGT import smoke test is intentionally lighter than inference. It adds `../vggt` to `sys.path` for the check, imports the minimum VGGT modules used later by Prototype 4, and exits before model construction, checkpoint loading, or inference.

Install only the smoke-test dependencies into the project `.venv` when you are ready to verify imports:

```bash
cd /afs/ece.cmu.edu/usr/zllenza/research/prototype4/prototype4_pipeline
source .venv/bin/activate
# Pick the correct CUDA/CPU wheel source for PyTorch on ece020.
pip install -r requirements/vggt-smoke.txt
.venv/bin/python scripts/check_vggt_imports.py
```

The dependency list is derived from `../vggt/requirements.txt`, `../vggt/pyproject.toml`, and the local tracking/COLMAP helper imports: `torch==2.3.1`, `torchvision==0.18.1`, `numpy==1.26.1`, `Pillow`, `huggingface_hub`, `einops`, `safetensors`, `hydra-core`, LightGlue from `git+https://github.com/cvg/LightGlue.git`, plus `pycolmap` and `trimesh` for the planned COLMAP export path. This check still does not download VGGT weights.

## Stages

1. Video ingestion
2. Frame sampling
3. VGGT world reconstruction
4. SAM 3 player segmentation and tracking
5. Active-player filtering
6. Player crop extraction
7. SAM-Body4D mesh reconstruction
8. Coordinate alignment
9. Visualization export
10. Saved metadata/artifacts

## Output Layout

Each run writes under `outputs/<run_id>/`:

- `world_reconstruction/`
- `player_masks/`
- `player_tracks/`
- `player_crops/`
- `player_meshes/`
- `visualizations/`
- `metadata/`

The manifests preserve track IDs, timestamps, mask references, crop paths, field-coordinate placeholders, mesh-output placeholders, and visualization metadata for future jersey-number recognition, roster metadata, heatmaps, workload metrics, and highlight/event analytics.

## Guarded VGGT Tiny Inference Smoke Test

A separate script exists for the next step, but it is intentionally not part of dry-run:

```bash
.venv/bin/python scripts/run_vggt_inference_smoke.py --run-id nll-test1 --frame-count 4 --allow-download-weights
```

This script uses only 4 to 8 sampled frames and writes lightweight metadata under `outputs/nll-test1/world_reconstruction/vggt_inference_smoke/`. It refuses to run without `--allow-download-weights`, because `VGGT.from_pretrained(...)` may download checkpoint weights. By default it requires CUDA; use `--device cpu` only for an intentional CPU smoke test. Do not use it for full-video inference.

## Integration Points

VGGT planned call:

```bash
python3 ../vggt/demo_colmap.py --scene_dir outputs/<run_id>/world_reconstruction/vggt_scene
```

Prototype 4 currently performs only lightweight VGGT integration checks. During dry-run it:

- Detects whether `../vggt` exists.
- Validates expected files such as `demo_colmap.py`, `vggt/models/vggt.py`, and geometry/load utility modules.
- Records expected import paths such as `vggt.models.vggt.VGGT` without importing or instantiating the model.
- Prepares sampled frames under `outputs/<run_id>/world_reconstruction/vggt_scene/images/`.
- Records missing Python dependencies needed for real VGGT inference.

Do not run VGGT inference until the environment and checkpoint plan are explicit. The next setup steps are:

```bash
cd /afs/ece.cmu.edu/usr/zllenza/research/prototype4/vggt
# Choose the CUDA-specific PyTorch install appropriate for ece020 first.
# VGGT's docs show torch==2.3.1 and torchvision==0.18.1 as the baseline.
pip install -r requirements.txt
pip install -e .
# Optional COLMAP export/demo extras used by demo_colmap.py:
pip install trimesh pycolmap
```

VGGT's quick-start model loading path may download weights from Hugging Face. Keep `models.vggt.enabled: false` until you are ready to authenticate/download checkpoints and run GPU inference.

SAM 3 planned API:

```python
from sam3.model_builder import build_sam3_video_predictor
predictor = build_sam3_video_predictor()
```

SAM-Body4D planned call:

```bash
python3 ../sam-body4d/scripts/offline_app.py --input_video <player_crop_video_or_frame_dir>
```

Only active on-field player tracks should be passed into SAM-Body4D.

See [docs/stage_contracts.md](docs/stage_contracts.md) for artifact contracts.
