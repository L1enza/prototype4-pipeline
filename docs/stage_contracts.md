# Prototype 4 Stage Contracts

This project stores explicit intermediate artifacts so model-backed stages can be swapped in later without changing downstream consumers.

## 1. Video Ingestion

Input: raw game footage path.

Output: `outputs/<run_id>/metadata/video_ingestion.json`

Fields:

- `video_path`
- `fps`
- `frame_count`
- `width`
- `height`
- `duration_sec`

## 2. Frame Sampling

Input: raw video and sampling config.

Output:

- `outputs/<run_id>/sampled_frames/frame_*.jpg`
- `outputs/<run_id>/metadata/sampled_frames.json`

Each frame record includes `frame_index`, `source_frame_index`, `timestamp_sec`, `path`, `width`, and `height`.

## 3. VGGT World Reconstruction

Input: sampled frames.

Output directory: `outputs/<run_id>/world_reconstruction/`

Dry-run manifest: `world_reconstruction_manifest.json`

Dry-run behavior:

- Creates `outputs/<run_id>/world_reconstruction/vggt_scene/images/`.
- Links or copies sampled frames into that image folder.
- Detects `../vggt` and records whether expected repo files are present.
- Records available VGGT entry points:
  - `../vggt/demo_colmap.py`
  - `../vggt/demo_viser.py`
  - `../vggt/demo_gradio.py`
  - Python API: `vggt.models.vggt.VGGT`
- Records expected import paths.
- Records dependency availability via lightweight module checks.
- Runs `scripts/check_vggt_imports.py` logic to try importing minimum VGGT modules when dependencies are available.
- Does not instantiate `VGGT`, download model weights, or run inference.

Manifest fields added by the VGGT integration:

- `status`: `placeholder` in dry-run mode.
- `dry_run`: true when model execution is disabled.
- `repo_present`: whether `../vggt` exists.
- `available_entry_points`: known demo/API entry points and their expected inputs/outputs.
- `validation.status`: one of `ready_for_dry_run`, `repo_incomplete`, or `dependencies_missing_for_inference`.
- `validation.expected_files`: required local VGGT files and presence checks.
- `validation.expected_import_paths`: model and utility import paths needed later.
- `validation.import_smoke_test`: result from adding `../vggt` to `sys.path` and importing minimum modules.
- `validation.missing_dependencies_for_inference`: packages needed before real VGGT execution.
- `input_summary`: frame count, timestamp range, source video, and VGGT scene image layout.

Future model outputs:

- camera intrinsics/extrinsics
- depth maps
- point maps or scene point cloud
- COLMAP sparse export
- lacrosse field plane/homography/alignment metadata
- estimated player footpoints in field coordinates

Wrapper target: `../vggt/demo_colmap.py`

Expected VGGT scene layout:

```text
outputs/<run_id>/world_reconstruction/vggt_scene/
├── images/
│   └── frame_*.jpg
└── sparse/                  # created by VGGT later, not by dry-run
    ├── cameras.bin
    ├── images.bin
    ├── points3D.bin
    └── points.ply
```

Next setup steps before enabling VGGT:

```bash
cd /afs/ece.cmu.edu/usr/zllenza/research/prototype4/vggt
pip install -r requirements.txt
pip install -e .
pip install trimesh pycolmap
```

Select the correct CUDA-specific PyTorch build for the ECE machine before relying on GPU inference. Do not enable VGGT in `configs/*.yaml` until checkpoint download/authentication is intentional.

Import smoke-test setup only:

```bash
cd /afs/ece.cmu.edu/usr/zllenza/research/prototype4/prototype4_pipeline
source .venv/bin/activate
pip install -r requirements/vggt-smoke.txt
.venv/bin/python scripts/check_vggt_imports.py
```

The smoke test may import `torch` and VGGT Python modules, but it must not instantiate the model, call `from_pretrained`, download weights, or process frames. The final smoke dependencies include `hydra-core` and LightGlue installed with `pip install git+https://github.com/cvg/LightGlue.git`.

Guarded tiny inference smoke script, not run by dry-run:

```bash
.venv/bin/python scripts/run_vggt_inference_smoke.py --run-id nll-test1 --frame-count 4 --allow-download-weights
```

Contract for `scripts/run_vggt_inference_smoke.py`:

- Uses only 4 to 8 sampled frames from an existing run.
- Requires `--allow-download-weights` before `VGGT.from_pretrained(...)` is called.
- Instantiates VGGT only inside that script.
- Defaults to CUDA and fails gracefully if CUDA is unavailable; `--device cpu` must be explicit.
- Writes only lightweight metadata first: prediction keys, tensor shapes, device, frame count, and output paths.
- Writes under `outputs/<run_id>/world_reconstruction/vggt_inference_smoke/`.
- Must not replace normal dry-run behavior.

## 4. SAM 3 Player Segmentation and Tracking

Input: raw video or sampled frames plus text prompts.

Output directories:

- `outputs/<run_id>/player_masks/`
- `outputs/<run_id>/player_tracks/`

Track fields:

- `track_id`
- `observations[]`
- `frame_index`
- `timestamp_sec`
- `bbox_xyxy`
- `mask_path`
- `mask_area_px`
- `field_overlap`
- `classification`
- `rejection_scores`
- `metadata_slots`

Wrapper target: `sam3.model_builder.build_sam3_video_predictor`

## 5. Active-Player Filtering

Input: SAM 3 tracks, VGGT field geometry.

Output: `outputs/<run_id>/player_tracks/active_player_tracks.json`

Filtering hooks:

- field-boundary filtering
- referee rejection
- bench/crowd suppression
- persistence thresholding
- future jersey-number recognition
- future roster/player metadata
- future heatmap generation

Only tracks accepted here should move into player crop extraction and SAM-Body4D.

## 6. Player Crop Extraction

Input: sampled frames and active-player tracks.

Output:

- `outputs/<run_id>/player_crops/<track_id>/frame_*.jpg`
- `outputs/<run_id>/player_crops/crops_manifest.json`

## 7. SAM-Body4D Mesh Reconstruction

Input: per-track player crops or per-track crop videos.

Output directory: `outputs/<run_id>/player_meshes/`

Future model outputs:

- per-player mesh sequence
- temporal pose/shape parameters
- mesh-root trajectory
- mesh-to-field alignment metadata

Wrapper target: `../sam-body4d/scripts/offline_app.py`

## 8. Coordinate Alignment

Input: VGGT camera/field geometry, active tracks, SAM-Body4D mesh outputs.

Output: `outputs/<run_id>/metadata/coordinate_alignment.json`

Expected data:

- 2D mask/box footpoints
- VGGT camera projection data
- field-plane transform
- field-space player coordinates
- mesh root positions in field coordinates

## 9. Visualization Export

Input: world reconstruction, active tracks, meshes, coordinate alignment.

Output directory: `outputs/<run_id>/visualizations/`

Planned outputs:

- world reconstruction viewer
- player mesh viewer
- combined field/world/player scene

## 10. Saved Metadata and Artifacts

All stages must write JSON manifests into `outputs/<run_id>/metadata/` or the stage-specific output folder. Manifests should prefer stable IDs and relative/absolute file paths over hidden process state.
