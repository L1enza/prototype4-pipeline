# Andrew `lacrosse_sam` Integration Plan

Inspection target:

- Repo: `https://github.com/andrewkang12345/lacrosse_sam`
- Inspected commit: `effff4019b9858b24192460e70c5746918fe2b14`
- Commit summary: `effff40 2026-06-25T20:11:49+00:00 Add source-fps CoTracker mask viewer`
- Local inspection clone: `/tmp/lacrosse_sam_inspect`

This is an integration plan only. Do not vendor Andrew's repo into Prototype 4, do not overwrite `nll-test1` or `nll_test4` outputs, and keep manual homography available as a fallback/debug path.

## Executive Summary

Andrew's repo already contains the missing conceptual bridge we care about:

`VGGT reconstruction -> fitted floor plane -> aligned 200 ft x 85 ft NLL field coordinates -> per-frame player floor positions`

The most useful outputs for Prototype 4 are not the mesh outputs yet. The best first integration target is one of Andrew's bird's-eye JSON files that already contains `floor_xy_ft` per player per frame. A small adapter can convert those field-foot coordinates into our existing `projected_player_points.json` format, then reuse our heatmap and side-by-side renderers unchanged or with only a small `field_xy_ft` input mode.

SAM-Body4D should remain later-stage. It can provide better body/foot geometry, but Andrew's non-mesh VGGT/SAM3 and CoTracker/VGGT outputs are enough for the first smoke bridge.

## Major Pipeline Stages In Andrew's Repo

1. Clip preparation
   - Script: `scripts/prepare_lacrosse_clip.py`
   - Purpose: sample a source video into frame directories and prompt contact sheets.
   - Typical outputs: frame images, prompt manifest, preview MP4.

2. Player and field segmentation
   - Scripts: `scripts/track_sam3_text.py`, `scripts/run_sam3_text_chunked.py`
   - Prompts used in the README include `player`, `green field`, `white lines on green field`, and `yellow outline around field border`.
   - Outputs:
     - SAM3 JSON with per-frame `object_ids`, `scores`, `boxes`
     - label masks as `.png`
     - per-instance masks as `.npz` containing `object_ids` and `masks`
     - review MP4s

3. Optional team/player appearance labeling
   - Scripts: `scripts/classify_sam3_teams_by_torso_color.py`, `scripts/classify_sam3_teams_by_transreid.py`
   - Purpose: team-color or ReID clustering of SAM3 tracked objects.
   - Outputs: SAM3-style JSON augmented with `teams`, `team_colors`, object-level labels, and review videos.

4. Image-space homography and dynamic homography tools
   - Scripts include:
     - `scripts/floor_free_click_annotator.py`
     - `scripts/floor_feature_annotator.py`
     - `scripts/fit_floor_homography_from_feature_clicks.py`
     - `scripts/estimate_dynamic_floor_homographies.py`
     - `scripts/fit_floor_homography_from_tracked_landmarks.py`
     - `scripts/render_birds_eye_locations.py`
   - These are close to our manual/multi-homography work and support per-frame dynamic homographies from tracked landmarks.
   - Useful, but not the main pivot if VGGT field fitting works.

5. VGGT reconstruction
   - Scripts: `scripts/run_vggt_reconstruction_clip.py`, `scripts/run_vggt_birds_eye.py`
   - VGGT compact output:
     - `vggt_predictions_compact.npz`
     - arrays: `frame_indices`, `extrinsic`, `intrinsic`, `depth_map`, `depth_conf`
     - `world_points` are reconstructed from depth/extrinsics/intrinsics via VGGT utilities, not always stored directly in the compact NPZ.
   - Metadata:
     - `metadata.json`
     - schema: `vggt_reconstruction_clip_v1`

6. VGGT field-plane and NLL field fitting
   - Main script: `scripts/fit_vggt_field_from_sam3_text_masks.py`
   - Inputs:
     - VGGT compact predictions
     - SAM3 field masks for green field, white lines, and yellow outline
   - Method:
     - sample SAM3 field-mask pixels
     - lift those pixels into VGGT 3D points
     - fit a robust floor plane
     - project field evidence into plane coordinates
     - fit a constrained 200 ft x 85 ft NLL synthetic field to the plane
   - Key output:
     - `field_fit_vggt_sam3_text_masks.json`
   - Important fields:
     - `schema`: `vggt_sam3_text_field_fit_v1`
     - `floor_plane.center`
     - `floor_plane.basis_u`
     - `floor_plane.basis_v`
     - `plane_to_floor_homography`
     - `alignment_metrics`
     - `per_frame_counts`

7. VGGT-derived player field coordinates
   - Script: `scripts/run_vggt_birds_eye.py`
   - Uses player masks, VGGT depth/world points, fitted floor plane, and `plane_to_floor_homography`.
   - Output:
     - `birds_eye_player_locations_vggt.json`
     - schema: `vggt_birds_eye_v1`
     - per frame: `players[]`
     - per player: `object_id`, `team`, `team_color`, `floor_xy_ft`, `source`, `sampled_points`, `median_depth_conf`

8. CoTracker + VGGT trajectories
   - Script: `scripts/run_cotracker_vggt_player_trajectories.py`
   - Uses SAM3 player masks to seed CoTracker points, samples VGGT depth at tracked pixels, then maps to fitted NLL field coordinates.
   - Output:
     - `cotracker_vggt_player_trajectories.json`
     - schema: `cotracker_vggt_player_trajectories_v1`
     - per frame: `players[]`
     - per player: `object_id`, `floor_xy_ft`, `camera_xy`, `visible_track_points`, `mask_gated_track_points`, `vggt_points`, `median_depth_conf`, `color`
   - This is likely the best first integration target when identity continuity matters.

9. SAM-Body4D mesh generation and VGGT mesh placement
   - Scripts:
     - `scripts/run_sam_body4d_from_sam3_boxes.py`
     - `scripts/run_sam_body4d_from_masks.py`
     - `scripts/render_sam_body4d_vggt_birds_eye.py`
   - Outputs:
     - mesh folders under `mesh_4d_individual/<object_id>/*.ply`
     - focal metadata under `focal_4d_individual/<object_id>/*.json`
     - bird's-eye mesh/player JSON:
       - `birds_eye_player_locations_sam_body4d_vggt.json`
       - schema: `sam_body4d_vggt_birds_eye_v1`
       - per player includes `floor_xy_ft`, `mesh_foot_pixels`, `vggt_samples`, `mesh_fit_points`, `mesh_render_points`
   - This is useful later, but should not be the first adapter because it requires heavy SAM-Body4D jobs.

## What Andrew's Repo Outputs

Andrew's repo can output the following artifacts that matter to Prototype 4:

- Camera intrinsics/extrinsics: yes, in `vggt_predictions_compact.npz`.
- Depth and confidence: yes, in `vggt_predictions_compact.npz`.
- VGGT point cloud/world points: indirectly yes; reconstructed from `depth_map`, `extrinsic`, and `intrinsic` using `vggt.utils.geometry.unproject_depth_map_to_point_map`.
- Field plane: yes, in `field_fit_vggt_sam3_text_masks.json`.
- Plane-to-NLL-field transform: yes, `plane_to_floor_homography`.
- Aligned synthetic NLL field: yes, via `nll_field_geometry.py` and render helpers using `FLOOR_LENGTH_FT = 200.0`, `FLOOR_WIDTH_FT = 85.0`.
- Player masks: yes, SAM3 instance `.npz` masks and SAM3 JSON.
- Player tracks: yes, SAM3 object IDs; optionally CoTracker trajectory objects.
- Body meshes: yes, via SAM-Body4D and/or 4D-Humans scripts.
- Per-frame player positions: yes, several outputs include `floor_xy_ft`.

## Does It Already Produce Player Coordinates On The Field?

Yes, for the outputs below:

- `birds_eye_player_locations_vggt.json`
  - Produced by `scripts/run_vggt_birds_eye.py`
  - Per-player field coordinates: `floor_xy_ft`
  - Source: SAM3 mask bottom pixels lifted through VGGT depth and transformed through the fitted field plane.

- `cotracker_vggt_player_trajectories.json`
  - Produced by `scripts/run_cotracker_vggt_player_trajectories.py`
  - Per-player field coordinates: `floor_xy_ft`
  - Source: CoTracker points seeded from SAM3 player masks, lifted through VGGT depth and transformed to NLL field coordinates.

- `birds_eye_player_locations_sam_body4d_vggt.json`
  - Produced by `scripts/render_sam_body4d_vggt_birds_eye.py`
  - Per-player field coordinates: `floor_xy_ft`
  - Source: SAM-Body4D mesh foot/body samples projected into VGGT scene and onto the fitted field.

The first two are enough for our heatmap adapter. The SAM-Body4D path should remain later.

## Missing Bridge Into Prototype 4

Our heatmap smoke stage currently expects projected points in template-image pixel coordinates:

```json
{
  "frame_index": 0,
  "track_id": 1,
  "projected_field_point": {"x": 123.4, "y": 567.8},
  "inside_field_template_bounds": true
}
```

Andrew's strongest outputs use field coordinates in feet:

```json
{
  "frame": 0,
  "players": [
    {
      "object_id": 3,
      "floor_xy_ft": [101.2, 40.5]
    }
  ]
}
```

So the adapter needs a field-coordinate-to-template-pixel map:

`NLL feet coordinates, 0..200 x 0..85 -> our top-down PNG template pixels`

This should be explicit and metadata-preserving. The adapter should save both:

- `field_xy_ft`: Andrew's original field coordinates.
- `projected_field_point`: template pixel coordinates for our existing heatmap renderer.

Do not silently discard the original `floor_xy_ft`.

## Proposed Adapter Scripts

1. `scripts/adapt_andrew_field_locations_to_heatmap.py`
   - Input schemas:
     - `vggt_birds_eye_v1`
     - `cotracker_vggt_player_trajectories_v1`
     - `sam_body4d_vggt_birds_eye_v1`
   - Inputs:
     - `--andrew-json`
     - `--field-template assets/field_templates/nll_field_topdown.png`
     - `--template-field-bounds-px x0,y0,x1,y1`
     - `--run-id`
     - `--output-dir`
   - Output:
     - `projected_player_points.json` in our existing heatmap-compatible format
     - `adapter_metadata.json`
     - `adapter_summary.json`
     - optional `adapted_points_topdown.png`
   - Mapping:
     - `track_id = object_id`
     - `frame_index = frame`
     - `field_xy_ft = floor_xy_ft`
     - `projected_field_point.x = x0 + floor_x_ft / 200.0 * (x1 - x0)`
     - `projected_field_point.y = y0 + floor_y_ft / 85.0 * (y1 - y0)`
   - Keep optional fields:
     - `source`
     - `team`
     - `team_color`
     - `median_depth_conf`
     - `vggt_points`
     - `sampled_points`
     - `camera_xy`

2. `scripts/run_andrew_vggt_heatmap_smoke.py`
   - Thin orchestrator after adapter exists.
   - Calls:
     - adapter script
     - our `scripts/run_heatmap_smoke.py`
     - our `scripts/render_side_by_side_field_tracking_smoke.py` only if camera-frame overlays are available
   - Should not run VGGT, SAM3, CoTracker, or SAM-Body4D.

3. Optional later: `scripts/run_andrew_vggt_world_smoke.py`
   - Runs only a very short Andrew VGGT/SAM3 field-fit path on a small frame subset.
   - Must be explicitly guarded because it may download/run models.
   - Not part of the first adapter smoke.

## Safest First Smoke Test

First smoke test should be adapter-only, no heavy inference:

1. Obtain or create one small Andrew output JSON containing `floor_xy_ft`.
   - Preferred: `cotracker_vggt_player_trajectories.json`
   - Acceptable: `birds_eye_player_locations_vggt.json`
   - Later: `birds_eye_player_locations_sam_body4d_vggt.json`

2. Run a dry adapter conversion:

```bash
.venv/bin/python scripts/adapt_andrew_field_locations_to_heatmap.py \
  --andrew-json /path/to/andrew/output/cotracker_vggt_player_trajectories.json \
  --run-id nll_test4 \
  --field-template assets/field_templates/nll_field_topdown.png \
  --template-field-bounds-px 0,0,1600,700 \
  --output-dir outputs/nll_test4/andrew_vggt_adapter_smoke/
```

The template bounds above are placeholders. Before using real heatmaps, confirm the exact playable-field bounds inside `assets/field_templates/nll_field_topdown.png`.

3. Run our existing heatmap script against the adapted projected points:

```bash
.venv/bin/python scripts/run_heatmap_smoke.py \
  --run-id nll_test4 \
  --projected-points outputs/nll_test4/andrew_vggt_adapter_smoke/projected_player_points.json \
  --field-template assets/field_templates/nll_field_topdown.png \
  --calibration-metadata outputs/nll_test4/andrew_vggt_adapter_smoke/adapter_metadata.json \
  --output-dir outputs/nll_test4/andrew_vggt_adapter_smoke/heatmap/
```

This proves whether Andrew's field-coordinate outputs can drive our heatmap renderer without re-running our manual homography.

## If Andrew Outputs Are Not Available Yet

Do not jump straight to SAM-Body4D. The lowest-risk inference path is:

1. Use a tiny selected frame directory from the same clean segment.
2. Run SAM3 field/player masks at low cadence.
3. Run VGGT only on the same tiny frame set.
4. Run `fit_vggt_field_from_sam3_text_masks.py`.
5. Run `run_vggt_birds_eye.py` or `run_cotracker_vggt_player_trajectories.py`.
6. Adapter into our heatmap format.

This should be a separate explicit smoke command and should write under a new folder, for example:

```text
outputs/nll_test4/andrew_vggt_field_coordinate_smoke/
```

Do not overwrite:

- `outputs/nll-test1/`
- `outputs/nll_test4/calibrated_segment_demos/`
- `outputs/nll_test4/field_calibration_smoke/`
- `outputs/nll_test4/multi_homography_demos/`

## Can VGGT Solve Camera Pans Better Than Manual Homography?

Potentially yes, but only if the VGGT reconstruction remains coherent across the pan/zoom segment.

Why it can help:

- Static image homography assumes one fixed camera projection. It breaks when the broadcast camera pans/zooms.
- VGGT estimates cameras and scene geometry across frames. If the same field plane is reconstructed in a shared 3D coordinate system, each frame can use its own predicted camera geometry and depth while still mapping player points onto one fitted field plane.
- Andrew's `plane_to_floor_homography` maps reconstructed plane coordinates to canonical NLL field feet, which is exactly what our heatmap needs.

Why it may still fail:

- VGGT may drift or scale inconsistently on broadcast footage with fast pans, zooms, motion blur, compression, and repeated turf texture.
- Hard camera cuts should be separate VGGT/view segments.
- Player/body points are dynamic objects, so using player pixels for reconstruction can be noisy. Field masks and line masks are better for fitting the plane.
- Field-line visibility may be sparse or occluded.
- Fitted field alignment may need anchors or refinement when only a partial zone is visible.
- Andrew's all-player SAM3 path may include bench players unless we add our active-player filtering/referee suppression upstream or downstream.

## Recommended Integration Path

Phase 1: adapter-only smoke.

- Input: Andrew `floor_xy_ft` JSON.
- Output: our `projected_player_points.json`.
- Render: our heatmap and field visualization.
- No VGGT/SAM3/SAM-Body4D inference.

Phase 2: short VGGT field-coordinate smoke on one selected segment.

- Use our selected `nll_test4` short segment.
- Run only low-cadence VGGT/SAM3 field fitting.
- Prefer `run_vggt_birds_eye.py` or `run_cotracker_vggt_player_trajectories.py`.
- Compare Andrew-VGGT heatmap against our manual/multi-homography heatmap.

Phase 3: improve player filtering and identity.

- Use our active-field filter/referee cleanup before heatmap rendering when possible.
- Keep `object_id` as `track_id` but record mapping metadata.
- Compare trajectory continuity against our stabilized tracker.

Phase 4: SAM-Body4D.

- Only after field projection and heatmap validation.
- Use `birds_eye_player_locations_sam_body4d_vggt.json` as a richer source of player/base positions.

## Risks And Unknowns

- Template coordinate mapping: Andrew outputs field feet; our template uses pixels. We need an explicit, validated foot-to-template-pixel map.
- Frame index alignment: Andrew's `frame` values refer to its frame directory ordering. We must map those back to our tracking/demo frame indices and timestamps if mixing outputs.
- Identity semantics: Andrew `object_id` may not match our `track_id`; adapter should preserve both when both exist.
- Active-player filtering: Andrew's `player` prompt can include bench/substitute players. Our cleanup stage remains useful.
- Referees: neither VGGT nor field-plane fitting solves referee rejection. Appearance filtering is still needed.
- Camera cuts: VGGT or multi-keyframe homography should segment by view; do not force one world/fit across unrelated broadcast cuts.
- Runtime/dependencies: CoTracker and SAM-Body4D add dependencies and runtime. Adapter-first avoids this.
- Scale/orientation validation: `plane_to_floor_homography` must be visually checked against the NLL template and known landmarks.

## Exact Next Commands

These commands are documentation only; do not run heavy jobs without explicit approval.

Adapter-only once an Andrew output JSON exists:

```bash
.venv/bin/python scripts/adapt_andrew_field_locations_to_heatmap.py \
  --andrew-json /path/to/andrew/output/cotracker_vggt_player_trajectories.json \
  --run-id nll_test4 \
  --field-template assets/field_templates/nll_field_topdown.png \
  --template-field-bounds-px 0,0,1600,700 \
  --output-dir outputs/nll_test4/andrew_vggt_adapter_smoke/

.venv/bin/python scripts/run_heatmap_smoke.py \
  --run-id nll_test4 \
  --projected-points outputs/nll_test4/andrew_vggt_adapter_smoke/projected_player_points.json \
  --field-template assets/field_templates/nll_field_topdown.png \
  --calibration-metadata outputs/nll_test4/andrew_vggt_adapter_smoke/adapter_metadata.json \
  --output-dir outputs/nll_test4/andrew_vggt_adapter_smoke/heatmap/
```

Potential later VGGT smoke, guarded and short:

```bash
# Documentation only. Do not run until approved.
python /path/to/lacrosse_sam/scripts/run_vggt_reconstruction_clip.py \
  --vggt-repo ../vggt \
  --frames-dir outputs/nll_test4/some_low_fps_frames \
  --output-dir outputs/nll_test4/andrew_vggt_field_coordinate_smoke/vggt \
  --max-frames 12
```

## Bottom Line

Andrew's repo does not just make a 3D visual. It already contains a plausible path from VGGT 3D reconstruction to canonical NLL field coordinates. The safest integration is not to merge the repo. Instead, add a narrow adapter from Andrew's `floor_xy_ft` player-location JSON into our `projected_player_points.json`, then reuse our heatmap renderer and compare against manual/multi-homography outputs.
