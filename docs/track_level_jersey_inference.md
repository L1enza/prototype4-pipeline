# Track-Level Jersey Inference

This stage turns crop-level jersey evidence into conservative track-level number
predictions. It does not train a model, regenerate tracking, or assign player
names unless team and number evidence are both strong.

## Inputs

Default config:

```bash
configs/nll_test4_track_jersey_inference.json
```

Default inputs:

- `outputs/nll_test4/enhanced_number_regions/enhanced_number_region_manifest.json`
- `outputs/nll_test4/jersey_ocr_clean_crops/clean_crop_metadata.json`
- `outputs/nll_test4/jersey_ocr_enhanced_local_results/crop_ocr_predictions.json`
- `outputs/nll_test4/calibrated_segment_demos/segment_20s_10s_calibrated/tracking_metadata.json`
- `outputs/nll_test4/team_assignment_demo_v2_hybrid/track_team_assignments_v2.json`
- `outputs/roster_metadata_validation/jersey_number_lookup.json`

## Evidence Selection

The stage groups enhanced number-region candidates by `track_id`, then by source
frame and source crop. Multiple preprocessing variants from the same original
crop are collapsed into one source-frame evidence item, so thresholded variants
cannot masquerade as independent temporal support.

Each track selects the best 3 to 5 distinct source frames, preferring:

- torso crops
- larger crop regions
- lower blur
- lower occlusion
- higher OCR-readiness score
- visible front/back jersey area when that heuristic is available

For each selected source frame the stage preserves:

- the original clean RGB crop
- a grayscale copy
- the best enhanced number-region view

These views are saved under `evidence_views/` in the output directory.

## Vision Backend

The implementation supports an OpenAI-compatible multimodal vision backend. It
sends several views from the same track together and asks for:

- candidate jersey number
- one-digit vs two-digit status
- full, partial, or unreadable visibility
- confidence
- evidence frame IDs
- alternative candidate
- conflict or unreadable reasons

The default config sets `allow_network` to `false`, so no API call is made unless
explicitly enabled:

```bash
.venv/bin/python scripts/run_track_level_jersey_inference.py --allow-network-vision
```

If `OPENAI_API_KEY` is absent or network vision is disabled, the stage falls back
to existing crop OCR evidence and records that the vision model was not run.

## Aggregation Policy

The stage never forces a number.

High confidence requires the same valid roster number from at least two distinct
source frames with no strong conflict.

Medium confidence requires one strong read plus compatible partial evidence and
roster validation.

Low or unknown covers conflicting reads, threshold artifacts, advertisement text,
insufficient visibility, missing team validation, or invalid roster numbers.

## Roster Validation

For `nll_test4`, the current confirmed mapping is:

- `team_a` -> Toronto Rock (`TOR`), white uniforms with blue/red trim
- `team_b` -> Oshawa FireWolves (`OSH`), dark maroon uniforms with tan/orange details

The mapping is stored in `team_label_to_abbreviation`, and the uniform notes are
stored in `team_label_metadata`. Labels such as `official` and `unknown` remain
unmapped and are not roster matched.

When a mapping is not set for a label, the stage can still report whether a
candidate number exists on any roster, but it will not assign a player name.

Player names are only attached when:

- a final jersey number is high or medium confidence
- the track's team label maps to a real roster abbreviation
- that team and jersey number resolve to exactly one roster row

## Outputs

Default output directory:

```bash
outputs/nll_test4/track_level_jersey_inference/
```

Required outputs:

- `track_jersey_predictions.json`
- `track_jersey_evidence.json`
- `track_jersey_summary.json`
- `track_jersey_review_contact_sheet.png`
- `detections_with_track_jersey_numbers.json`

`detections_with_track_jersey_numbers.json` propagates an accepted track-level
number to every detection for that track, while leaving unresolved tracks as
`null`.

## Run

```bash
.venv/bin/python scripts/run_track_level_jersey_inference.py
```

This does not run VGGT, SAM-Body4D, homography, tracking regeneration, or model
training.
