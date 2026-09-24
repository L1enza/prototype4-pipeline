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

The implementation supports an OpenAI-compatible multimodal vision backend.

In the default `per_frame` mode it reads each selected source frame in its own
request, using the single-image prompt `JERSEY_READ_PROMPT`, so every read is
independent evidence. `scripts/eval_jersey_readers.py` scores that same prompt
against the manual eval set, so a reader's measured accuracy applies directly to
this stage. Reads marked `visibility: none` are dropped.

The legacy `track` mode sends every view of a track in one request. Its answer
counts as one read: frames the model says it used are kept as
`claimed_evidence_frames` but never count as independent support.

The default config sets `allow_network` to `false`, so no request is made unless
explicitly enabled. A cloud endpoint needs the key named by `api_key_env`; a
localhost endpoint such as Ollama needs no key:

```bash
.venv/bin/python scripts/run_track_level_jersey_inference.py --allow-network-vision
.venv/bin/python scripts/run_track_level_jersey_inference.py --allow-network-vision \
    --vision-endpoint http://localhost:11434/v1/chat/completions --vision-model gemma3:12b
```

If the key is absent or network vision is disabled, the stage falls back to
existing crop OCR evidence and records that the vision model was not run.

Measured on the 130-crop eval set, small local models (gemma3:4b) and Tesseract
both invent numbers on unreadable crops, and gemma3:4b repeats the same wrong
number across frames. Score a reader with `eval_jersey_readers.py` before
trusting it here.

## Aggregation Policy

The stage never forces a number.

High confidence requires the same number, valid on the assigned team's roster,
from at least `aggregation.min_agreeing_frames` (default 2) distinct source frames
with no conflicting candidate.

Medium confidence (one strong read plus roster validation) is off by default.
Readers were measured to be confidently wrong, so a single read stays `low`
unless `aggregation.allow_single_frame_medium` is set to `true`.

After all tracks are aggregated, a number claimed by two same-team tracks whose
time spans overlap is withdrawn from both, since one team cannot field two players
with the same number at once. Non-overlapping duplicates are kept, because they
usually mean one player's track was split into fragments.

Low or unknown covers conflicting reads, threshold artifacts, advertisement text,
insufficient visibility, missing team validation, or invalid roster numbers.

## Roster Validation

Team assignment names its two colour clusters by size (`team_a` is the larger
one), not by team, so `team_a` can be either team depending on the clip and on
which tracks were found. The mapping to real teams is therefore confirmed by a
person for each team-assignment result:

```bash
# 1. Review: prints each label's track count and whether its shirts look light or dark.
python scripts/confirm_team_mapping.py

# 2. After checking the team assignment overlay or contact sheets, confirm:
python scripts/confirm_team_mapping.py --team-a TOR --team-b OSH --confirmed-by "your name" \
    --notes "team_a white with blue/red trim, team_b maroon"
```

This writes `inputs.team_mapping_confirmation`
(`configs/nll_test4_team_mapping_confirmation.json` for `nll_test4`) with a
fingerprint of every track's team label. The stage uses the mapping only when
that fingerprint matches the current team assignments. If team assignment is
re-run and any label changes, including `team_a`/`team_b` swapping, the
confirmation is reported as `stale` and must be redone. Any
`team_label_to_abbreviation` written directly into the config is ignored.

The summary's `team_mapping.status` is one of `confirmed`, `missing`, `stale`,
`invalid`, or `no_team_assignments`. Anything other than `confirmed` leaves every
team unmapped. The stage still reports whether a candidate number exists on any
roster, but it assigns no final number and no player name. Labels such as
`official` and `unknown` are never mapped.

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
