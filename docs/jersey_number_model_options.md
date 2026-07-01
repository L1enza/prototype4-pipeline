# Jersey Number Model Options

## Target architecture

```text
track
  -> many clean RGB full-body/torso crops
  -> per-crop jersey-number predictions
  -> temporal confidence aggregation
  -> team + roster constraints
  -> player identity or explicit unknown
```

No single crop should assign identity. The system should preserve abstentions,
partial numbers, alternate candidates, model confidence, crop quality, and the
source frame for every observation.

## Immediate recommendation

Start with an OCR baseline on the new clean crops, while treating it as a
measurement tool rather than the final architecture. In parallel, label a small
NLL crop benchmark with these states:

- complete readable number
- partial number
- number present but unreadable
- front/side/no number visible
- referee/non-player
- track identity contamination

This reveals whether the current bottleneck is crop selection, digit
localization, recognition, resolution, or tracking. Do not fine-tune until this
error breakdown exists.

## Option 1: OCR baseline on clean crops

Run a scene-text OCR detector/recognizer on torso crops, constrain accepted text
to one or two digits, and retain raw detections and confidence. PaddleOCR is a
realistic baseline because it provides separate text detection and recognition
components, supports fine-tuning, and is substantially smaller than a general
3B VLM. See the [official PaddleOCR repository](https://github.com/PaddlePaddle/PaddleOCR)
and [recognition training documentation](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version2.x/ppocr/model_train/recognition.en.md).

Advantages:

- fastest way to establish a measurable baseline
- returns text and confidence rather than general VLM prose
- can run per crop and preserve alternative predictions
- no jersey-specific labels required for the first test

Risks:

- generic OCR is trained mostly on larger, flatter text
- jersey digits are curved, occluded, stylized, and motion blurred
- detector may split a double-digit number or miss one digit
- front logos, advertisements, and uniform text can produce false positives

Acceptance rule: never accept a one-digit prediction merely because it has the
highest single-frame score. Aggregate complete strings over the track and retain
digit boxes so adjacent digits can be reconsidered.

## Option 2: Digit detector plus digit grouping

Train or adapt a lightweight detector with classes `0` through `9`. Detect each
visible digit, then group detections into a jersey number using horizontal
alignment, scale similarity, spacing, reading order, and torso containment.

Advantages:

- explicit control over double-digit grouping
- partial visibility is represented naturally
- digit boxes make failures easy to inspect
- a detector can learn stylized NLL fonts and perspective

Risks:

- requires digit-box annotations
- grouping rules can join unrelated marks or split a number
- very small digits may remain below the detector's useful resolution
- duplicated digits and severe overlap need careful non-maximum suppression

This is the strongest next step if generic OCR frequently finds only one digit
of double-digit jerseys.

## Option 3: Fine-tuned jersey-number recognizer

Fine-tune a crop-level model to predict `00`-`99`, separate left/right digit
distributions, or a sequence with an explicit `not_visible` state. Research on
low-resolution broadcast footage emphasizes keyframe selection because jersey
numbers are absent or unreadable in most frames. See
[Jersey Number Recognition using Keyframe Identification from Low-Resolution Broadcast Videos](https://arxiv.org/abs/2309.06285)
and [A General Framework for Jersey Number Recognition in Sports Video](https://arxiv.org/abs/2405.13896).

Advantages:

- directly optimizes the real task
- can learn NLL uniforms, fonts, blur, and camera characteristics
- can include visibility/orientation prediction

Risks:

- requires a balanced labeled dataset and held-out games/venues
- a 100-class classifier can hide partial-digit errors
- class imbalance follows roster number frequencies
- track identity errors can contaminate weak labels

Prefer a model with explicit digit positions and a `not readable` output over a
forced 0-99 classification. Orientation-aware methods are relevant because
front, side, and back views have different number visibility; see
[Generalized Jersey Number Recognition Using Multi-task Learning With Orientation-guided Weight Refinement](https://arxiv.org/abs/2406.01033).

## Option 4: Temporal aggregation over each track

Temporal aggregation is required regardless of the recognizer. Keep multiple
high-quality, time-separated crops and combine their evidence rather than
choosing one best frame. Track-level sequence models have been used for hockey
player identification; see
[Ice hockey player identification via transformers and weakly supervised learning](https://arxiv.org/abs/2111.11535).

Initial non-trained aggregation should:

1. Weight predictions by OCR confidence and crop quality.
2. Require repeated complete-string agreement.
3. Downweight correlated adjacent frames.
4. Prefer a supported two-digit candidate over a conflicting partial one-digit
   candidate, without automatically discarding the one-digit evidence.
5. Reject consensus when the track likely contains an identity switch.
6. Emit `unknown` when evidence is insufficient.

Later, a learned sequence model can consume crop features, orientation,
visibility, and digit logits, but only after the simpler voter is benchmarked.

## Option 5: Roster-constrained player matching

Roster matching is the final identity layer, not part of OCR. Inputs should be:

- accepted track-level jersey-number distribution
- team/color evidence
- game roster and active-player list
- optional position, shift, and event-time constraints

Use the roster to remove impossible candidates, not to turn weak OCR into a
confident identity. Duplicate numbers across teams require reliable team
evidence. A referee or uncertain team assignment must remain unresolved.

Proposed output:

```json
{
  "track_id": 12,
  "number_candidates": [
    {"number": "27", "confidence": 0.84},
    {"number": "7", "confidence": 0.11}
  ],
  "team_candidates": [
    {"team_id": "home", "confidence": 0.92}
  ],
  "player_id": "roster-player-id-or-null",
  "identity_status": "accepted_or_uncertain",
  "evidence_frames": [4, 9, 17, 28]
}
```

## Practical sequence

1. Extract clean RGB crops from the original video.
2. Manually label a small crop/track evaluation set.
3. Run a generic OCR baseline without training.
4. Measure complete-number accuracy, one-digit truncation, false positives, and
   abstention quality by crop resolution and orientation.
5. Add digit detection/grouping if double-digit splitting dominates.
6. Fine-tune only after the failure mode and annotation target are clear.
7. Add conservative temporal aggregation.
8. Add team and roster constraints last.

## Data contracts to preserve

Per crop:

- track ID, frame index, timestamp, source video frame
- full-body and torso crop paths
- original bbox and crop box
- resolution, sharpness, blur, overlap, and quality scores
- orientation/visibility labels when available
- digit boxes, raw text, candidates, and recognizer confidence

Per track:

- every per-crop observation
- weighted number distribution
- temporal coverage and number of independent observations
- identity-switch/quality flags
- team evidence
- roster match status and reasons

Raw tracking and crop data must remain immutable when recognizers or aggregation
heuristics change.
