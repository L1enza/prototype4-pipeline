# Jersey Number Repository Integration Plan

## Decision summary

The linked repository is **not a jersey-number recognition repository**. The
`NVlabs/Eagle/Embodied` subtree contains **LocateAnything-3B**, a 3B-parameter
general vision-language grounding model. It accepts an RGB image plus a text
prompt and emits generated text with structured boxes or points. Its documented
OCR feature is **scene-text localization**, not a dedicated jersey-number OCR,
digit classifier, player recognizer, or roster matcher.

The lowest-risk use of Eagle is therefore optional and narrow:

1. Existing tracker selects multiple crops for each track.
2. Eagle optionally localizes likely jersey/text regions in those crops.
3. A dedicated OCR recognizer reads complete one- or two-digit numbers.
4. Prototype 4 aggregates evidence over time.
5. A separate roster matcher maps a stable number plus team evidence to a player.

Do not let Eagle assign player identity directly. Do not install Eagle's pinned
dependencies into Prototype 4's working `.venv` until compatibility has been
tested in an isolated environment.

## Repository inspection

Inspected source: [NVlabs/Eagle `Embodied`](https://github.com/NVlabs/Eagle/tree/main/Embodied)
and the [LocateAnything-3B model card](https://huggingface.co/nvidia/LocateAnything-3B).

Useful files:

| Path | Relevance |
| --- | --- |
| `Embodied/README.md` | Installation, task descriptions, worker examples, and output token format. |
| `Embodied/locateanything_worker.py` | Main inference wrapper. `LocateAnythingWorker` exposes `predict`, `detect`, `ground_multi`, `ground_text`, and `detect_text`. |
| `Embodied/pyproject.toml` | Large pinned dependency set. |
| `Embodied/eaglevl/` | Core model/training implementation; should remain external. |
| `Embodied/evaluation/` | Grounding/detection evaluations, not jersey-number evaluation. |
| `Embodied/LICENSE_MODEL` | NVIDIA model license; non-commercial research/evaluation use only. |

The worker loads `AutoTokenizer`, `AutoProcessor`, and `AutoModel` with
`trust_remote_code=True`; defaults to CUDA and BF16; and returns an `answer`
string plus optional timing/history data. It does not return calibrated box,
digit, or jersey-number confidence scores.

## Answers to integration questions

1. **What does it do?**
   LocateAnything performs natural-language visual grounding, object detection,
   point localization, and scene-text localization. It is built from a Moon-ViT
   encoder, Qwen2.5-3B-Instruct decoder, and multimodal projector.

2. **Does it recognize jersey numbers?**
   Not as a documented task. `detect_text()` asks it to locate text in box
   format. A custom prompt may generate a number, but that is experimental and
   must not be treated as a supported, calibrated jersey OCR API.

3. **Expected input?**
   A PIL/RGB image and natural-language prompt. It does not require a tracking
   video. Prototype 4 should pass player crops, preferably torso/back crops,
   rather than full broadcast frames.

4. **Output?**
   Generated text containing semantic labels and normalized box/point tokens.
   Boxes use integer coordinates in `[0, 1000]`. No documented per-box or
   per-number confidence, team ID, player ID, or roster identity is returned.

5. **Double-digit support?**
   The language model can emit multi-character text, so double digits are
   representable. The repo publishes no jersey-specific double-digit benchmark.

6. **Can it prevent one digit from a double-digit number?**
   Not reliably by itself. The adapter preserves one- or two-digit candidates as
   atomic strings and downweights a single digit when matching two-digit evidence
   exists on the same track. Final acceptance requires repeated temporal evidence.

7. **Low-resolution broadcast crops?**
   Unknown. The model card supports native-resolution images up to 2.5K, but it
   does not report jersey OCR performance on crops near 100-250 pixels wide.
Current `nll-test1` best crops are approximately 112-266 pixels wide, so this
is a substantial empirical risk. Upscaling cannot restore missing digit detail.
The current crop stage also applies a green SAM mask highlight. That is useful
for tracking review but can distort jersey color and text pixels; real OCR should
re-extract clean RGB torso/back crops from source frames using the saved boxes.

8. **Fine-tuning required?**
   A guarded zero-shot smoke test is possible, but likely insufficient for
   reliable NLL jersey recognition. Fine-tuning should wait for a labeled crop
   benchmark and comparison against a dedicated OCR baseline.

9. **Dependencies/checkpoint?**
   The repo pins `transformers==4.57.1`, `tokenizers==0.22.0`,
   `accelerate==1.5.2`, `peft==0.12.0`, `deepspeed==0.15.4`, and
   `liger_kernel==0.3.1`, with many additional packages. The checkpoint is
   `nvidia/LocateAnything-3B`. The model card reports BF16 Transformers inference
   and testing on A100/H100; an A100 4K example used 11.71 GiB with its optional
   `la_flash` path and 35.12 GiB with SDPA. Actual crop memory may differ.

10. **Lowest-risk adapter?**
    Keep Eagle as a sibling checkout with a separate virtual environment and a
    JSON boundary. Start with crop selection and output contracts in the current
    project. Only then run a tiny opt-in Eagle test. A dedicated OCR backend can
    later consume Eagle-localized torso/text regions without changing tracking.

## ECE setup assessment

Inspection on `ece020` found:

- `../Eagle/Embodied` is not cloned.
- The project `.venv` has PyTorch, Pillow, OpenCV, decord, and timm.
- It lacks transformers, tokenizers, accelerate, peft, and deepspeed.
- CUDA was unavailable in the inspection process; GPU availability depends on
  the ECE allocation/session.

No clone, package install, checkpoint download, model construction, or inference
was performed. If Eagle is tested later, prefer:

```bash
git clone https://github.com/NVlabs/Eagle.git ../Eagle
python3 -m venv .venv-eagle
.venv-eagle/bin/pip install -e ../Eagle/Embodied
```

Review `LICENSE_MODEL` before any use beyond academic/non-profit research. The
model license is non-commercial, even though the repository's Python package
metadata identifies an Apache software license.

## Prototype 4 adapter contract

Initial script:

```text
scripts/run_jersey_number_repo_smoke.py
```

Default inspection-safe command:

```bash
.venv/bin/python scripts/run_jersey_number_repo_smoke.py
```

This command ranks a small number of existing crops and writes all required
schemas and the review contact sheet. It does not import Eagle or load weights.

Future explicitly guarded experiment:

```bash
.venv-eagle/bin/python scripts/run_jersey_number_repo_smoke.py \
  --eagle-repo ../Eagle/Embodied \
  --run-inference \
  --allow-download-weights \
  --device cuda \
  --max-tracks 3 \
  --crops-per-track 3
```

The two flags are intentionally separate. Remote model loading cannot occur with
only `--run-inference`.

Outputs:

```text
outputs/jersey_number_smoke/
  crop_predictions.json
  track_number_predictions.json
  jersey_number_contact_sheet.png
  jersey_number_smoke_summary.json
```

Per-crop records preserve crop path, track/frame/mask IDs, crop quality fields,
raw model answer, parsed atomic number candidates, recognizer confidence (null
when unavailable), and error/status fields. Per-track records preserve every
observation, weighted votes, temporal consensus, partial-digit protection,
accept/reject status, and a future `roster_match` field.

## First real smoke test plan

1. Manually inspect the generated contact sheet and label which crops show a
   front, back, readable number, partial number, or no number.
2. Re-extract clean RGB crops without the SAM visualization tint for any crop
   selected for actual OCR evaluation.
3. Clone Eagle into `../Eagle` and build an isolated `.venv-eagle`; do not alter
   the working Prototype 4 environment.
4. Run only 3 tracks x 3 crops on an allocated CUDA node.
5. Retain raw answers and reject ambiguous/multiple candidates.
6. Measure complete-number accuracy, digit recall, double-digit truncation, and
   abstention on unreadable crops.
7. Compare against a dedicated OCR baseline before deciding whether Eagle adds
   value as a text/torso localizer.

## Later roster identity connection

Roster matching should consume a conservative track-level record:

```json
{
  "track_id": 12,
  "team_id": "optional-team-evidence",
  "jersey_number": "27",
  "temporal_consensus": 0.84,
  "usable_prediction_count": 5,
  "status": "accepted",
  "roster_match": null
}
```

Only map this to a roster row when team and number agree and the consensus passes
configured thresholds. Duplicate numbers, traded players, goalkeepers, officials,
and uncertain team color must remain unresolved rather than forcing an identity.

## Proposed follow-on modules

- `prototype4_pipeline/integrations/eagle_text_localizer.py`
- `prototype4_pipeline/integrations/jersey_ocr.py`
- `prototype4_pipeline/identity/temporal_number_aggregation.py`
- `prototype4_pipeline/identity/roster_matcher.py`
- `scripts/evaluate_jersey_number_crops.py`

These names keep external localization, OCR recognition, temporal aggregation,
and roster identity as replaceable modules.
