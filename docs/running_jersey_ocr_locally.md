# Running Jersey OCR Locally

The ECE package contains pre-extracted number-region PNGs and the metadata needed
to reconnect every OCR result to its track and source frame. It does not contain
player-name assignments, trained jersey models, or the original video.

## 1. Download the package

From a terminal on the Mac:

```bash
scp zllenza@ece020.ece.cmu.edu:/afs/ece.cmu.edu/usr/zllenza/research/prototype4/prototype4_pipeline/outputs/nll_test4/jersey_ocr_portable_package/jersey_ocr_portable_package.zip ~/Downloads/
cd ~/Downloads
unzip jersey_ocr_portable_package.zip
```

The archive expands to `jersey_ocr_portable_package/`.

## 2. Install Tesseract and Python dependencies

```bash
brew install tesseract
cd ~/Downloads/jersey_ocr_portable_package
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-local.txt
```

Confirm the OCR binary is visible:

```bash
tesseract --version
```

## 3. Run OCR

The zip includes a copy of the portable runner, so it can run without a separate
Prototype 4 checkout:

```bash
python scripts/run_jersey_number_ocr_baseline.py \
  --crop-metadata clean_crop_metadata.json \
  --visibility-predictions crop_visibility_predictions.json \
  --number-regions-dir number_regions \
  --output-dir jersey_ocr_local_results \
  --engine tesseract
```

From a Prototype 4 checkout with the package beside `scripts/`, the equivalent
command is:

```bash
python scripts/run_jersey_number_ocr_baseline.py \
  --crop-metadata jersey_ocr_portable_package/clean_crop_metadata.json \
  --visibility-predictions jersey_ocr_portable_package/crop_visibility_predictions.json \
  --number-regions-dir jersey_ocr_portable_package/number_regions \
  --output-dir jersey_ocr_local_results \
  --engine tesseract
```

The runner consumes the packaged number regions directly. Absolute ECE paths in
the snapshot metadata are retained as provenance and are not opened in portable
mode.

## 4. Review results

Inspect these files under `jersey_ocr_local_results/`:

- `track_jersey_number_predictions.json`: conservative temporal votes per track
- `crop_ocr_predictions.json`: raw text, confidence, preprocessing attempts, and
  candidate digits for every crop
- `jersey_ocr_summary.json`: engine status and aggregate counts
- `jersey_ocr_contact_sheet.png`: crop-level OCR review
- `number_region_contact_sheet.png`: packaged number regions
- `per_track_evidence_contact_sheet.png`: strongest evidence grouped by track

A one-digit result remains uncertain because it may be a partial reading of a
two-digit jersey. No player name or roster identity is assigned by this command.

## 5. Copy results back to ECE

From the package directory on the Mac:

```bash
scp -r jersey_ocr_local_results \
  zllenza@ece020.ece.cmu.edu:/afs/ece.cmu.edu/usr/zllenza/research/prototype4/prototype4_pipeline/outputs/nll_test4/
```

Use a new destination name if `jersey_ocr_local_results/` already exists on ECE;
do not replace prior OCR or tracking artifacts.
