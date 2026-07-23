# NJ Devils Hockey Computer Vision Demo

This project applies the Prototype 4 sports computer vision pipeline to a short NHL broadcast clip featuring the New Jersey Devils and Philadelphia Flyers.

## Purpose

The demo converts ordinary broadcast hockey footage into structured spatial analytics. It detects and tracks players, classifies teams and officials, registers the moving broadcast camera to a canonical top-down rink, projects player locations onto the rink, and generates synchronized broadcast and top-down visualizations.

## Main features

- Player detection and multi-frame tracking
- Stabilized bounding boxes through short detection gaps
- Track-level team classification
- New Jersey Devils identification using red torso evidence
- Philadelphia Flyers identification using white torso evidence
- Official detection using black-and-white stripe evidence
- Dynamic broadcast-to-rink homography registration
- Visible broadcast-region polygon on the canonical rink
- Team-position polygons
- Team heatmaps
- Track-review interface for manual label correction
- Broadcast, rink, and side-by-side demonstration videos

## Project structure

- `scripts/`
  - Hockey registration, annotation, tracking, classification, review, and rendering scripts
- `tests/`
  - Direct tests for registration, annotation, V2 processing, and label review
- `assets/`
  - Canonical rink image
- `configs/`
  - Canonical rink landmarks and manually labeled broadcast correspondences
- `sample_outputs/`
  - Representative metrics, heatmaps, contact sheets, and diagnostic images

## Pipeline

1. Detect players in every broadcast frame.
2. Associate detections into tracks.
3. Smooth track boxes and bridge short missed-detection gaps.
4. Analyze upper-torso crops for team and official evidence.
5. Register broadcast frames to the canonical rink.
6. Project fresh player observations onto rink coordinates.
7. Generate broadcast and top-down team polygons.
8. Generate Devils and Flyers heatmaps.
9. Review track-level labels in the HTML reviewer.
10. Render final corrected outputs.

## Team definitions

- Devils: predominantly red jersey or torso with black pants
- Flyers: predominantly white jersey or torso with black pants and possible orange accents
- Officials: black-and-white striped torso
- Black pants are not used as primary team evidence because both teams wear them

## V2 results

- Source frames: 277
- Registration accepted: 277/277
- Legacy raw track fragments: 165
- Enhanced fragments before stitching: 148
- Fragments after stitching: 146
- Accepted tracks: 60
- Short detection-gap events bridged: 222
- Center-jitter reduction: 71.21%
- Stabilized track-label transitions: 0
- SAM3 segmentation coverage: 0/277 because CUDA was unavailable on the execution machine

## Segmentation limitation

The SAM3 repository and checkpoint were present, but PyTorch could not access a CUDA device and `nvidia-smi` could not communicate with an NVIDIA driver. No artificial masks were represented as SAM3 output.

## Generated data

Large generated videos, raw detections, crops, and intermediate output directories are intentionally excluded from normal Git tracking. They can be reproduced using the scripts and original source video.

The source broadcast video is not included in this repository.
