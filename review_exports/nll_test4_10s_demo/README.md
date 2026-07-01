# nll_test4 10-Second Review Export

This is a curated review export, not the full Prototype 4 `outputs/` directory.
It contains only the videos and JSON checkpoint files needed to review the same
10-second `nll_test4` segment.

## Videos

The included MP4s show the 10-second tracking, heatmap, and field-projection
demo. Included video files: `side_by_side_tracking_field.mp4`, `tracking_overlay.mp4`, `heatmap_by_frame.mp4`.

## Debug evidence

`debug_tracks.json` is the consolidated per-track evidence export for this exact
segment. `debug_tracks_summary.json` reports coverage and source availability.
The stage-specific source JSON files remain in the main project outputs and are
not duplicated here.

## Current identity status

- Tesseract 5.5.2 was run locally on a Mac, where the OCR engine was available.
- Local OCR processed 90 number-region crops and produced 1 readable crop.
- Accepted track-level jersey numbers: 0.
- Player names and resolved player identities were not assigned.
- Roster metadata remains available only for future constrained matching.

## Geometry and scope

This demo uses a manually selected homography for the broadcast camera view.
Camera pan or zoom can make the static projection drift. This is a short review
segment, not full-game inference.

The Andrew 3D-world integration is planned and reviewed separately; it is not
part of this export.
