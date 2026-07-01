# nll_test4 10-Second Review Export

This is a curated review export, not the full Prototype 4 `outputs/` directory.
It contains only the videos and JSON checkpoint files needed to review the same
10-second `nll_test4` segment.

## Videos

The included MP4s show the 10-second tracking, heatmap, and field-projection
demo. Included video files: `side_by_side_tracking_field.mp4`, `tracking_overlay.mp4`, `heatmap_by_frame.mp4`.

## Debug evidence

`debug_tracks.json` is the original consolidated track export.
`debug_tracks_with_local_ocr.json` adds the basic local Tesseract evidence, and
`debug_tracks_with_enhanced_ocr.json` adds the enhanced-region experiment while
preserving the basic evidence. Import summaries retain source hashes and counts.

## Current identity status

- Enhanced number regions were generated and tested locally with Tesseract 5.5.2.
- Enhanced OCR processed 384 regions and produced 2 readable crops.
- Accepted track-level jersey numbers: 0.
- No player names or resolved player identities were assigned.
- The current bottleneck is jersey digit recognition, not roster lookup.

## Geometry and scope

This demo uses a manually selected homography for the broadcast camera view.
Camera pan or zoom can make the static projection drift. This is a short review
segment, not full-game inference.

The Andrew 3D-world integration is planned and reviewed separately; it is not
part of this export.
