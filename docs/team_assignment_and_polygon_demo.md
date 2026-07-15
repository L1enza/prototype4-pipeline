# Team Assignment And Polygon Demo

This smoke stage adds team-level analytics on top of existing tracked and
field-projected player positions. It does not identify players, read jersey
numbers, or assign roster names.

## Inputs

The demo consumes existing outputs from the calibrated `nll_test4` 10-second
segment:

- stabilized tracking metadata with bboxes and stable track IDs
- projected field points from the homography stage
- the original clean MP4, reopened only for color sampling
- the top-down NLL field template

It does not rerun SAM 3, VGGT, OCR, or any training.

## Team Color Estimation

For each track, the script samples multiple high-quality detections across the
track. It reopens the original video frame and extracts a torso-focused crop from
the tracking bbox, avoiding rendered overlays and labels.

Each observation records:

- median RGB
- median HSV
- median Lab
- normalized HSV histograms
- basic sharpness and crop quality
- neutral/white/dark color fractions used by the official heuristic

Track-level color evidence is a quality-weighted aggregate across observations,
not a single-frame decision.

## Clustering And Confidence

The script first marks only strong referee-like tracks as `official`, using a
conservative black/white/gray heuristic. Remaining tracks with enough color
evidence are clustered into two uniform groups with a small k-means pass over
Lab, HSV, and histogram features.

Cluster labels are intentionally named `team_a` and `team_b`. They are not mapped
to real team names yet.

Confidence combines:

- number and quality of observations
- distance to the assigned cluster center
- margin to the other cluster

Low-confidence tracks are assigned `unknown` instead of being forced into a team.

## Scorebug Priors

The script accepts an optional scorebug-prior JSON path, but reliable automatic
scorebug parsing is not implemented in this smoke stage. Scorebug colors should
eventually help map `team_a` and `team_b` to named teams, but should not override
strong player-uniform evidence.

## Polygon Construction

For each rendered frame, projected foot points are grouped by assigned track
class:

- `team_a` and `team_b` are included in team dots, trails, centroids, heatmaps,
  and polygons.
- `official` and `unknown` are drawn distinctly in broadcast overlays but are
  excluded from team polygons.

When a team has at least three visible projected players, the field view draws a
convex hull polygon. With two players it draws a line. With one player it draws
only a point.

Per-frame metrics include visible team counts, polygon areas, team centroids,
centroid distance, unknown counts, and official counts.

## Andrew Floor Coordinates Later

The current implementation uses manual homography-projected image coordinates
from `projected_player_points.json`. The same rendering and metrics layer can
later consume Andrew `floor_xy_ft` or any calibrated field-coordinate adapter by
converting those coordinates into the field-template pixel space, then emitting
the same per-frame point schema:

```json
{
  "frame_index": 0,
  "track_id": 12,
  "projected_field_point": {"x": 100.0, "y": 200.0},
  "inside_field_template_bounds": true
}
```

That keeps team polygons and heatmaps independent from the coordinate source.

## Known Limitations

- white uniforms and officials can look similar
- goalies may have different colors or padding
- lighting, shadows, and broadcast compression affect color features
- similar uniform colors may cluster poorly
- tracking ID switches affect team trails and polygons
- a single static homography can drift during camera motion
- no player names, jersey numbers, or roster matching are used here
