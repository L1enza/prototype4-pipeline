# Team Assignment V2

V2 improves the first team-level demo by moving from one representative RGB-like
track color to a track-level uniform signature built from multiple masked
upper-torso observations.

## Why Whole-Box Average RGB Is Insufficient

Full player bboxes include turf, boards, shadows, sticks, gloves, legs, and
nearby players. A single average RGB value can turn white uniforms gray, dark
uniforms green, and officials into player-like colors. It also cannot represent
secondary colors such as shoulders, trim, stripes, or shorts.

## Torso Isolation

For each selected detection, V2 uses the upper torso:

- x from 15% to 85% of bbox width
- y from 12% to 62% of bbox height

When an upstream mask is available, it is cropped into the torso region to remove
background pixels. The script rejects extreme highlights, deep shadows, and turf
or background pixels while preserving enough uniform material for secondary
colors.

## HSV And Lab Features

Each observation records brightness-robust color evidence:

- normalized HSV hue/saturation histogram
- normalized Lab histogram
- median HSV and Lab colors
- top three dominant Lab colors and fractions
- white, dark, and neutral pixel fractions
- color entropy
- valid-pixel coverage

The track signature uses robust aggregation over many observations rather than a
single crop.

## Appearance Embeddings

V2 has a modular appearance backend:

- `auto`
- `torchvision_resnet18`
- `none`

If cached ImageNet ResNet-18 weights are available, the script removes the
classifier head and uses the L2-normalized penultimate feature vector. It never
downloads weights unless `--allow-download-weights` is explicitly passed. If
weights are unavailable, V2 continues in color-only mode and records that in
`feature_backend_metadata.json`.

No model is trained or fine-tuned.

## Track-Level Robust Aggregation

For each track, V2 selects diverse high-quality observations using bbox size,
detection confidence, crop quality, valid torso coverage, and frame spacing.
Observation outliers are rejected using color histogram distance and, when
available, appearance embedding distance. Remaining features are pooled with a
robust median/trimmed mean.

The output is one stable track-level signature with:

- observations selected
- observations rejected
- color consistency
- appearance consistency
- evidence quality
- evidence mode

## Probabilistic Clustering

Non-official tracks are clustered into two team groups with a small diagonal
Gaussian mixture implementation. If scikit-learn is installed later, this can be
swapped to `GaussianMixture` without changing the output contract.

For every track, V2 records:

- automatic cluster
- posterior probability
- distances to team centers
- assignment margin
- final class
- unknown reason when withheld

Tracks become `unknown` when posterior probability, margin, or evidence quality
is too weak.

## Official And Unknown Handling

Officials are separated before team clustering using neutral/dark/white color
fractions, stripe-like edge density, and existing V1 official evidence. Unknown
tracks are preserved in metadata and excluded from team polygons. V2 is
conservative: uncertain evidence stays `unknown` instead of being forced into a
team.

## Scorebug Priors

`configs/nll_test4_team_color_priors.json` can later provide manually confirmed
team colors. Priors may map cluster labels to named teams and slightly influence
confidence, but they must not override strong torso evidence. V2 does not do
scorebug OCR.

## V1 Versus V2 Audit

The V2 output includes `v1_v2_comparison.json` and CSV. These report assignment
changes, class counts, projected observation counts by team, newly unknown
tracks, newly official tracks, and whether the original 12-versus-4 imbalance
looks like real track fragmentation or a classification error.

## Andrew Coordinates Later

The team classifier is independent of the field-coordinate source. The polygon
renderer currently consumes manual-homography `projected_player_points.json`, but
it can later consume Andrew `floor_xy_ft` coordinates by converting them into the
same field-template point schema:

```json
{
  "frame_index": 0,
  "track_id": 12,
  "projected_field_point": {"x": 100.0, "y": 200.0},
  "inside_field_template_bounds": true
}
```

That keeps team movement analytics separate from calibration strategy.
