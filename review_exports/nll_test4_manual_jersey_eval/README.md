# Manual Jersey Digit Evaluation Set

This folder contains selected enhanced jersey-number region crops from `nll_test4` for manual digit-label review.

Fill in `label_sheet.csv` or `label_sheet.json`:

- `manual_label`: enter the visible jersey number exactly as seen, for example `4`, `21`, or `42`.
- Use `unknown` when the number is not readable.
- Leave ambiguous cases as `unknown` and explain briefly in `notes` if useful.
- `manual_readable`: use `yes`, `no`, or `partial`.
- Do not enter player names.
- Do not assign player identities.

This dataset is only for evaluating digit recognition. The later identity layer should aggregate jersey-number evidence across a track and only map to roster/player identity when number/team evidence is strong.
