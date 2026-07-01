# Jersey OCR Experiment Summary

The `nll_test4` jersey OCR experiments use tracked, clean RGB crops and preserve
all predictions as evidence. None of these experiments assigns player names or
resolved roster identities.

| Experiment | OCR availability | Regions processed | Readable crops | Accepted track numbers |
| --- | --- | ---: | ---: | ---: |
| ECE baseline | Unavailable | 90 prepared | 0 evaluated | 0 |
| Local basic Tesseract 5.5.2 | Available | 90 | 1 | 0 |
| Local enhanced Tesseract 5.5.2 | Available | 384 | 2 | 0 |

Enhanced spatial crops, contrast adjustment, thresholding, enlargement,
sharpening, and inversion increased readable crop results from one to two. The
track-level outcome did not change: no jersey number had enough consistent,
independent-frame evidence to pass the conservative acceptance rules.

The current bottleneck is crop-level jersey digit recognition, not roster
lookup. Roster constraints should remain downstream and must not convert weak
or inconsistent OCR evidence into a player identity.

## Recommendation

Move away from generic Tesseract as the main jersey-number recognizer. Next:

1. Evaluate stronger OCR or digit-detection models on the saved number-region crops.
2. Build a small manually labeled jersey-digit evaluation set from the best crops.
3. Use temporal voting only after the crop-level digit recognizer improves.
