# Card Crop References

This directory is for manually cropped card images used as visual quality references.

These images are not used for pixel-perfect tests.  They define the expected visual shape of a good crop:

- the rounded card body is naturally included
- the left round icon is not clipped
- the factor name text is not clipped
- all three star slots are visible
- the bottom border/shadow is not severely clipped
- adjacent cards and character portraits are not largely included

Quantitative IoU evaluation must use `tests/fixtures/card_bbox_expected.json`, because standalone crop images do not contain their coordinates in the original stitched image.
