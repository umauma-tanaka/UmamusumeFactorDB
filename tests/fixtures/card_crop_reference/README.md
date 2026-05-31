# Card Crop Fixture

This fixture was generated from the three uploaded stitched images.

It contains 118 card crops and `card_bbox_expected.json` with source-image bbox coordinates.

## Important

The bboxes are practical reference crops, not pixel-perfect manual annotations.
Use them to evaluate whether automatic card detection crops a usable full card.
Do not use this fixture as OCR text/star ground truth; `expected_name` and `expected_stars` are intentionally `null`.

## Quality contract

A valid crop should include:

- full rounded card body
- left circular factor icon
- factor name text
- all three star slots
- card bottom border/shadow

A valid crop should not include:

- a large part of neighboring cards
- character face icon
- scrollbar
- unrelated UI elements

## Files

- `card_bbox_expected.json`: bbox list.
- `crops/`: generated card crop images.
- `sources/`: source stitched images.
- `*_bbox_overlay.png`: bbox overlay for each source image.
- `contact_sheet.png`: quick visual check of all crops.
