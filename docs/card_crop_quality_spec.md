# Card Crop Quality Spec

## 目的

この仕様書は、因子カードbbox検出の品質基準を定義します。
手動crop画像とピクセル単位で完全一致させることが目的ではありません。
OCR text ROI と star ROI に進める自然なカードcropを安定して得ることが目的です。

## 参照データ

```text
tests/fixtures/card_crop_reference/
```

主な内容:

- `card_bbox_expected.json`: source画像上の手動bbox座標。IoU評価に使用する。
- `sources/`: bbox座標の基準になるstitched画像。
- `crops/`: 見た目の参照crop。完全一致テストには使わない。
- `*_bbox_overlay.png`: source画像上に手動bboxを重ねた確認画像。
- `contact_sheet.png`: 参照crop一覧。

`expected_name` と `expected_stars` は、このfixtureではOCR/星数の正解ではなく、必要に応じた補助メタデータです。

## bbox JSON形式

```json
{
  "images": [
    {
      "id": "stitched_1",
      "path": "sources/stitched_1.png",
      "role_default": "parent"
    }
  ],
  "items": [
    {
      "id": "stitched_1_parent_r00_c0",
      "image_id": "stitched_1",
      "source_image": "sources/stitched_1.png",
      "role": "parent",
      "row": 0,
      "col": 0,
      "bbox": [179, 82, 600, 168],
      "crop_path": "crops/stitched_1_parent_r00_c0_blue.png",
      "tolerance_px": 8
    }
  ]
}
```

`bbox` はsource画像座標の `[x1, y1, x2, y2]` です。
`x2` / `y2` はexclusiveとして扱います。

## 合格すべきcrop

card cropには次の要素が自然に含まれている必要があります。

- カードの丸角を含むカード全体
- 左の丸アイコン
- 因子名テキスト全体
- 星3スロット全体
- カード下端の枠線または影
- OCR text ROI と star ROI を後段で安定して切り出せる余白

## 含めすぎてはいけない要素

次の混入はNGです。

- 上下左右の隣接カードが大きく入る
- 左側のキャラクター顔アイコンが大きく入る
- スクロールバーが主領域に入る
- 無関係なUI要素が入る
- cropの幅や高さが過大で、text ROI / star ROI の相対位置が不安定になる

## 共通条件

青、赤、緑、白/灰色カードで共通に満たす条件です。

- 左丸アイコンが欠けない
- 因子名の先頭・末尾が欠けない
- 星スロットが欠けない
- 下端の枠線や影が極端に欠けない
- bbox幅と高さが同一UIスケール内で大きくばらつかない
- 手動bboxより大きく内側へ入り込まない
- 手動bboxより極端に大きくならない

## Hard Failure

IoUは補助指標です。主判定ではhard failureを重視します。

NG条件:

- 自動bboxが手動bboxより明らかに内側に入り、左丸アイコン、文字、星、下端を欠けさせる可能性がある
- bbox高さが手動bbox高さに対して極端に小さい
- bbox幅が手動bbox幅に対して極端に小さい
- bboxが隣接カードを大きく含む
- card height / width が手動参照の範囲から大きく外れる
- aspect ratio が大きく外れる

現在の目安:

- `tolerance_px` を超えて自動bboxが手動bboxの内側に入ったらNG
- 幅または高さが手動bboxの88%未満ならNG
- 幅が手動bboxの125%超ならNG
- 高さが手動bboxの130%超ならNG
- 面積が手動bboxの160%超ならNG
- aspect ratio が `3.8` 未満または `6.5` 超ならNG
- IoU が `0.60` 未満ならNG

## 正式検出方式

正式方式は `card-body` です。

`card-body` は、カードbody外枠のマスクと投影から左右2列のx範囲、row pitch、body topを推定し、body外枠に安全マージンを足して最終card bboxを作る方式です。
丸アイコンのHough中心は最終bbox決定に使いません。

判定優先度:

1. hard failure数が少ない
2. mean IoU が高い
3. card_w / card_h のばらつきが小さい

カードcropは外枠ぴったりより、少し余白を含む方がOCRに有利です。
そのため、IoUだけではなく、文字・星・左丸アイコン・下端を欠けさせないことを優先します。
