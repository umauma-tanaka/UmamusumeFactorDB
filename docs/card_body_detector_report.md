# Card Body Detector Report

## 概要

現在の正式なカード検出は `card-body` 方式です。
今後は以下を正式名称として扱います。

- 実装: `src/umafactor/detection/card_body_detector.py`
- デバッグCLI: `scripts/debug_card_body_detector.py`
- テスト: `tests/test_card_body_detector.py`

この方式は、丸アイコン中心やHough変換を最終bbox決定に使いません。
カードbodyのマスク、左右2列の投影、行グリッド推定からカードbboxを生成します。

## 処理フロー

1. stitched画像を読み込む。
2. 画像サイズから因子一覧領域を推定する。
3. HSV/Labベースでカードbodyマスクを生成する。
4. 小ノイズを除去し、横長カードbodyのconnected componentを抽出する。
5. component群から左右2列のx範囲をmedianで推定する。
6. 列ごとにy方向projectionを作り、カードbody bandを検出する。
7. 左右列の近いband中心を同一rowとしてmergeする。
8. row/colごとにbody bboxを生成する。
9. body bboxへ安全マージンを足して、OCR/星検出用のitem bboxを生成する。
10. debug指定時はoverlay、mask、projection、crop、CSV/JSONを出力する。

## 依存範囲

`card-body` 検出器は画像処理のみを担当します。
以下には依存しません。

- OCRエンジン
- PaddleOCR / RapidOCR
- 星数最終判定
- Submission
- Google Sheets連携
- legacyフロー

## 評価fixture

手動bbox fixture:

```text
tests/fixtures/card_crop_reference/card_bbox_expected.json
```

手動cropは完全一致用ではなく、カードcrop品質の参照です。
評価ではIoUを補助指標とし、文字・星・左アイコン・カード下端が欠ける hard failure を重視します。

## 代表評価結果

過去のfixture評価では、3枚のstitched画像でhard failureは0でした。

| image | detected cards | median item_w | median item_h | mean IoU | min IoU | hard failure |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `stitched_1.png` | 44 | 421 | 92 | 0.8985 | 0.8128 | 0 |
| `stitched_2.png` | 40 | 421 | 92 | 0.9005 | 0.8128 | 0 |
| `stitched_3.png` | 34 | 421.5 | 92 | 0.8925 | 0.8230 | 0 |

## デバッグ出力

```powershell
python scripts\debug_card_body_detector.py tests\fixtures\card_crop_reference\sources\stitched_1.png --out outputs\card_body_stitched_1
```

主な出力:

- `card_body_mask.png`
- `card_body_mask_clean.png`
- `x_projection.png`
- `row_projection_left.png`
- `row_projection_right.png`
- `card_body_detection_overlay.png`
- `card_body_detection_result.json`
- `card_body_detection_debug.csv`
- `crops/*.png`
- `contact_sheet.png`

## 本体接続

本体のfactor-list OCRフローでは、`src/umafactor/detection/factor_list_cards.py` の
`detect_factor_list_cards()` からこの検出器を呼び出します。
OCRエンジンは検出器から独立しており、`rapidocr` / `paddle` を選択できます。
