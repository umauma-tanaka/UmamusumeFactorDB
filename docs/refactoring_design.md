# リファクタリング設計書

## 目的

画像取得、画像結合、因子領域検出、ONNX 推論、OCR、テンプレート照合、候補統合、結果整形が混在している現在の実装を、段階的に小さな責務へ分割する。

最初の目標は性能改善そのものではなく、性能改善を安全に実施できる構造にすること。初期フェーズでは原則として認識結果を変えず、移動・抽出・テスト整備を優先する。

## 確認済み要件

- 結合機能、画像取得機能、認識機能を適切な Class とファイルへ分割する。
- コアの画像処理部分に集中できるファイル構成にする。
- リグレッションテストとデバッグ出力を整備する。
- 性能評価を定量的に行えるようにする。
- フォルダ構成とファイル構成を拡張しやすく、保守しやすくする。
- `umacapture` の設計を中核の参照元にする。

## 前提

- 既存の公開入口である `analyze_image(image_path, submitter_id, ...)` は維持する。
- 初期リファクタリングでは、既存の認識ロジックの意味を変えない。
- `umacapture` の scraper / stitcher / recognizer の分離方針を参考にする。
- 現行の `cropper.py` と `pipeline.py` は、既存挙動を保つ互換レイヤーとして一時的に残す。

## 未確定事項

- 入力画像は、結合済みの因子画像だけを扱うのか、生のスクロール連続画像も扱うのか。
- `config/scene_stitcher.json` はローカル値と `umacapture` 上流値のどちらを正とするか。
- `star_classifier` はローカル独自拡張として維持するのか、上流寄せの rank model 中心へ戻すのか。
- ONNX モデルを zip から自動展開するか、明示配置を前提にするか。

## ブロッカー

- 現状では `tests/fixtures/colored_factors/recognition_results.json` が存在せず、既存の pytest がそのままでは回らない。
- `pytest` がローカル環境に未導入だったため、現時点で回帰テストを実行できていない。
- `models/modules/star_classifier/prediction.onnx` が見当たらないため、星分類経路は実行時に失敗する可能性がある。

## 現状の問題

### `pipeline.py`

画像読込、正規化、セクション検出、候補生成、ONNX 推論、OCR、テンプレート照合、候補マージ、スロット確定、レビュー項目生成、Submission 生成が 1 ファイルに集中している。

この状態では、因子名の改善、星数判定の改善、結合精度の改善、レビュー UI 用データ生成の変更が相互に影響しやすい。

### `cropper.py`

画像正規化、ウマ娘セクション検出、因子行検出、星検出、星クラスタリング、FactorBox 生成、legacy fallback が同居している。

`umacapture` 由来の背景色スキャン型の考え方と、現在の HSV / star 主導の検出が混在しており、どちらが主経路なのか読み取りにくい。

### `infer.py`

ONNX Runtime の汎用ラッパー、factor 用 softmax 出力追加、カテゴリ制限推論、perturbation、star classifier が同居している。

モデル仕様の検証、モデルロード、推論戦略、特殊モデル対応を分ける必要がある。

### `templates.py`

赤青因子、星、緑因子名のテンプレートロードと照合処理が重複している。

テンプレート照合は認識の主経路ではなく、補助シグナルとして扱うべきである。

### `scripts/`

診断スクリプトが多数存在するが、共通のデバッグ出力形式、メトリクス形式、評価コマンドが統一されていない。

## 目標アーキテクチャ

```
src/umafactor/
  app/
    analyzer.py              # 既存 analyze_image の内部実装。全体 orchestration
    result_builder.py        # 認識結果 -> Submission / ReviewQueue 変換

  core/
    geometry.py              # Rect, Size, Point, 座標変換
    anchors.py               # umacapture 互換の anchor / rect 解決
    image.py                 # ImageFrame, normalize, crop utility
    debug.py                 # DebugSink, DebugArtifact, debug manifest
    config_loader.py         # JSON config loader と dataclass 化

  capture/
    source.py                # FileImageSource, UploadedImageSource
    scroll_estimator.py      # scroll bar + AKAZE/RANSAC offset estimator
    stitcher.py              # ScrollAreaStitcher / post-hoc stitcher
    scraper_types.py         # fragment, offset, stitch metadata

  detection/
    sections.py              # CharaSectionDetector
    rows.py                  # FactorRowDetector
    boxes.py                 # FactorBoxExtractor
    colors.py                # factor color 判定
    stars.py                 # star slot detection / classification

  recognition/
    onnx_runtime.py          # ONNX Runtime wrapper
    model_registry.py        # model path, labels, required model validation
    character.py             # CharacterRecognizer
    factor_name.py           # FactorNameRecognizer
    factor_rank.py           # FactorRankRecognizer
    ocr.py                   # OCR adapter。既存 ocr.py の整理先
    template_matcher.py      # 汎用 TemplateMatcher
    candidate_fusion.py      # ONNX / OCR / template の統合

  evaluation/
    dataset.py               # fixture / expected 読み込み
    metrics.py               # accuracy, runtime, stitch metrics
    runner.py                # regression / benchmark runner
    report.py                # JSON / Markdown 出力

  schema.py                  # 外部データ構造。既存 API 維持
  sheet_writer.py            # Sheets 連携
  review.py
  review_ui.py
```

## 主要 Class 設計

### `UmaFactorAnalyzer`

責務: 1 枚または複数枚の入力画像から `Submission` と `ReviewQueue` を生成する。

依存先:

- `ImageSource`
- `ImagePreprocessor`
- `CharaSectionDetector`
- `FactorBoxExtractor`
- `FactorRecognizer`
- `SubmissionBuilder`
- `DebugSink`

既存の `analyze_image()` は、このクラスを呼ぶ薄い互換関数にする。

### `ImageSource`

責務: ファイル、アップロード、将来の画面キャプチャなど、画像取得手段を抽象化する。

初期実装:

- `FileImageSource`
- `SingleImageInput`

将来実装:

- `ScrollFrameSequenceInput`
- `CaptureDeviceImageSource`

### `ScrollAreaStitcher`

責務: 複数のスクロール断片から縦長の因子タブ画像を生成する。

`umacapture` の方針に合わせ、理想的には「スクロール中に新規表示領域だけ保存し、最後に縦結合」する。

既存入力が結合済み画像だけの場合は、このクラスは通さず、後段の検出から開始する。

### `ScrollOffsetEstimator`

責務: 隣接フレーム間のスクロール量を推定する。

推定シグナル:

- スクロールバー位置
- AKAZE 特徴点
- FLANN matching
- RANSAC による平行移動の検証

出力:

- `offset_y`
- `confidence`
- `inlier_count`
- `reject_reason`

### `CharaSectionDetector`

責務: 本人、親1、親2の 3 セクションを検出する。

初期実装では現在の低彩度 run / star fallback を維持する。次フェーズで `umacapture` の anchor と背景色スキャンに寄せる。

### `FactorBoxExtractor`

責務: セクション内の因子スロット矩形を検出する。

内部戦略:

- `BackgroundScanRowDetector`
- `StarBasedRowDetector`
- `LegacyRowDetector`

初期フェーズでは現在の挙動を維持しつつ、戦略クラスとして分離する。

### `StarCounter`

責務: 星スロット画像から星数を決定する。

入力:

- star slot crop
- HSV 検出結果
- star classifier 結果
- rank model fallback 結果

出力:

- `star`
- `source`
- `confidence`
- `debug_evidence`

### `FactorRecognizer`

責務: 1 つの `FactorBox` から因子名・星数・候補一覧を返す。

内部依存:

- `FactorNameRecognizer`
- `FactorRankRecognizer`
- `OCRRecognizer`
- `TemplateMatcher`
- `CandidateFusion`
- `StarCounter`

### `CandidateFusion`

責務: ONNX、OCR、テンプレート照合の候補を統合する。

初期フェーズでは既存の `_merge_candidates` と `_merge_candidates_v2` を移動するだけにする。スコア式の変更は別 PR に分ける。

## データ構造

### `RecognitionCase`

```
case_id: str
input_images: list[Path]
expected: ExpectedSubmission
tags: list[str]
```

### `DebugManifest`

```
run_id: str
case_id: str
input_files: list[str]
pipeline_version: str
model_versions: dict
stages: list[StageDebug]
metrics: dict
```

### `FactorBoxDebug`

```
uma_index: int
row_index: int
col_index: int
bbox: [x0, y0, x1, y1]
rank_bbox: [x0, y0, x1, y1] | null
detected_color: str
star_source: str
gold_star_count: int | null
empty_star_count: int | null
candidates: list
selected_name: str
selected_star: int
```

## デバッグ出力設計

出力先:

```
outputs/debug/<run_id>/<case_id>/
  manifest.json
  input_normalized.png
  sections_overlay.png
  boxes_overlay.png
  stitch/
    offsets.json
    stitched.png
    seams_overlay.png
  crops/
    uma0_row00_col0_text.png
    uma0_row00_col0_rank.png
  candidates.json
  result.json
  metrics.json
```

原則:

- テスト fixture と debug output を混在させない。
- 自動生成物は `outputs/` 配下に出し、通常は Git 管理しない。
- 失敗時に、どの stage で失敗したかを JSON で追えるようにする。

## 評価指標

### 認識精度

- `character_accuracy`
- `blue_type_accuracy`
- `blue_star_accuracy`
- `red_type_accuracy`
- `red_star_accuracy`
- `green_name_accuracy`
- `green_star_accuracy`
- `white_name_accuracy`
- `white_star_accuracy`
- `slot_detection_recall`
- `slot_detection_precision`
- `error_case_rate`

### 画像結合精度

- `offset_error_px`
- `ransac_inlier_count`
- `stitch_confidence`
- `duplicate_band_px`
- `missing_band_px`
- `seam_discontinuity_score`
- `stitch_reject_rate`

### 性能

- `total_runtime_ms`
- `preprocess_ms`
- `stitch_ms`
- `section_detection_ms`
- `box_extraction_ms`
- `onnx_ms`
- `ocr_ms`
- `template_ms`
- `candidate_fusion_ms`
- `peak_memory_mb`

### 回帰判定

最低限の gating:

- 既存 fixture の認識結果が悪化しないこと。
- 認識不能件数が増えないこと。
- 平均処理時間が一定閾値を超えて悪化しないこと。

精度改善フェーズでは、色別・項目別の改善/悪化件数を必ず出す。

## テスト設計

### Unit test

- `geometry` の座標変換
- `anchors` の rect 解決
- `CandidateFusion` の順位決定
- `TemplateMatcher` のスコア計算
- `StarCounter` の fallback 優先順位
- `FactorBoxExtractor` の行検出

### Regression test

```
tests/fixtures/recognition_cases/
  sample_001/
    input.png
    expected.json
  sample_002/
    input.png
    expected.json
```

実行コマンド:

```
python scripts/run_regression.py --cases tests/fixtures/recognition_cases
python -m pytest tests/test_recognition.py
```

### Stitch test

```
tests/fixtures/stitch_cases/
  scroll_001/
    frames/
      000.png
      001.png
      002.png
    expected_stitched.png
    expected_offsets.json
```

評価項目:

- 推定 offset と期待 offset の差分
- 結合後画像の高さ
- seam 周辺の不連続スコア
- 重複/欠落 band の推定値

### Benchmark

```
python scripts/benchmark_pipeline.py --cases tests/fixtures/recognition_cases --repeat 3
```

出力:

```
outputs/evaluation/<timestamp>/
  metrics.json
  metrics.md
  failures.json
```

## 移行計画

### Phase 0: 現状固定

目的: リファクタリング前の基準値を作る。

作業:

- `recognition_results.json` を再生成できるコマンドを整備する。
- 現在の fixture で期待値 JSON を固定する。
- モデル存在確認コマンドを追加する。

完了条件:

- 1 コマンドで regression 実行ができる。
- 失敗時に不足ファイルか認識差分かが区別できる。

### Phase 1: 型とデバッグ基盤の追加

目的: 既存処理を移動する前に、共通データ構造を作る。

作業:

- `core/geometry.py`
- `core/debug.py`
- `evaluation/metrics.py`
- `recognition/candidate_fusion.py`

完了条件:

- 既存の `pipeline.py` から候補マージだけを移しても結果が変わらない。
- debug manifest が生成できる。

### Phase 2: `cropper.py` の分割

目的: 画像検出系を責務ごとに分ける。

作業:

- `detection/sections.py`
- `detection/stars.py`
- `detection/rows.py`
- `detection/boxes.py`
- `cropper.py` は互換 import のみ残す。

完了条件:

- `extract_factor_boxes()` の既存呼び出しが壊れない。
- 検出 overlay を debug output に出せる。

### Phase 3: `infer.py` の分割

目的: ONNX モデル管理と推論戦略を分ける。

作業:

- `recognition/model_registry.py`
- `recognition/onnx_runtime.py`
- `recognition/factor_rank.py`
- `recognition/character.py`
- `recognition/stars.py`

完了条件:

- 必須モデルの有無を起動時またはテスト時に検査できる。
- ONNX 入力 shape と出力名をログに出せる。

### Phase 4: `pipeline.py` の分割

目的: orchestration と認識ロジックを分離する。

作業:

- `app/analyzer.py`
- `app/result_builder.py`
- `recognition/factor_name.py`
- `recognition/template_matcher.py`
- `recognition/ocr.py`

完了条件:

- `analyze_image()` は薄い facade になる。
- `FactorRecognizer` 単体で候補一覧をテストできる。

### Phase 5: `umacapture` 寄せの画像結合を追加

目的: 画像結合を独立機能として扱えるようにする。

作業:

- `capture/scroll_estimator.py`
- `capture/stitcher.py`
- `capture/scraper_types.py`
- stitch regression fixture を追加する。

完了条件:

- 生スクロール画像がある場合は stitcher を通せる。
- 結合済み画像の場合は stitcher をスキップできる。
- offset / seam / duplicate の定量評価が出せる。

## 互換性方針

- `run.py`、`server/main.py`、`sheet_writer.py`、`schema.py` の外部インターフェースは維持する。
- `src/umafactor/cropper.py`、`src/umafactor/infer.py`、`src/umafactor/pipeline.py` は段階的に薄い wrapper へ移行する。
- 既存の診断スクリプトは、移行期間中は壊さない。新 API へ移した後、不要なものだけ廃止する。

## 実装時のルール

- 1 PR / 1 変更単位で進める。
- 挙動変更とファイル移動を同じ PR に混ぜない。
- アルゴリズム変更時は、必ず before / after のメトリクスを添える。
- 閾値変更は定数化し、評価結果とセットで記録する。
- fallback 経路は debug manifest に残す。

## 推奨する最初の変更

最初に行うべき変更は、候補統合と評価基盤の分離。

理由:

- `pipeline.py` の中でも副作用が少ない。
- 精度改善の影響を定量化する土台になる。
- ONNX、OCR、画像検出の大きな移動に入る前にテスト可能な単位を作れる。

最初の PR 候補:

1. `recognition/candidate_fusion.py` を追加する。
2. `_merge_candidates` と `_merge_candidates_v2` を移動する。
3. 既存 `pipeline.py` から import して使う。
4. `tests/test_candidate_fusion.py` を追加する。
5. 認識結果が変わらないことを regression で確認する。

## 参照

- `umacapture` scene scraper: https://github.com/umasagashi/umacapture/blob/develop/native/src/chara_detail/chara_detail_scene_scraper.h
- `umacapture` scene stitcher: https://github.com/umasagashi/umacapture/blob/develop/native/src/chara_detail/chara_detail_scene_stitcher.h
- `umacapture` recognizer: https://github.com/umasagashi/umacapture/blob/develop/native/src/chara_detail/chara_detail_recognizer.h
- `umacapture` ONNX wrapper: https://github.com/umasagashi/umacapture/blob/develop/native/src/cv/model.h
