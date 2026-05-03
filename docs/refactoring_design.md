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

## 実装後アップデート（2026-05-04）

Phase 0 から Phase 3 までの実装を通して、当初計画から以下の変更が発生した。

- Phase 0 の基準は `tests/fixtures/colored_factors/recognition_results.json` を常時保持する方式ではなく、`scripts/run_phase0_regression.py --refresh --skip-ocr --compare-golden` で `outputs/phase0/<run_id>/recognition_results.json` を生成し、`tests/fixtures/colored_factors/phase0_golden_skip_ocr.json` と比較する方式にした。
- OCR はフル回帰の主目的から外し、リファクタリング確認では `--skip-ocr` を標準にした。OCR 精度・速度改善は別フェーズで扱う。
- `models/modules/star_classifier/prediction.onnx` は提供済みになったため、星分類は ONNX 経路を標準とする。HSV fallback は ONNX 欠落時の明示 opt-in のみ維持する。
- Phase 1 は当初の `candidate_fusion.py` だけでなく、`slots.py`、`candidate_generation.py`、`star_rank.py`、`assignment.py`、`green_prepass.py`、`characters.py`、`image_crops.py`、`factor_recognition.py`、`context.py`、`results.py` まで小分けで抽出済み。
- Phase 2 は `detection/sections.py`、`detection/stars.py`、`detection/rows.py`、`detection/boxes.py`、`detection/types.py`、`detection/constants.py` へ分割済み。`cropper.py` は互換 facade として残す方針に変更なし。
- Phase 3 は `recognition/model_registry.py`、`recognition/onnx_runtime.py`、`recognition/stars.py` を追加し、`infer.py` を互換 facade 化した。計画上の `recognition/factor_rank.py` と `recognition/character.py` は新設せず、既に抽出済みの `recognition/star_rank.py` と `recognition/characters.py` を正式な分割先として扱う。
- `scripts/check_test_env.py --include-model-io` で ONNX 入力 shape と出力名を確認できるようにした。
- Phase 4 では、`pipeline.py` に残っていた orchestration と結果構築を `app/` へ分離した。`pipeline.py` は互換 facade として残す。
- Phase 5 では、生スクロール連続画像 fixture を後回しにする前提で、`capture/` の型、結合済み画像 skip、metadata offset による stitcher、stitch 評価 helper、stitch case manifest loader を先に実装した。

## 未確定事項

- 入力画像は、結合済みの因子画像だけを扱うのか、生のスクロール連続画像も扱うのか。
- `config/scene_stitcher.json` はローカル値と `umacapture` 上流値のどちらを正とするか。
- `star_classifier` は当面ローカル独自拡張として維持する。将来 `umacapture` 寄せにする場合は、rank model fallback との役割分担を別途再設計する。
- ONNX モデルを zip から自動展開するか、明示配置を前提にするか。
- Phase 5 の画像結合評価に使う、生スクロール連続画像 fixture をどこから取得するか。

## ブロッカー

- Phase 5 の画像ベース offset 推定と実画像 stitcher 評価に進む上での主なブロッカーは、生スクロール連続画像 fixture が未整備であること。
- `tests/fixtures/colored_factors/recognition_results.json` は常設しない運用にしたため、`tests/test_recognition.py` 単体実行時は `UMAFACTOR_RECOGNITION_RESULTS` を指定するか、先に Phase 0 回帰を refresh する必要がある。
- Phase 5 の画像結合評価には、生スクロール連続画像 fixture が必要。現状の 56 枚 full fixture だけでは stitcher の定量評価は不足する。

## 現状の問題

### `pipeline.py`

Phase 4 で `app/analyzer.py` と `app/result_builder.py` へ分割済み。

`pipeline.py` は `analyze_image()`、review 反映、既存診断スクリプト向け crop helper alias を公開する互換 facade として残す。

### `cropper.py`

Phase 2 で `detection/` 配下へ分割済み。

`cropper.py` は互換 facade として残す。今後の改善対象は `detection/sections.py`、`detection/rows.py`、`detection/stars.py`、`detection/boxes.py` 側で扱う。

### `infer.py`

Phase 3 で `recognition/onnx_runtime.py`、`recognition/model_registry.py`、`recognition/stars.py` へ分割済み。

`infer.py` は互換 facade として残す。必須モデル確認と ONNX I/O 表示は `model_registry.py` と `scripts/check_test_env.py --include-model-io` で扱う。

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
    characters.py            # CharacterRecognizer
    candidate_generation.py  # FactorNameRecognizer 相当の候補生成
    star_rank.py             # FactorRankRecognizer 相当の星ランク推論
    assignment.py            # 候補からスロット結果への割当
    factor_recognition.py    # FactorRecognizer 相当の認識実行
    context.py               # predictor / OCR / config の実行時依存
    ocr.py                   # OCR adapter。既存 ocr.py の整理先
    template_matcher.py      # 汎用 TemplateMatcher
    candidate_fusion.py      # ONNX / OCR / template の統合

  evaluation/
    dataset.py               # fixture / expected 読み込み
    metrics.py               # accuracy, runtime, stitch metrics
    stitch_dataset.py        # stitch case manifest loader
    stitch_metrics.py        # offset / seam / gap-overlap metrics
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

現行のリファクタリング回帰は、画像処理の正誤を評価するものではなく、リファクタリング前後で同じ結果が得られることを確認する golden 比較とする。

現行コマンド:

```
python scripts/run_phase0_regression.py --refresh --skip-ocr --compare-golden
```

golden:

```
tests/fixtures/colored_factors/phase0_golden_skip_ocr.json
```

単体の `tests/test_recognition.py` を使う場合は、`UMAFACTOR_RECOGNITION_RESULTS` で生成済み JSON を指定する。

将来のケース単位 regression fixture 案:

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

上記 `run_regression.py` はまだ未実装。Phase 4 以降で `evaluation/runner.py` を作る時に統合する。

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

現時点では専用 benchmark runner は未実装。Phase 0 回帰の report と `batch_recognize.log` で総処理時間を確認している。

将来コマンド案:

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

### Phase 0: 現状固定（完了）

目的: リファクタリング前の基準値を作る。

作業:

- `scripts/run_phase0_regression.py` で回帰レポートを生成する。
- `tests/fixtures/colored_factors/phase0_golden_skip_ocr.json` を golden として固定する。
- `scripts/check_test_env.py` で依存 package、fixture、モデル存在確認を行う。
- OCR スキップ時でも基本機能の回帰を確認できるようにする。

完了条件:

- 1 コマンドで regression 実行ができる。
- 失敗時に不足ファイルか認識差分かが区別できる。
- golden 比較で `golden diffs: 0` を確認できる。

### Phase 1: 型とデバッグ基盤の追加（完了）

目的: 既存処理を移動する前に、共通データ構造と認識ヘルパーを作る。

作業:

- `core/geometry.py`
- `core/debug.py`
- `evaluation/metrics.py`
- `recognition/candidate_fusion.py`
- `recognition/slots.py`
- `recognition/candidate_generation.py`
- `recognition/star_rank.py`
- `recognition/assignment.py`
- `recognition/green_prepass.py`
- `recognition/characters.py`
- `recognition/image_crops.py`
- `recognition/image_preprocessing.py`
- `recognition/context.py`
- `recognition/factor_recognition.py`
- `results.py`

完了条件:

- 既存の `pipeline.py` から候補生成、候補統合、slot 割当、character 認識、結果整形を移しても結果が変わらない。
- debug manifest が生成できる。
- 各抽出単位に unit test がある。

### Phase 2: `cropper.py` の分割（完了）

目的: 画像検出系を責務ごとに分ける。

作業:

- `detection/constants.py`
- `detection/types.py`
- `detection/sections.py`
- `detection/stars.py`
- `detection/rows.py`
- `detection/boxes.py`
- `cropper.py` は互換 import のみ残す。

完了条件:

- `extract_factor_boxes()` の既存呼び出しが壊れない。
- 検出 overlay を debug output に出せる。

### Phase 3: `infer.py` の分割（完了）

目的: ONNX モデル管理と推論戦略を分ける。

作業:

- `recognition/model_registry.py`
- `recognition/onnx_runtime.py`
- `recognition/stars.py`
- `infer.py` を互換 facade にする。
- `recognition/context.py` と `detection/boxes.py` の依存先を分割後モジュールへ向ける。

計画変更:

- `recognition/factor_rank.py` と `recognition/character.py` は新設しない。既に抽出済みの `recognition/star_rank.py` と `recognition/characters.py` を継続利用する。

完了条件:

- 必須モデルの有無を起動時またはテスト時に検査できる。
- ONNX 入力 shape と出力名をログに出せる。
- `infer.py` の既存 import surface が壊れない。

### Phase 4: `pipeline.py` の分割（完了）

目的: `pipeline.py` に残った orchestration、入出力処理、結果構築を分離する。

実装後の計画変更:

- 認識ロジックの大半は Phase 1 で `recognition/` 配下へ移動済みのため、Phase 4 では新しい認識アルゴリズムを作らない。
- `analyze_image()` の公開 API は維持し、内部実装だけを `app/analyzer.py` へ移す。
- `pipeline.py` はしばらく互換 facade として残す。

作業:

- `app/analyzer.py`
- `app/result_builder.py`
- `pipeline.py` から入力画像読込、正規化、認識実行、Submission 生成の orchestration を段階的に移す。
- `recognition/factor_recognition.py` の呼び出し単位を `UmaFactorAnalyzer` から扱える形に整える。
- `results.py` と `schema.py` の境界を整理し、Submission / ReviewQueue 生成を `app/result_builder.py` に寄せる。
- 既存 `ocr.py` は Phase 4 では移動しない。OCR adapter 化は PaddleOCR 等を検討する別フェーズに回す。
- `templates.py` の `template_matcher.py` 化は、Phase 4 の後半または Phase 6 とする。まずは `pipeline.py` の責務削減を優先する。

完了条件:

- `analyze_image()` は薄い facade になる。
- `UmaFactorAnalyzer` 単体で、画像 path から `Submission` と `ReviewQueue` を生成する流れをテストできる。
- Phase 0 golden 比較で `golden diffs: 0` を維持する。
- 既存の `tests/test_factor_recognition.py`、`tests/test_results.py`、`tests/test_context.py` が継続して通る。

実装結果:

- `src/umafactor/app/analyzer.py` に `UmaFactorAnalyzer` を追加した。
- `src/umafactor/app/result_builder.py` に Submission 構築と review 反映を移した。
- `src/umafactor/pipeline.py` と `src/umafactor/results.py` は互換 facade として残した。
- `tests/test_analyzer.py` を追加し、orchestration の順序と `pipeline.analyze_image()` facade をテストした。

### Phase 5: `umacapture` 寄せの画像結合を追加（非画像部分は完了）

目的: 画像結合を独立機能として扱えるようにする。

作業:

- `capture/scroll_estimator.py`
- `capture/stitcher.py`
- `capture/scraper_types.py`
- stitch regression fixture を追加する。
- 生スクロール連続画像 fixture を取得する。現行の 56 枚 full fixture は認識回帰用であり、stitcher 評価には不足する。

完了条件:

- 生スクロール画像がある場合は stitcher を通せる。
- 結合済み画像の場合は stitcher をスキップできる。
- offset / seam / duplicate の定量評価が出せる。

実装済み:

- `capture/scraper_types.py` に `ScrollFrame`、`FrameOffset`、`StitchPlacement`、`StitchResult` を追加した。
- `capture/scroll_estimator.py` に metadata offset を読む `MetadataOffsetEstimator` を追加した。
- `capture/stitcher.py` に結合済み単一画像の skip 経路と、提供済み offset による deterministic stitcher を追加した。
- `evaluation/stitch_metrics.py` に offset error、duplicate/missing band、seam discontinuity、size match の評価 helper を追加した。
- `evaluation/stitch_dataset.py` と `tests/fixtures/stitch_cases/README.md` で stitch case manifest 形式を定義した。

未実装:

- 生スクロール連続画像 fixture の追加。
- AKAZE / FLANN / RANSAC 等による画像ベース offset 推定。
- 実 fixture に対する stitch regression runner。

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

## 次に推奨する変更

次に行うべき変更は、生スクロール連続画像 fixture を追加し、画像ベース offset 推定の実装に入ること。

理由:

- Phase 0 から Phase 4 で、認識回帰、検出、推論、アプリ orchestration の分離は完了した。
- 型、manifest、skip 経路、提供済み offset stitcher、評価 helper は追加済み。
- 現行 fixture は結合済み full 画像中心で、AKAZE / RANSAC 等の offset 推定を定量評価できない。
- 生スクロール fixture が揃えば、実アルゴリズムの before / after 評価を同じ `stitch_metrics.py` で比較できる。

次の PR 候補:

1. `tests/fixtures/stitch_cases/<case_id>/frames/` に生スクロール連続画像を追加する。
2. `case.json` に期待 offset / stitched size / expected stitched image を記録する。
3. `capture/scroll_estimator.py` に画像ベース offset estimator を追加する。
4. `capture/stitcher.py` に重複帯の扱い、欠落検出、seam debug 出力を追加する。
5. `evaluation` 側に stitch regression runner を追加する。

## 参照

- `umacapture` scene scraper: https://github.com/umasagashi/umacapture/blob/develop/native/src/chara_detail/chara_detail_scene_scraper.h
- `umacapture` scene stitcher: https://github.com/umasagashi/umacapture/blob/develop/native/src/chara_detail/chara_detail_scene_stitcher.h
- `umacapture` recognizer: https://github.com/umasagashi/umacapture/blob/develop/native/src/chara_detail/chara_detail_recognizer.h
- `umacapture` ONNX wrapper: https://github.com/umasagashi/umacapture/blob/develop/native/src/cv/model.h
