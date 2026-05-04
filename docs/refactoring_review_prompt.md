# UmamusumeFactorDB リファクタリングレビュー用プロンプト

UmamusumeFactorDB のリファクタリング結果をレビューしてください。

## 対象

- リポジトリ: <https://github.com/umauma-tanaka/UmamusumeFactorDB>
- ブランチ: `master`
- 確認対象の最新 HEAD: `866920b`

## プロジェクト概要

このプロジェクトは、ウマ娘の因子画面画像を解析し、`Submission` と `ReviewQueue` を生成する画像認識アプリケーションです。

主な処理は以下です。

- 画像読込
- 画像正規化
- ウマ娘セクション検出
- 因子ボックス検出
- ONNX 推論
- OCR
- 候補統合
- 星数判定
- 結果整形

## リファクタリング目的

- 画像取得、画像結合、検出、ONNX 推論、OCR、候補統合、結果構築を責務ごとに分離する。
- `pipeline.py` や `infer.py` に集中していた処理を小さいモジュールへ移す。
- 画像処理アルゴリズム自体の精度改善ではなく、今後の改善を安全に行える構造にする。
- リファクタリング前後で同じ結果が得られることを golden regression で確認する。
- 性能評価やデバッグ出力を拡張しやすくする。

## 重要な制限事項

- 公開 API の `umafactor.pipeline.analyze_image(...)` は維持する。
- `src/umafactor/cropper.py`、`src/umafactor/infer.py`、`src/umafactor/pipeline.py`、`src/umafactor/results.py` は互換 facade として残してよい。
- 今回のレビューでは画像認識精度そのものの OK/NG は主目的ではない。リファクタリング前後で挙動が保たれているかを重視する。
- OCR は遅いため、基本回帰では `--skip-ocr` を使う。
- 生スクロール連続画像 fixture はまだ未整備。Phase 5 の画像ベース stitcher 精度評価は未完了として扱う。
- `models/modules/factor_tag.json`、`models/modules/license.md`、`models/modules/skill_tag.json` にローカル未コミット変更が残っている場合があるが、今回のリファクタリング成果とは別件として扱う。

## 実施済みフェーズ

### Phase 0: 回帰テスト基盤

- `scripts/run_phase0_regression.py`
- golden: `tests/fixtures/colored_factors/phase0_golden_skip_ocr.json`

### Phase 1: 認識 helper / debug / metrics 分割

- `recognition/candidate_fusion.py`
- `recognition/candidate_generation.py`
- `recognition/slots.py`
- `recognition/assignment.py`
- `recognition/green_prepass.py`
- `recognition/characters.py`
- `recognition/factor_recognition.py`
- `recognition/context.py`

### Phase 2: `cropper.py` 分割

- `detection/sections.py`
- `detection/rows.py`
- `detection/stars.py`
- `detection/boxes.py`
- `detection/types.py`
- `detection/constants.py`

### Phase 3: `infer.py` 分割

- `recognition/onnx_runtime.py`
- `recognition/model_registry.py`
- `recognition/stars.py`
- `infer.py` は互換 facade

### Phase 4: `pipeline.py` 分割

- `app/analyzer.py`
- `app/result_builder.py`
- `pipeline.py` は `UmaFactorAnalyzer` を呼ぶ薄い facade

### Phase 5: 生スクロール画像なしで対応できる範囲

- `capture/scraper_types.py`
- `capture/scroll_estimator.py`
- `capture/stitcher.py`
- `evaluation/stitch_metrics.py`
- `evaluation/stitch_dataset.py`
- `tests/fixtures/stitch_cases/README.md`

## 重点レビュー観点

### 1. 互換性

- `pipeline.analyze_image()` の引数・戻り値が維持されているか。
- 既存スクリプトが参照する `_display_crop_from_original` などの alias が壊れていないか。
- `infer.py`、`results.py`、`cropper.py` の facade が妥当か。

### 2. 責務分離

- `app/analyzer.py` は orchestration に集中しているか。
- `app/result_builder.py` は `Submission` 構築と review 反映に集中しているか。
- `recognition/`、`detection/`、`capture/`、`evaluation/` の境界が自然か。

### 3. 挙動維持

- golden regression で差分 0 を維持できるか。
- 画像処理ロジックや閾値が意図せず変わっていないか。

### 4. テスト

- 新規モジュールに対応する unit test があるか。
- monkeypatch による orchestration test が過剰に実装詳細へ依存していないか。
- stitcher は生画像なしで検証できる範囲に留まっているか。

### 5. Phase 5 の扱い

- `MetadataOffsetEstimator` と deterministic stitcher は、将来の AKAZE / RANSAC 実装前の基盤として妥当か。
- `stitch_metrics.py` の offset / duplicate / missing / seam 指標が今後の評価に使える形か。
- 生スクロール fixture 未整備の範囲を、実装済みと誤解していないか。

## 推奨確認コマンド

```powershell
python -m pytest tests\test_capture_stitcher.py tests\test_stitch_metrics.py tests\test_stitch_dataset.py tests\test_analyzer.py tests\test_check_test_env.py tests\test_model_registry.py tests\test_stars.py tests\test_infer_facade.py tests\test_context.py tests\test_detection_compat.py tests\test_geometry.py tests\test_debug.py tests\test_metrics.py tests\test_candidate_fusion.py tests\test_results.py tests\test_image_preprocessing.py tests\test_factor_recognition.py tests\test_candidate_generation.py tests\test_assignment.py tests\test_green_prepass.py tests\test_image_crops.py tests\test_characters.py tests\test_slots.py tests\test_star_rank.py -q -p no:cacheprovider
```

```powershell
python scripts\run_phase0_regression.py --refresh --skip-ocr --compare-golden
```

```powershell
python scripts\check_test_env.py --skip-ocr --include-model-io
```

## 期待される既知結果

- 関連 pytest は 127 件程度通過する想定。
- Phase 0 golden 回帰は `golden matched: True`、`golden diffs: 0` の想定。
- `image errors: 2` は golden に含まれる既存エラーであり、今回のリファクタリング差分ではない。

## レビュー出力形式

- 重大な不具合、互換性破壊、テスト不足を優先して列挙してください。
- ファイル名と該当箇所を明示してください。
- 問題がなければ「Phase 0-5 非画像部分のリファクタリングとして大きな問題なし」と明記してください。
- 残リスクとして、生スクロール fixture 未整備と OCR 有効回帰未実施を挙げてください。
