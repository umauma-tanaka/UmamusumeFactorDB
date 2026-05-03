# Phase 0 テスト基盤

## 目的

現在の認識結果を基準値として固定し、以後のリファクタリングで出力差分を検出できる状態にする。

Phase 0 では、アルゴリズムの改善は行わない。環境チェック、認識結果 JSON の生成、ゴールデン比較、必要に応じた期待値との比較、評価レポート出力を整備する。

リファクタリング時の主判定は「画像処理として正しいか」ではなく「リファクタリング前と同じ結果が得られるか」とする。現時点の誤認識や検出失敗もゴールデンに含め、出力が一致すれば PASS とする。

## 追加した入口

### 環境チェック

```
python scripts/check_test_env.py
```

確認内容:

- Python package の import 可否
- 必須 config / fixture / labels の存在
- ローカル EasyOCR モデルの存在
- 必須 ONNX モデルの存在
- `models/modules.zip` 内に不足モデルが含まれるか
- 回帰テスト用画像数
- `recognition_results.json` の存在

JSON レポートも出力できる。

```
python scripts/check_test_env.py --json outputs/phase0/env_check.json
```

### Phase 0 回帰実行

既存の `recognition_results.json` を使って評価する。

```
python scripts/run_phase0_regression.py
```

認識結果を再生成してから評価する。

```
python scripts/run_phase0_regression.py --refresh
```

pytest を省略し、メトリクスだけ作成する。

```
python scripts/run_phase0_regression.py --skip-pytest
```

`star_classifier/prediction.onnx` が未提供の環境だけ、HSV による暫定 fallback で partial baseline を作成できる。

```
python scripts/run_phase0_regression.py --refresh --allow-missing-star-classifier
```

このモードは正式な baseline ではない。`summary.md` と `metrics.json` には `star_classifier_mode: hsv_fallback_partial` と記録される。

OCR やモデル配置の smoke 確認だけ行う場合は、少数画像に限定できる。

```
python scripts/run_phase0_regression.py --refresh --limit 1 --skip-pytest
```

`--limit` 指定時は、評価対象も生成された画像だけに絞られる。この場合の `recognition_results.json` は `outputs/phase0/<timestamp>/` 配下だけに出力し、正式な `tests/fixtures/colored_factors/recognition_results.json` は上書きしない。

OCR の動作確認は 1 枚だけで行い、フル件数では OCR を止めて基本機能だけを確認できる。

```
python scripts/run_phase0_regression.py --refresh --limit 1 --skip-pytest
python scripts/run_phase0_regression.py --refresh --skip-ocr --basic-only
```

`--skip-ocr` は EasyOCR の初期化と推論を行わず、OCR 候補を空として扱う。`--basic-only` は OCR 依存度が高い `green_name` を評価対象から外し、`character` / `blue_type` / `blue_star` / `red_type` / `red_star` / `green_star` を比較する。

### リファクタリング用ゴールデン比較

初回、または意図して現在挙動を更新する場合のみゴールデンを更新する。

```
python scripts/run_phase0_regression.py --refresh --skip-ocr --update-golden
```

通常のリファクタリング確認では、ゴールデンを更新せずに比較する。

```
python scripts/run_phase0_regression.py --refresh --skip-ocr --compare-golden
```

既定のゴールデン:

```
tests/fixtures/colored_factors/phase0_golden_skip_ocr.json
```

ゴールデン比較では、毎回変わる `submission_id` と `submitted_at` を除外して比較する。画像処理の誤認識や `error` は正誤判定せず、ゴールデンと同じであれば PASS とする。

ローカル実行時の EasyOCR モデルは、既定で `models/easyocr/` に保存する。Cloud Run などで `EASYOCR_MODULE_PATH` を指定している場合は既存の指定を優先し、`EASYOCR_DOWNLOAD_ENABLED=1` を明示した場合だけダウンロードを許可する。

出力先:

```
outputs/phase0/<timestamp>/
  env_check.json
  env_check.log
  batch_recognize.log        # --refresh 指定時
  recognition_results.json
  metrics.json
  failures.json
  pytest.log                 # --skip-pytest 未指定時
  summary.md
```

## 評価対象

リファクタリング用途では `phase0_golden_skip_ocr.json` との完全一致を主判定にする。

画像処理精度を確認したい場合は、`tests/fixtures/expected_labels.csv` を正解データとして以下を比較する。

- `character`
- `blue_type`
- `blue_star`
- `red_type`
- `red_star`
- `green_name`
- `green_star`

`--basic-only` 指定時は `green_name` を除外する。

白因子スロットは、現時点の `expected_labels.csv` に含まれていないため Phase 0 の比較対象外。

## pytest 連携

`tests/conftest.py` は、既定では以下を読む。

```
tests/fixtures/colored_factors/recognition_results.json
```

別の認識結果 JSON を使う場合は、環境変数で指定する。

```
$env:UMAFACTOR_RECOGNITION_RESULTS="outputs/phase0/20260503_120000/recognition_results.json"
python -m pytest tests/test_recognition.py -q
```

`scripts/run_phase0_regression.py` は、この環境変数を自動で設定して pytest を実行する。

## 現環境で想定される不足

環境によって、以下が不足している可能性がある。

- `pytest`
- `models/modules/factor/prediction.onnx`
- `models/modules/factor_rank/prediction.onnx`
- `models/modules/character/prediction.onnx`
- `tests/fixtures/colored_factors/recognition_results.json`

`factor`、`factor_rank`、`character` は `models/modules.zip` に含まれている場合がある。`star_classifier` は `models/modules/star_classifier/` 配下の `prediction.onnx` と `prediction.onnx.data` を使用する。

## テスト画像の扱い

Phase 0 の認識回帰テストには、既存の `tests/fixtures/*.png` を使う。現時点で追加画像の取得は不要。

ただし、画像結合機能のテストは別。`umacapture` 方式のスクロール結合を検証する Phase 5 では、以下のような生スクロール連続画像が必要になる。

```
tests/fixtures/stitch_cases/<case_id>/
  frames/
    000.png
    001.png
    002.png
  expected_offsets.json
  expected_stitched.png      # 任意。最初はなくてもよい
```

取得条件:

- 同じウマ娘詳細画面の因子タブを、少しずつ縦スクロールしながら連続撮影する。
- 画像サイズと端末アスペクト比は同一にする。
- 通知、通信表示、タップ跡、個人情報が写らないようにする。
- 1 ケースあたり 3 から 8 枚程度でよい。

## 推奨実行順

1. 環境チェック

```
python scripts/check_test_env.py
```

2. 不足パッケージの導入

```
python -m pip install -r requirements.txt
```

3. ONNX モデルの配置確認

```
python scripts/check_test_env.py
```

`factor`、`factor_rank`、`character` が `models/modules.zip` にだけ存在する場合は、以下で展開する。

```
tar -xf .\models\modules.zip -C .\models
```

`star_classifier/prediction.onnx` は `models/modules/star_classifier/prediction.onnx.data` を参照するため、両方を同じディレクトリに配置する必要がある。

ONNX が提供されるまで作業を進める場合は、以下で missing star classifier を WARN 扱いにできる。

```
python scripts/check_test_env.py --allow-missing-star-classifier
python scripts/run_phase0_regression.py --refresh --allow-missing-star-classifier
```

または環境変数でも指定できる。

```
$env:UMAFACTOR_ALLOW_MISSING_STAR_CLASSIFIER="1"
python scripts/run_phase0_regression.py --refresh
```

4. 認識結果の生成

```
python scripts/run_phase0_regression.py --refresh
```

5. 以後のリファクタリング後の比較

```
python scripts/run_phase0_regression.py --refresh --skip-ocr --compare-golden
```

## Phase 0 完了条件

- `scripts/check_test_env.py` が required failure なしで完了する。
- `scripts/run_phase0_regression.py --refresh --skip-ocr --compare-golden` が `golden matched: True` で完了する。
- `outputs/phase0/<timestamp>/summary.md` を生成する。
- 以後の変更で `metrics.json` と `golden_diff.json` を比較できる。
