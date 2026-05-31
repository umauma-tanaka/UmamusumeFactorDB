# Fork元への画像処理・OCR改善取り込み方針レポート

作成日: 2026-05-10

## 1. 結論

現在のForkは、画像処理・OCR改善部分をGoogle Sheets書き込みからかなり分離できている。
総合評価は **B: 小さなI/F整理で移植可能** と判断する。

ただし、既存の `analyze_image()` をいきなり置き換える形でFork元へ戻すと影響範囲が広くなる。
安全に戻す場合は、まず「結合済み因子画像またはスクロールフレーム列を入力し、因子一覧の中間データを返す独立モジュール」として追加し、既存の `Submission` / Google Sheets 書き込みへは薄い変換ラッパーで接続するのがよい。

## 2. 確認範囲

ローカルForkとFork元の主要ファイルを確認した。

### ローカルFork

- `run.py`
- `server/main.py`
- `src/umafactor/pipeline.py`
- `src/umafactor/app/analyzer.py`
- `src/umafactor/app/result_builder.py`
- `src/umafactor/schema.py`
- `src/umafactor/sheet_writer.py`
- `src/umafactor/recognition/image_preprocessing.py`
- `src/umafactor/recognition/context.py`
- `src/umafactor/recognition/factor_recognition.py`
- `src/umafactor/recognition/candidate_generation.py`
- `src/umafactor/recognition/candidate_fusion.py`
- `src/umafactor/recognition/assignment.py`
- `src/umafactor/recognition/star_rank.py`
- `src/umafactor/recognition/factor_list_ocr.py`
- `src/umafactor/recognition/paddle_ocr_adapter.py`
- `src/umafactor/detection/factor_list.py`
- `src/umafactor/detection/star_slots.py`
- `src/umafactor/capture/static_stitch.py`
- `src/umafactor/capture/window_capture.py`
- `src/umafactor/capture/control_window.py`
- `scripts/capture_factor_list.py`
- `scripts/evaluate_factor_ocr.py`

### Fork元

- `run.py`
- `src/umafactor/pipeline.py`
- `src/umafactor/cropper.py`
- `src/umafactor/ocr.py`
- `src/umafactor/sheet_writer.py`

## 3. 現在のコード構造の要約

既存互換の入口は維持されている。
`run.py` は `analyze_image()` を呼び、レビュー後に `append_submission()` でGoogle Sheetsへ書き込む。
`sheet_writer.py` は `Submission.to_rows()` の結果だけを送る構造であり、OCR処理には直接依存していない。

画像処理・OCRは大きく2系統ある。

### 3.1 既存互換系

```text
run.py
  -> src/umafactor/pipeline.py
  -> src/umafactor/app/analyzer.py
  -> src/umafactor/recognition/*
  -> src/umafactor/schema.py
  -> src/umafactor/sheet_writer.py
```

`pipeline.py` は互換ファサードとして機能している。
既存の外部呼び出し側から見ると、`analyze_image()` の入口を維持できている。

### 3.2 新しい因子一覧系

```text
scripts/capture_factor_list.py
  -> src/umafactor/capture/window_capture.py
  -> src/umafactor/capture/control_window.py
  -> src/umafactor/capture/static_stitch.py
  -> src/umafactor/detection/factor_list.py
  -> src/umafactor/detection/star_slots.py
  -> src/umafactor/recognition/factor_list_ocr.py
  -> src/umafactor/recognition/paddle_ocr_adapter.py
```

新しい因子一覧系は、スクロール画像結合、因子カード検出、星数検出、PaddleOCR実行を既存Sheets書き込みから独立した形で持っている。
ただし、現状は `FactorListTile` とCSV / overlay出力が中心で、既存 `Submission` への正式変換I/Fが弱い。

## 4. 責務分割の確認

| 責務 | 現状の主なファイル | 評価 |
|---|---|---|
| 画像読み込み | `recognition/image_preprocessing.py`, `capture/static_stitch.py`, `window_capture.py` | 分離済み。ただし path / ndarray のI/F整理余地あり |
| 画像結合 | `capture/static_stitch.py` | 分離済み。移植対象として扱いやすい |
| 因子カード領域検出 | `detection/factor_list.py`, 既存 `detection/*` | 分離済み |
| OCR対象ROI抽出 | `recognition/factor_list_ocr.py`, `recognition/image_crops.py` | 分離済み |
| PaddleOCR実行 | `recognition/paddle_ocr_adapter.py` | 分離済み。ただし共有インスタンス管理は追加推奨 |
| OCR結果正規化 | `ocr.py`, `paddle_ocr_adapter.py` | EasyOCR系とPaddle系に分散。共通化余地あり |
| 因子名マスタ照合 / fuzzy match | `ocr.py`, `candidate_generation.py` | 既存系は実装済み。新しい因子一覧系では未統合気味 |
| 星数検出 | `detection/star_slots.py`, `recognition/star_rank.py` | 新方式は独立性が高い |
| 結果データ構造への変換 | `app/result_builder.py`, `schema.py`, `assignment.py` | 既存系は良い。新方式からの変換アダプタが不足 |
| Google Sheets書き込み | `sheet_writer.py`, `server/main.py` | OCRから概ね分離済み |
| CLI / GUI / 実行エントリポイント | `run.py`, `server/main.py`, `scripts/capture_factor_list.py`, `control_window.py` | 実験CLIが大きくなっている |

## 5. 画像処理・OCR部分の依存関係

新しい画像処理・OCR改善部分は、Google Sheetsには強く依存していない。

```text
capture/static_stitch.py
  -> core/geometry.py
  -> opencv / numpy

detection/factor_list.py
  -> detection/sections.py
  -> detection/star_slots.py
  -> detection/types.py
  -> opencv / numpy

recognition/factor_list_ocr.py
  -> recognition/paddle_ocr_adapter.py
  -> detection/factor_list.FactorListTile
  -> opencv / numpy

recognition/paddle_ocr_adapter.py
  -> paddleocr
  -> project root cache path
```

Google Sheets連携は以下に閉じている。

```text
run.py / server/main.py
  -> schema.Submission
  -> sheet_writer.append_submission()
```

このため、改善部分だけを戻すことは可能。
ただし、新しい因子一覧系の出力を既存 `Submission` に変換する層が必要になる。

## 6. Fork元機能を維持したまま組み込めるか

組み込みは可能と判断する。

理由:

- `sheet_writer.py` と `schema.py` はOCR改善部分から独立している。
- `pipeline.py` は互換ファサードとして機能している。
- 新しい結合・カード検出・PaddleOCR・星検出は概ねモジュール化されている。
- `run.py` のデフォルト動作を維持したまま、新フローをオプションとして追加できる。

注意点:

- `scripts/capture_factor_list.py` と `scripts/evaluate_factor_ocr.py` に実験・評価・overlay処理が多く残っており、そのまま移植するには重い。
- `FactorListTile` から `Submission` への正式変換I/Fが未整備。
- PaddleOCRの初期化共有、設定外出し、exe化時のパス解決は追加整理が必要。

## 7. 組み込みやすさの評価

評価: **B: 小さなI/F整理で移植可能**

補足:

- 新しい画像処理・OCRモジュールを既存処理と並列に追加するならB。
- 既存 `analyze_image()` を完全置換する前提ならC寄り。
- Google Sheets書き込みまで含めて一括変更するのは避けるべき。

## 8. 推奨する外部I/F

画像処理・OCRモジュールは、Google Sheetsを知らない形にするのが安全。

### 8.1 入力

```python
image_path: Path | str | None
image: np.ndarray | None
frames: Sequence[np.ndarray] | None
options: FactorOcrOptions
```

用途に応じて以下を受け取れるようにする。

- 静的スクリーンショット画像パス
- 結合済み画像
- スクロール中に取得した画像配列
- スクロールバー補助推定ON/OFF
- PaddleOCR設定
- debug出力先

### 8.2 出力

```python
FactorOcrResult(
    stitched_image,
    factors=[
        RecognizedFactor(
            role="parent" | "ancestor1" | "ancestor2",
            order=int,
            row=int,
            col=int,
            raw_name=str,
            normalized_name=str | None,
            category=str | None,
            stars=int,
            bbox=tuple[int, int, int, int],
            ocr_confidence=float | None,
            match_confidence=float | None,
            needs_review=bool,
        )
    ],
    debug=FactorOcrDebug(...)
)
```

### 8.3 既存データ構造への変換

既存機能へ接続する場合は、別に薄い変換関数を用意する。

```python
to_submission(
    result: FactorOcrResult,
    submitter_id: str,
    image_path: str,
) -> Submission
```

この形にすると、Google Sheets仕様は `Submission` 以降に閉じ込められる。

## 9. 推奨するモジュール構成

最小追加なら以下の構成がよい。

```text
src/umafactor/factor_list_pipeline.py
  recognize_factor_list_image()
  recognize_factor_list_frames()
  to_submission()

src/umafactor/capture/static_stitch.py
  既存利用

src/umafactor/detection/factor_list.py
  既存利用

src/umafactor/detection/star_slots.py
  既存利用

src/umafactor/recognition/factor_list_ocr.py
  既存利用

src/umafactor/recognition/paddle_ocr_adapter.py
  既存利用

src/umafactor/debug/overlay.py
  現在 scripts 側にある overlay 処理を必要最小限だけ移動
```

`run.py` や `sheet_writer.py` は原則そのままにする。
必要なら `--ocr-mode factor-list` のようなオプションで新経路を選べるようにする。
デフォルト動作は変えない方が安全。

## 10. 最小変更での移植方針

### 10.1 基本方針

- 既存 `run.py` のデフォルト動作を変えない。
- 既存 `sheet_writer.py` と `schema.py` を変えない。
- 新しい画像処理・OCRは独立モジュールとして追加する。
- 既存 `Submission` への変換は薄いアダプタに閉じ込める。
- PaddleOCRはインスタンスを再利用する。
- debug出力は通常処理と分離する。

### 10.2 移植対象

優先度順の移植対象は以下。

- 必須: `src/umafactor/capture/static_stitch.py`
- 必須: `src/umafactor/detection/factor_list.py`
- 必須: `src/umafactor/detection/star_slots.py`
- 必須: `src/umafactor/recognition/factor_list_ocr.py`
- 必須: `src/umafactor/recognition/paddle_ocr_adapter.py`
- 必須: `src/umafactor/core/geometry.py`
- 必要に応じて: `src/umafactor/capture/window_capture.py`
- 必要に応じて: `src/umafactor/capture/control_window.py`
- 推奨追加: `src/umafactor/factor_list_pipeline.py`
- 推奨追加: `src/umafactor/debug/overlay.py`
- 依存更新: `requirements.txt`

## 11. 変更対象ファイル一覧

実装する場合の変更候補は以下。

### 11.1 追加または移植

- `src/umafactor/factor_list_pipeline.py`
- `src/umafactor/capture/static_stitch.py`
- `src/umafactor/detection/factor_list.py`
- `src/umafactor/detection/star_slots.py`
- `src/umafactor/recognition/factor_list_ocr.py`
- `src/umafactor/recognition/paddle_ocr_adapter.py`
- `src/umafactor/debug/overlay.py`

### 11.2 最小変更

- `run.py`
  - 新フローを明示指定した場合のみ呼ぶオプションを追加する。
- `requirements.txt`
  - PaddleOCR関連依存を追加する。

### 11.3 変更しない、または極力触らない

- `src/umafactor/schema.py`
- `src/umafactor/sheet_writer.py`
- `config/apps_script_webhook.json`
- Google Apps Script側の受信仕様
- 既存 `Submission.to_rows()` の列順
- 既存 `run.py` のデフォルト動作
- 既存 `server/main.py` のリクエスト/レスポンス仕様

## 12. 変更しない方がよい箇所

以下は既存互換の要なので、最初の移植では変更しない方がよい。

- `schema.Submission`
- `schema.UmaFactors`
- `schema.FactorEntry`
- `Submission.to_rows()`
- `sheet_writer.append_submission()`
- Apps Script webhook payload
- Google Sheetsの列定義
- `run.py` の通常実行フロー
- `server/main.py` のAPI仕様

変更する場合は、必ずラッパーで吸収し、既存I/Fを残す。

## 13. 現在のコード上の注意点

### 13.1 関数・スクリプトが大きい箇所

- `scripts/capture_factor_list.py`
  - キャプチャ、結合、OCR、CSV、overlayをまとめて持っている。
  - ライブラリI/Fとしては大きい。
- `scripts/evaluate_factor_ocr.py`
  - 評価、overlay、メトリクスが大きく、移植対象ではなく参考実装扱いがよい。
- `capture/static_stitch.py`
  - 分離はされているが、内部helperと閾値が多い。

### 13.2 暗黙状態・グローバル状態

- EasyOCR系は `@lru_cache` やグローバルreaderを使っており、性能面では妥当。
- PaddleOCRはスクリプト単位では再利用されているが、サーバーやGUIに組み込むなら明示的なシングルトンまたはファクトリが必要。
- `build_recognition_context()` がOCR実装のprivate属性に触れている箇所がある。

### 13.3 パス指定

- `PROJECT_ROOT` 基準のパスが多い。
- exe化時は `Path(__file__)` 基準が崩れる可能性がある。
- `paddleocr_cache` をプロジェクト直下に置く方針は良いが、配布時は初回ダウンロード、オフライン、モデル同梱の方針を決める必要がある。

### 13.4 debug出力

- debug画像出力は維持すべき。
- ただし通常処理とは `DebugSink` のようなI/Fで分ける方がよい。
- `prepare_factor_image()` は画像読み込み、正規化、検出、debug出力が密結合している。

### 13.5 設定値

- `StarSlotConfig` など一部はdataclass化されている。
- 一方で、結合、ROI、OCR、閾値類にはまだコード内定数が多い。
- 最初から全面設定ファイル化する必要はないが、ユーザー調整が必要な値はoptionsに寄せるべき。

## 14. 実装する場合の作業手順

1. 既存の `run.py --dry-run` とGoogle Sheets書き込み仕様を固定する回帰テストを用意する。
2. `FactorOcrResult` / `RecognizedFactor` / `FactorOcrOptions` の最小データ構造を追加する。
3. `factor_list_pipeline.py` を追加し、`Path | np.ndarray | frames` 入力から `FactorOcrResult` を返す。
4. `FactorOcrResult -> Submission` の変換ラッパーを追加する。
5. `PaddleFactorOCR` の生成をファクトリ化し、同一プロセス内で再利用する。
6. overlay出力を通常処理から分離し、任意の `debug_dir` 指定時だけ出す。
7. `run.py` にはデフォルト無変更で、オプション指定時のみ新フローを呼ぶ。
8. `server/main.py` は後回しにし、まずCLIで互換性を確認する。
9. OCRなし、PaddleOCRあり、debugあり、Sheets dry-run の4系統で確認する。
10. 友人側へ戻す差分は、モジュール追加と薄い入口追加に絞る。

## 15. 動作確認チェックリスト

### 15.1 既存互換

- `run.py --dry-run` のJSON構造が変わらない。
- `Submission.to_rows()` の列数・列順が変わらない。
- Google Sheetsへの送信payloadが変わらない。
- 既存のレビューUI導線が壊れない。
- 既存のEasyOCR / ONNX系フローを選ぶ場合、従来通り動作する。

### 15.2 新しい画像処理・OCR

- PaddleOCRモデルキャッシュがプロジェクト直下に作られる。
- PaddleOCRインスタンスがカードごとに再生成されない。
- 結合済み画像から parent / ancestor1 / ancestor2 が混ざらず抽出される。
- 緑カードの黄色アイコンが星数に影響しない。
- 青、赤、緑、白/灰色カードの星数が安定して取れる。
- OCR対象ROIがカード内の文字領域に一致している。
- 複数カード結合OCRで精度が劣化しない。

### 15.3 debug・運用

- overlayなし通常実行で不要なdebug画像が出ない。
- overlayあり実行で親・祖1・祖2すべてのbbox、OCR結果、星数が確認できる。
- OCR失敗時に対象カード、bbox、raw OCR結果が追跡できる。
- PaddleOCR未導入環境で、エラーメッセージが分かりやすい。
- exe化想定で、モデル、設定、キャッシュパスを相対解決できる。

## 16. 未検証項目

- 実際のGoogle Sheets送信互換性。
- PaddleOCR込みの実行時間。
- exe化時のパス解決。
- 友人側の最新ローカル未公開コードとの差分。
- 新フローから `Submission` へ変換した場合の完全な列互換。

## 17. 残リスクと推奨次アクション

最大のリスクは、新しい因子一覧OCRフローがまだ評価スクリプト寄りで、既存 `Submission` への正式I/Fが薄い点である。

次に進める場合は、実装前に以下を小さく定義するのがよい。

- `FactorOcrResult`
- `RecognizedFactor`
- `FactorOcrOptions`
- `to_submission()`

この4点を先に固定すれば、既存 `schema.py` と `sheet_writer.py` を変更せずに、画像処理・OCR改善部分だけを安全に接続できる。
