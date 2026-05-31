# 因子一覧OCR改善 追加レビュー材料

作成日: 2026-05-10

## 1. 見てほしい差分

差分patch:

- `outputs/review_factor_list_ocr/review_diff.patch`

patchには以下を含めている。

- `run.py`
- `src/umafactor/factor_list_pipeline.py`
- `src/umafactor/factor_ocr_types.py`
- `src/umafactor/debug/__init__.py`
- `src/umafactor/debug/overlay.py`
- `tests/test_factor_list_pipeline.py`
- `docs/fork_integration_report.md`

生成コマンド:

```powershell
$out = 'outputs\review_factor_list_ocr\review_diff.patch'
git diff -- run.py src\umafactor\recognition\paddle_ocr_adapter.py | Set-Content -Path $out -Encoding utf8
foreach ($path in @(
  'src\umafactor\factor_ocr_types.py',
  'src\umafactor\factor_list_pipeline.py',
  'src\umafactor\debug\__init__.py',
  'src\umafactor\debug\overlay.py',
  'tests\test_factor_list_pipeline.py',
  'docs\fork_integration_report.md'
)) {
  git diff --no-index -- NUL $path 2>$null | Add-Content -Path $out -Encoding utf8
}
```

## 2. 実画像dry-run結果

入力画像:

- `datasets/test_factor_01/expected_stitched.png`

実行コマンド:

```powershell
.\.venv-paddle\Scripts\python.exe run.py datasets\test_factor_01\expected_stitched.png --submitter review --dry-run --ocr-mode factor-list --debug-crops outputs\review_factor_list_ocr_after3
```

実行結果:

- exit code: `0`
- `factor_count`: `118`
- role別件数:
  - `parent`: `44`
  - `ancestor1`: `40`
  - `ancestor2`: `34`
- `needs_review`: `118`
- normalized name内の星文字混入: `0`
- raw name内の `★★★` 混入: `0`

生成artifact:

- `outputs/review_factor_list_ocr_after3/factor_ocr_result.json`
- `outputs/review_factor_list_ocr_after3/factor_ocr_overlay.png`
- `outputs/review_factor_list_ocr_after3/stitched.png`

補足:

- 通常の `python` は 3.14.4 で、`paddlepaddle` が未導入のため失敗した。
- `.venv-paddle` は Python 3.13.13 で、`paddle` / `paddleocr` が導入済みだったため実行できた。
- 修正後は `paddle_mode="recognition"`, `ocr_execution_mode="batch"`, `ocr_batch_size=12` をデフォルトにした。
- `max_side_limit=4000` のcanvas縮小警告は出なくなった。

## 3. 実画像OCR結果の観察メモ

今回の実画像dry-runは通ったが、OCR文字列にはまだ誤認識が残っている。

例:

- `ゆきあかり、おいかけて` 相当が `ゆき有uい`
- `桜花賞` は `樱花赏` / `桜花赏` 系の誤字を正規化で `桜花賞` へ補正できた。
- `ヴィクトリアマイル` は `ウィクトリアマイル` を正規化で `ヴィクトリアマイル` へ補正できた。
- `NHKマイルC` は検出できた。
- `巨步` は正規化で `巨歩` へ補正できた。
- `ancestor1` 側で出ていた `★★★皐月賞` のような `★★★` 混入は消えた。
- raw nameに単発の `☆` が混入した箇所はあるが、normalized nameでは除去されている。

つまり、接続設計は確認できたが、レビューAIの指摘どおり **fuzzy match未接続時の扱い** は本番前に詰める必要がある。

## 4. schema.pyの接続先定義

今回の実装では `schema.py` は変更していない。
`to_submission()` は既存の `Submission`, `UmaFactors`, `FactorEntry` に変換するだけ。

確認対象:

```python
@dataclass
class FactorEntry:
    color: str
    name: str
    star: int


@dataclass
class UmaFactors:
    character: str = ""
    blue_type: str = ""
    blue_star: int = 0
    red_type: str = ""
    red_star: int = 0
    green_name: str = ""
    green_star: int = 0
    skills: list[FactorEntry] = field(default_factory=list)


@dataclass
class Submission:
    submitter_id: str
    image_filename: str
    main: UmaFactors = field(default_factory=UmaFactors)
    parent1: UmaFactors = field(default_factory=UmaFactors)
    parent2: UmaFactors = field(default_factory=UmaFactors)
    submission_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    submitted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_rows(self) -> list[list[str]]:
        timestamp = self.submitted_at.isoformat()
        return [
            self._build_row("main", self.main, timestamp),
            self._build_row("parent1", self.parent1, timestamp),
            self._build_row("parent2", self.parent2, timestamp),
        ]
```

`_build_row()` は `blue_type`, `red_type`, `green_name`, `skills[0..59]` を既存列順で展開する。
今回の実装では `Submission.to_rows()` の列数・列順は変更していない。

## 5. to_submission() の接続仕様

実装場所:

- `src/umafactor/factor_list_pipeline.py`

role mapping:

```text
parent    -> submission.main
ancestor1 -> submission.parent1
ancestor2 -> submission.parent2
```

カテゴリ mapping:

```text
blue  -> UmaFactors.blue_type / blue_star
red   -> UmaFactors.red_type / red_star
green -> UmaFactors.green_name / green_star
white or その他 -> UmaFactors.skills に FactorEntry として追加
```

注意:

- 同一role内で2個目以降の `blue` / `red` / `green` が出た場合は `skills` 側に回す。
- `stars` は `0..3` に丸める。
- 因子名は現時点ではOCR rawの軽い正規化のみで、因子名マスタ / fuzzy match は未接続。

## 6. PaddleOCR遅延importについて

`run.py` は `factor_list_pipeline` をimportするが、`PaddleFactorOCR` 本体は `get_paddle_factor_ocr()` 内で遅延importする。

確認点:

- `python run.py --help` は通常環境でも成功。
- `--ocr-mode legacy` のデフォルト経路ではPaddleOCRを初期化しない。
- `--ocr-mode factor-list` を指定した場合だけPaddleOCRが必要。

通常環境で `--ocr-mode factor-list` を使うと、現在は以下のように明示的に失敗する。

```text
RuntimeError: PaddleOCR requires the 'paddlepaddle' package, but it is not installed.
Current Python is 3.14.4; if pip cannot find a paddlepaddle wheel, create a Python 3.13 or 3.12 virtual environment for OCR.
```

## 7. テスト結果

実行コマンド:

```powershell
python -m pytest tests\test_factor_list_pipeline.py tests\test_results.py tests\test_analyzer.py tests\test_pipeline_facade.py tests\test_factor_list_ocr_preprocess.py tests\test_star_slots.py tests\test_paddle_ocr_adapter.py -q
```

結果:

```text
36 passed
```

確認していること:

- `to_submission()` が既存 `Submission` に変換できる。
- `Submission.to_rows()` の行数・列数が既存 `COLUMNS` と一致する。
- `debug_dir` なしではdebug画像を出さない。
- `debug_dir + enable_overlay` でoverlayを出す。
- デフォルトOCRファクトリがカードごとではなく1回だけ呼ばれる。
- 既存 `pipeline` facade周辺テストが通る。

## 8. 現時点の判断

設計方針と差分の切り方は、既存機能を壊さない方向に収まっている。

本番相当で使う前に追加確認すべき点は以下。

1. PaddleOCRのcanvas縮小警告を避ける設定、またはrole内分割の調整。
2. OCR rawに混入する `★★★` や文字誤認識への後処理。
3. 因子名マスタ / fuzzy match 未接続時のレビューUI・DB登録方針。
4. Python 3.13/3.12 venv、またはexe同梱時のPaddleOCRモデル配置。
