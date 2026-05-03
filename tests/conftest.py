"""pytest 共通 fixture。

- ROOT を CWD に固定（cv2.imread が日本語を含む絶対パスを扱えない Windows 対策）
- src/ を sys.path に追加
- session-scoped で recognition_results.json を読み込み、テストで使い回す
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

_DEFAULT_REC_PATH = ROOT / "tests" / "fixtures" / "colored_factors" / "recognition_results.json"


def _recognition_results_path() -> Path:
    value = os.environ.get("UMAFACTOR_RECOGNITION_RESULTS")
    if not value:
        return _DEFAULT_REC_PATH
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


@pytest.fixture(scope="session")
def recognition_results() -> dict:
    """scripts/batch_recognize.py が生成した認識結果 JSON を読み込む。

    TDD ループ:
      1. コードを修正
      2. `.venv/Scripts/python.exe scripts/batch_recognize.py` で認識を走らせ直す
      3. `pytest tests/test_recognition.py -v` で Red/Green を確認

    コード修正のたびに pytest 側で analyze_image を呼ぶと 26 画像 × 15s ≈ 6分で
    とても重くなるため、認識は scripts 側で一括してキャッシュする運用にする。
    """
    rec_path = _recognition_results_path()
    if not rec_path.exists():
        pytest.fail(
            f"{rec_path} が存在しません。"
            "先に `python scripts/run_phase0_regression.py --refresh` "
            "または `python scripts/batch_recognize.py` を実行してください。"
        )
    return json.loads(rec_path.read_text(encoding="utf-8"))
