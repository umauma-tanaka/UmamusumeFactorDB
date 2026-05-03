"""Detection constants for character section and factor box extraction."""

from __future__ import annotations

BASE_WIDTH = 540
BASE_HEIGHT = 960

LOW_SAT_THRESHOLD = 15.0
MIN_GRID_RUN_LEN = 200
SELF_HEADER_HEIGHT = 480
PARENT_HEADER_HEIGHT = 130

# 行検出パラメータ
ROW_CONTENT_STD_THRESHOLD = 15.0  # std がこれを超える行は content
ROW_MERGE_GAP = 4  # 隣接 content run をマージする許容 gap (px)。行間は通常 13px 以上
MIN_ROW_HEIGHT = 24  # 有効な因子行の最小高さ（継承元バナー h=22 を除外する値）
TARGET_ROW_PITCH = 42  # 因子行の実測ピッチ（px）。merge 後の分割に使う
SPLIT_THRESHOLD = 50  # merge 後の高さがこれを超えたら、複数行がまとまった結果として分割
# Row 0（青/赤因子）は低彩度 run 検出範囲より上に位置することがあるので、
# 各セクションで lookback 分だけ上へスキャンを広げる
# 本人は適性バッジなど UI 要素まで距離があるため大きめ、parent は banner 直下で小さめ
SELF_ROW0_LOOKBACK = 90
# parent は factor_y_start より上にも Row 0 が延びることがある。
# ただし名前/評価テキスト領域までは拾わないよう、banner 近傍のみ許容する
PARENT_ROW0_LOOKBACK = 70

# === ★検出（新経路）パラメータ =========================================
# 金★（埋まっている★）を HSV で拾う範囲。
GOLD_STAR_HSV_LO = (15, 120, 180)
GOLD_STAR_HSV_HI = (40, 255, 255)
# 旧名（後方互換のため残す）
STAR_HSV_LO = GOLD_STAR_HSV_LO
STAR_HSV_HI = GOLD_STAR_HSV_HI
# 空★（未点灯★）を HSV で拾う範囲。実測では S=7-80 (mean 24)、V=161-255 (mean 222)
# と金★の S=131-246 とは明確に分離できる。H は薄ピンク〜薄黄色を許容するため広めに。
# S の下限を 10 にすることでテキスト背景の真っ白 (S~0) を除外する。
EMPTY_STAR_HSV_LO = (0, 10, 200)
EMPTY_STAR_HSV_HI = (45, 90, 255)
GREEN_RELAX_GOLD_HSV_LO = (10, 30, 80)
GREEN_RELAX_GOLD_HSV_HI = (45, 255, 255)
GREEN_RELAX_EMPTY_HSV_LO = (0, 5, 180)
GREEN_RELAX_EMPTY_HSV_HI = (45, 100, 255)
# 空★は金★より少し小さめ（縁のみ塗り）だが ★形なので、最小 area を上げて
# 細切れノイズ（テキストのカーニング部など）を除外する。
EMPTY_STAR_MIN_AREA = 80
# ★連結成分のサイズ制限（正規化幅 540 基準、金/空共通）
STAR_MIN_W = 5
STAR_MAX_W = 25
STAR_MIN_H = 5
STAR_MAX_H = 25
STAR_MIN_AREA = 15
STAR_MAX_AREA = 400
# 同一行とみなす y 許容（★の中心間距離）
# 行ピッチは ~45px、★高さは ~15px。閾値を小さくし過ぎると、
# 単一因子行の★でも y がわずかにバラつくと 2 行に分裂してしまう
# （アド・アストラ 1 行が分裂して次行を誤検出するケースで顕在化）。
STAR_ROW_Y_TOL = 12
# 新経路の因子タイル寸法（正規化幅 540 基準）
# 実測：チップ(x=88付近) 〜 ★パディング右端(x=260付近) で幅 ≒ 175px
TILE_WIDTH = 175
TILE_HEIGHT = 27  # 旧 box_h=27 を踏襲（pipeline._crop_rank_from_original の y オフセットと整合）
# ★中心が bbox 内でこの y オフセットに位置するよう bbox.y0 を決める
# 旧 rank 領域 y=11..27 の中央 y=19 と一致させる
STAR_Y_IN_TILE = 19
# 金★最右端からタイル右端までの余白（金★3個の最右から空★+タイル縁までの距離）。
# タイル右端推定は金★のみで行うため、空★分 ~36px を足す想定。
TILE_RIGHT_PADDING = 36
# タイル右端推定に使う★最右端分布の percentile（★3個行の値を基準にする）
TILE_RIGHT_PERCENTILE = 90
# 新経路で必要な最小行数（これ未満なら legacy にフォールバック）
MIN_DETECTED_ROWS = 3
# タイル左端推定に使う★のサンプル数（少なすぎるとノイズに弱い）
MIN_STARS_PER_COLUMN = 3
# 金★の空間配置フィルタ（左詰め + 等間隔）
# 金★は UI 上で左端から等間隔で並ぶため、隣接金★の中心 x ピッチが
# この範囲内の連続群のみ有効な★スロットとみなす。
# TILE_WIDTH=175 に★3 個が並ぶと理論ピッチ ~18 px、±4 px の許容。
STAR_PITCH_MIN = 14
STAR_PITCH_MAX = 22
# Step 4 の空間配置フィルタをオン/オフするフラグ（問題発生時の即 rollback 用）
ENABLE_GOLD_LAYOUT_FILTER = True
