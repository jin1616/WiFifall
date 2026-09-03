"""全局配置：路径、滑窗、模型超参与可视化默认设置。"""
from __future__ import annotations

import os
from pathlib import Path

# ----------------------------------------------------------------------------
# 路径
# ----------------------------------------------------------------------------
# 包位置: WiFall/wifalldemo/config.py  -> WiFall 才是数据根目录
BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = BASE_DIR  # 数据根目录直接存放 ID0~ID9
OUTPUT_DIR: Path = BASE_DIR / "outputs"
FIG_DIR: Path = OUTPUT_DIR / "figures"
MODEL_DIR: Path = OUTPUT_DIR / "models"
CACHE_DIR: Path = OUTPUT_DIR / "cache"
for d in (OUTPUT_DIR, FIG_DIR, MODEL_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# 类别（按目录名匹配，大小写不敏感）
# ----------------------------------------------------------------------------
# "跌倒"为正类（label=1），其他动态/静态动作合并为"非跌倒"（label=0）
POSITIVE_ACTIVITIES = {"fall"}
NEGATIVE_ACTIVITIES = {"jump", "sit", "stand", "walk"}
ACTIVITY_LABELS = {0: "Non-Fall", 1: "Fall"}

# 用户 ID 目录前缀
USER_PREFIX = "ID"

# ----------------------------------------------------------------------------
# CSI 解析参数
# ----------------------------------------------------------------------------
# 经过对 local_timestamp 间隔的统计，CSI 包速率约 83Hz。
# data 列每行有 104 个浮点 (52 子载波 × (实部, 虚部) 交织)
DATA_LEN_PER_ROW = 104
N_SUBCARRIERS = DATA_LEN_PER_ROW // 2
SAMPLING_RATE_HZ = 83.0  # 与 measured median delta≈12ms 对齐

# ----------------------------------------------------------------------------
# 滑窗参数
# ----------------------------------------------------------------------------
# WINDOW_SIZE≈1.54s, WINDOW_STEP≈0.39s，窗间重叠大以增加训练样本。
WINDOW_SIZE = 128
WINDOW_STEP = 32

# ----------------------------------------------------------------------------
# 实验设置
# ----------------------------------------------------------------------------
# 按人员（domain）划分训练/测试，避免同一受试者样本泄漏。
TEST_USERS = ("ID8", "ID9")
# 从训练人员中再留出的"验证人员"，用于跨人地挑选决策阈值。
VAL_USERS = ("ID6", "ID7")
RANDOM_STATE = 42

# ----------------------------------------------------------------------------
# 模型设置
# ----------------------------------------------------------------------------
RF_N_ESTIMATORS = 300
RF_MAX_DEPTH = None

LR_C = 1.0
LR_MAX_ITER = 2000
