"""全局配置：路径、滑窗、模型超参与可视化默认设置。"""
from __future__ import annotations

import os
from pathlib import Path

# ----------------------------------------------------------------------------
# 路径
# ----------------------------------------------------------------------------
# 包位置: WiFall/wifalldemo/config.py  -> WiFall 才是数据根目录
BASE_DIR: Path = Path(__file__).resolve().parent.parent
# 原始 CSI 数据（ID0~ID9）统一放在 BASE_DIR/DATA/ 下。
# 没建 DATA/ 目录时退回项目根布局，兼容旧 checkout。
DATA_DIR: Path = BASE_DIR / "DATA" if (BASE_DIR / "DATA").exists() else BASE_DIR
# 自己新采的数据（tools/csi_collect_labeled.py 等的默认落盘位置）
REALDATA_DIR: Path = BASE_DIR / "realdata"
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
# 逐子载波归一化（去掉静态 |H(f)| 形状）
# ----------------------------------------------------------------------------
# 为什么需要：主模型 CNN1D 吃的是**裸幅度**，|H(f)| 的静态形状（频率选择性）
# 会原封不动进网络。实测给 H(f) 加一点扰动，CNN 窗级准确率 0.975 → 0.525，
# 而且是**漏报**方向；而 MLP（特征里有一半是逐子载波的相对量）同样扰动只掉 2.5 个点。
# 自家板子和 Widar 的选择性差一倍（2.5~2.9× vs 5.6×），等于每个人的 H(f) 形状
# 都不一样 —— 模型只能记住"这是谁"，换人就翻车。
#
# 做法：每条样本自己算一个**逐子载波的滑动参考量**（1~2 s 窗内的中位数），
# amp_norm[t,s] = amp[t,s] / ref[t,s]。这对乘性信道 H(f) 是**数学上精确不变**的：
# amp 乘 g(s) 时 ref 也乘 g(s)，比值不变（实测变化 0.0%）。
#
# 只对 CNN 的原始窗通路生效（MLP/DLSmall 的手工特征里本来就有逐子载波相对量
# rel_amp，再归一化反而会让 rel 退化成常数、白丢一半特征）。
NORMALIZE_PER_SUBCARRIER = True
# 参考量的滑动窗长度（秒）。1~2 s：短到能跟上环境慢变，长到不被 0.5~1 s 的
# 跌倒瞬态自己抬起来。
NORM_WINDOW_S = 1.5
# 参考窗取法：False=居中（离线训练/评估，质量略好）；True=只用历史帧（因果）。
# 流式推理天然只能用因果窗，所以**部署前应该用 causal=True 训一次**，
# 让训练和推理吃同一套参考量。见 run_train.py --norm-causal。
NORM_CAUSAL = False

# ----------------------------------------------------------------------------
# 音频通路（第 4 路特征）
# ----------------------------------------------------------------------------
# 采集端（csi_audio_sync.py）每条样本会录一段 16 kHz 麦克风音频，并写一份
# `<sample>.sync.json` 记录音频与 CSI 的时间轴映射。开启后，每个滑窗在 58 维
# CSI 手工特征之后再拼 31 维声学特征（见 wifalldemo/audio.py::AUDIO_FEATURE_NAMES），
# 融合模型因此能同时看到"人体扰动多径"和"落地撞击声"。
#
# 默认关：Widar3.0 数据集没有音频，开了也只会得到全零占位列；自采数据用
# run_train.py --audio 打开即可，无需改这里。
AUDIO_ENABLED = False
# 没有音频的样本填全零 + audio_present=0，而不是丢样本 —— 融合模型可以同时在
# "有声音/没声音"的数据上训练，部署时麦克风坏了也只是退化掉音频那一半证据。
AUDIO_ALLOW_MISSING = True

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
VAL_USERS = ("ID6", "ID7")   # 阈值选择的验证人员（Widar 默认；realdata 实验请用 --val-users 显式指定）
RANDOM_STATE = 42

# ----------------------------------------------------------------------------
# 模型设置
# ----------------------------------------------------------------------------
# 原随机森林/逻辑回归超参保留作历史参考；当前训练流程已切到深度学习。
RF_N_ESTIMATORS = 300
RF_MAX_DEPTH = None

LR_C = 1.0
LR_MAX_ITER = 2000

# ---- 深度学习超参 (PyTorch) ------------------------------------------------
# 这些量供 train_mlp / train_dlsmall / train_cnn 读取；任何修改都不影响下游
# 调用接口，只是改变训练行为。
DL_DEVICE = "cpu"               # 训练样本量 ~1.1k 窗，CPU 足够且方便复现
DL_SEED = 42                    # PyTorch/numpy 随机种子
MLP_HIDDEN = (256, 128, 64)     # MLP 三层隐藏单元，替换原 RF
MLP_DROPOUT = 0.4
DL_EPOCHS = 60                  # MLP 训练轮数 (DLSmall 用 DLS_EPOCHS)
DL_BATCH_SIZE = 64
DL_LR = 1e-3                    # 初始学习率
DL_WEIGHT_DECAY = 1e-4          # AdamW 权重衰减
DL_PATIENCE = 10                # MLP 早停耐心 (epochs)

# CNN 单独的批大小/学习率 (CNN 输入为 (128, 52)，更大体积)
CNN_EPOCHS = 40
CNN_BATCH_SIZE = 64
CNN_LR = 1e-3
CNN_WEIGHT_DECAY = 1e-4
CNN_PATIENCE = 8

# ---- 模型改进 (v0.3.0) ---------------------------------------------------
# 历史版本 DLSmall (单层 64) 在 1100 窗上欠拟合；提升到 (128, 64) + Dropout 0.3
# 实测：窗 Acc 0.612 → 0.698，已保留。
DLS_HIDDEN = (128, 64)
DLS_DROPOUT = 0.3
DLS_EPOCHS = 80
DLS_PATIENCE = 12

# CNN 训练样本约 1100 窗，过强正则会欠拟合；保持 Dropout=0.3 + 不做标签平滑
# 实测对比: Dropout=0.5+label_smoothing=0.05 让 CNN 窗 Acc 从 0.856→0.763，
# 故仍走原强度 (Dropout=0.3)。
CNN_DROPOUT = 0.3
CNN_LABEL_SMOOTHING = 0.0

# ---- 数据增强 + 结构增强 (v0.4.0) ------------------------------------------
# 小样本跨人场景下，CNN 直接吃几十条原始窗会硬背特定子载波/幅度/时延（AUC 在
# 0.097~0.909 之间剧烈摆动就是典型症状）。在线增强等价于把样本量凭空放大若干倍，
# 强迫模型学"动作本身的形态"而非"某个人/某个房间的静态指纹"。
CNN_AUGMENT = True          # 训练时在线增强原始窗（子载波掩码/增益抖动/噪声/平移）
CNN_AUG_P_MASK = 0.5        # 每次 batch 触发子载波掩码的概率
CNN_AUG_MAX_MASK = 4        # 最多掩码的子载波数（52 个里最多遮 4 个，保留大部分信息）
CNN_AUG_SCALE = (0.9, 1.1)  # 全局增益抖动范围（乘性，符合幅度物理）
CNN_AUG_JITTER = 0.02       # 高斯噪声标准差（相对幅度 ~1.0 基线很小）
CNN_AUG_MAX_SHIFT = 4       # 时间平移最大帧数（约 48 ms）

CNN_USE_SE = True           # 在卷积块里加 SE 通道注意力
CNN_SE_REDUCTION = 16       # SE 降维比
