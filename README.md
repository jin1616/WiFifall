# WiFall — Wi-Fi CSI 跌倒检测（PyTorch 深度学习版）

> 基于 Wi-Fi CSI（信道状态信息）的**跌倒/非跌倒二分类**完整演示工程。
> 输入是 Widar3.0 风格的 CSI 数据集（每条样本是一段 `ID/<activity>/*.csv`），
> 输出主模型 + 指标 + 多张可视化图。从 **v0.3.0** 起，三个分类器全部走 PyTorch DL；
> 之前的 RandomForest / LogisticRegression 仅作历史基线保留，代码已不再调用。
> 从 **v0.4.0** 起，自采数据里的 `.wav` 录音可以作为第 4 路特征参与训练
> （`run_train.py --audio`），CSI+声音融合见「音频特征」一节。

> 🚀 **第一次用？直接看 [`新手上手指南.md`](新手上手指南.md)** —— 换台电脑从零开始，
> 含装环境、烧预编译固件（不用装 ESP-IDF）、验证链路、采集重训的全部命令。

---

## 状态

- **当前版本**：v0.4.0（新增音频通路：`.wav` 作为第 4 路特征）
- **数据规模**：928 个原始 CSI 样本（10 个用户 × 5 个动作）→ 滑窗后 ~4k 训练窗；
  自采 `realdata/` 404 样本 / 5 人，**每条都带 16 kHz 麦克风录音 + 时间轴对齐文件**
- **主模型（Widar）**：**CNN1D**（Conv-BN-ReLU-AdaptivePool-FC），窗级 Acc **0.856** /
  文件级 Acc **0.875**（`--no-normalize` 口径）
- **主模型（自采 + 音频）**：**MLP-Fusion**（70 维 CSI + 31 维音频 = 101 维），
  文件级判定为 **top-k 窗投票**（取概率最高的 3 个窗）
- **跨人 LOPO（5 人各留出一次，主口径 = top-k 窗投票，音频 v2 + 增益增广）**：
  文件级 Acc **0.790** / F1 **0.675** / 窗级 AUC 0.731；纯 CSI 同口径 Acc 0.443。
  Acc 比上一批（0.842）低是** honest 的代价**：旧口径靠 ID15 折"全判不跌倒"刷分（F1=0），
  现在 ID15 跌倒召回被音频居中救回（该折 F1 0→0.357），F1 整体 +0.04、最差折大幅收窄
  —— 音频增益在 5 个人身上依旧全部为正（详见"修复批次 v3"一节）
- **跨人评估**：Widar 用 ID8 / ID9 测试、ID6 / ID7 选阈值；
  自采用 `run_audio_ablation.py --lopo`（测试人轮流做，验证人自动避开）
- **测试**：6 个测试模块全通过（模型接口 / CNN 架构 / 目录扫描 / 逐子载波归一化 /
  音频对齐与降级 / 窗内转换特征，`pytest` 不依赖）

---

## 已知局限（代码改不动的那些）

诚实列在这里，写报告时不要回避：

1. **只有 5 个人、一个房间、一套摆位**。任何指标都带 ±4% 抖动，跨环境未验证。
2. **stand 仍有约 30% 误报**（修复批次：93% → 47% → 现在，各折判对率
   1.0 / 0.41 / 1.0 / 0.8 / 0.4）。静止站立与"跌倒后的静止"在 CSI 上本就相似，
   "坠落后静止"规则在本数据上未通过验证集调参（默认关闭，见修复批次 v3），
   再往下要补采 stand→walk 转换段的易混负样本。
3. **ID15 仍是短板折，但已从"塌"降为"弱"**：音频逐文件居中把该折 audio-only 的
   文件级 Acc 从 0.75（F1=0，全判负的假象）救到 0.963/0.923；融合模型该折
   0.550/0.357（受 CSI 分布漂移拖累，CSI-only AUC 仅 0.42）。**补采数据仍然最划算**。
4. **Audio-only 仍不能单独部署**：批次混杂被居中大幅缓解后它的数字变得很漂亮
   （LOPO 文件级 Acc 0.950），但跳跃落地也有撞击声，单模态在 jump 上必然翻车
   （ID15 折 jump 判对率 0）——它只作消融对照，主模型仍从 {mlp, dlsmall, mlp_av} 择优。
5. **CNN1D 在自采数据上 recall=0**（Widar 上 0.856）。数据量不足以支撑
   (128, 52) 张量端到端训练，换到真实采集就失效 —— 本版起 **CNN 不参与主模型
   择优与部署链路**，只作评估对照保留，报告里必须解释这个反差。
6. 全量训练（`final_all`）没有留出人，它的阈值是在训练集内选的（虚高），
   真实性能请引用 LOPO 数字。

---

## 数据说明

工程默认的数据布局：

```
<project_root>/DATA/
├── ID0/fall/2023-11-12_21-21-03-546_104_1.csv
├── ID0/walk/...
├── ID0/sit/...
├── ID1/fall/...
└── ...
```

> 原始 CSI 数据全部收在 **`DATA/`** 目录下（`config.DATA_DIR` 指向它）。
> 换位置只要改 `wifalldemo/config.py` 里的 `DATA_DIR` 一行，其余代码不受影响。

- 每个 `CSV` 是一段连续动作，逐行为一个 CSI 包（`type=CSI_DATA`）。
- `data` 列每行 104 个浮点数 = **52 个子载波 × [实部, 虚部] 交织**。
  与 Widar3.0 系列单链路 CSI 序列（Nexmon / ESP32 类采集端导出）一致。
- 目录名 → 二分类标签：`fall` → 跌倒（正类，label=1）；`jump / sit / stand / walk` → 非跌倒（负类，label=0）。
- 实测包速率 ≈ **83 Hz**（按 `local_timestamp` 间隔统计）。

### 声学旁车文件（自采数据专有）

采集端（`csi_audio_sync.py`）每条样本会同时录一段麦克风，产物是四件套：

```
realdata/ID11/fall/fall_001.csv           CSI（训练主通路）
realdata/ID11/fall/fall_001.wav           16 kHz / 16 bit 单声道，约 7 s
realdata/ID11/fall/fall_001.sync.json     音频↔CSI 的时间轴映射 + 对齐质量
realdata/ID11/fall/fall_001_audio_env.csv 逐 CSI 行的声音能量（质检旁证，不参与训练）
```

`run_train.py --audio` 会读前三个：`wav` 提声学特征，`sync.json` 把特征对齐到
CSI 的行时间轴。当前 `realdata/` 下 **404 条样本全部有 wav + sync**，一一对应。
Widar3.0（`DATA/`）没有音频，开 `--audio` 只会得到全零占位列，模型自动退化。

如果用手动方式采集、或只想复用别人录的 wav，只要保证三件套同名同目录即可；
只有 wav 没有 sync 时**无法对齐**，会按"无音频"处理（凭空猜偏移比不用更糟）。

如果用你自己的数据根目录，修改 `wifalldemo/config.py` 的 `DATA_DIR`；如果某条链路不是 52 子载波，改 `N_SUBCARRIERS`。

两处数据根目录（`config.py` 里都有常量）：

| 常量 | 默认路径 | 内容 |
|---|---|---|
| `DATA_DIR` | `DATA/` | Widar3.0 原始数据集 ID0~ID9（928 个 CSV） |
| `REALDATA_DIR` | `realdata/` | 自己用两块 ESP32-S3 采的数据（采集脚本的落盘位置） |

> **数据源**：训练/测试入口默认从 `outputs/data_sequence.pkl` 读（928 样本，120MB）。
> 这个 pickle 是用 `run_build_sequence.py` 把 `DATA/ID*/<action>/*.csv` 一次性打包后的结果。
> 想跳过 pickle、直接从 CSV 原始目录读：`--data-dir DATA`（`config.DATA_DIR`）。
> 详细对比见 [`docs/data-source.md`](#)（或 `.workbuddy/memory/MEMORY.md` 的 "data_sequence.pkl vs CSV：实测对比" 章节）。

---

## 安装与运行

### 1. 依赖

```bash
pip install -r requirements.txt   # numpy / pandas / matplotlib / scikit-learn / joblib
# 此外还需要 PyTorch (CPU 版即可)：
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

> **环境实测**：唯一可用的 conda 环境是 `big_create`（Python 3.11）。
> 首次跑 `run_train.py` 务必装 matplotlib 时用阿里源：
> `E:/Miniconde/envs/big_create/python.exe -m pip install -i https://mirrors.aliyun.com/pypi/simple/ matplotlib`

### 2. 数据准备（如尚未生成 pickle）

```bash
python run_build_sequence.py      # 输出 outputs/data_sequence.pkl
```

### 3. 训练 + 评估 + 出图

```bash
python run_train.py               # 默认：ID8/ID9 留出，启用用户均衡，训练 MLP/DLsmall/CNN
python run_train.py --no-cache    # 强制重建特征缓存（缓存命中可加速）
python run_train.py --no-cnn      # 跳过 CNN（CNN 训练原始 (128,52) 张量，最慢）
python run_train.py --verbose-dl  # 打印 MLP/DLsmall/CNN 每个 epoch 的损失

# 归一化相关（见下节；换数据源时这三个开关决定结果对不对）
python run_train.py --no-normalize          # 关掉逐子载波归一化（复现 Widar 基线必须用）
python run_train.py --norm-window 2.5       # 改归一化参考窗长度(秒)，默认 1.5
python run_train.py --norm-causal           # 参考窗只用历史帧（部署前训一次）

# 自采数据（--val-users / --no-balance / --no-cache 三个都别省）
python run_train.py --data-dir realdata --test-users ID13 \
    --val-users ID11 --no-balance --no-cache

# 部署模型：加 --norm-causal 让训练与流式推理的参考量口径一致
python run_train.py --data-dir realdata --test-users ID13 \
    --val-users ID11 --no-balance --norm-causal --no-cache

# 音频通路：把采集时录的 .wav 当第 4 路特征（58+31 维融合，详见"音频特征"一节）
python run_train.py --data-dir realdata --test-users ID13 \
    --val-users ID11 --no-balance --norm-causal --audio --no-cache

# 音频消融：对 ID12~ID15 各跑一遍 CSI-only / 融合 / 仅音频，出表 + 柱状图
python run_audio_ablation.py
```

主模型按 **文件级 F1** 择优挑选（通常是 CNN1D），落到 `outputs/models/primary.joblib` 或 `primary.pt`。

> ⚠ `--val-users` 不给的话会用 `config.VAL_USERS`（Widar 的 ID6/ID7），训自采数据时
> 阈值会退化成默认 0.5。`--no-cache` 也别省：特征缓存名不含数据目录，换数据源不加强制
> 重建会**静默复用上一份特征**，训出来的是假结果且毫无提示。

### 4. 推理演示

```bash
python run_demo_offline.py         # 离线回放：单样本逐窗打分，默认走 ID9，挑 fall/stand/walk 三段
python run_demo_realtime.py        # 流式检测模拟：逐包推理 + 报警
```

两个 demo 都支持模型选择：

```bash
python run_demo_offline.py --model cnn       # 显式指定 CNN1D
python run_demo_offline.py --model mlp       # 或 MLP
python run_demo_offline.py --model primary   # 默认走 metrics.json.primary_model
```

### 5. 单元测试

```bash
python -m tests.test_models_interface        # MLP / DLSmall / CNN 接口一致性
python -m tests.test_cnn_arch                # CNN1D 架构 / 训练 / 持久化
python -m tests.test_data_loader_scan        # 目录扫描只收 CSI 样本（跳过 *_audio_env.csv）
python -m tests.test_transition_features    # 窗内转换特征：冲击/匀速/静止三种形状可分
python -m tests.test_audio_features         # 音频对齐/降级/拼接（合成咔哒声定位到窗）
python -m tests.test_subcarrier_normalization  # 逐子载波归一化：对乘性 H(f) 不变 / 因果不泄漏未来
python -m tests.test_audio_features          # 音频：sync 对齐 / 缺失降级 / 58+31 维拼接
```

不需要 `pytest`；测试通过 `tests/__init__.py` 把项目根压到 `sys.path`，用纯 `assert` 验证。

> 采集侧的测试（`test_csi_*`）随 `tools/` 一起搬到 `../ESP32-S3` 了，在本仓库已不存在。

---

## 模型与指标

### 当前模型（v0.4.0）

| 模型 | 类型 | 通路 | 参数量 | 角色 |
|---|---|---|---|---|
| **CNN1D** | 1D-CNN | 原始 128×52 张量 | ~191k | 主模型（Widar） |
| **MLP-Fusion** | FC (256→128→64) | 58 维 CSI + 31 维音频 = 89 维 | ~57k | 主模型（自采，`--audio`） |
| MLP | FC (256→128→64) | 58 维手算特征 | ~24k | 强基线 |
| DLSmall | FC (128→64) | 58 维手算特征 | ~10k | 轻量基线 |
| MLP-AudioOnly | FC (256→128→64) | 31 维音频 | ~55k | 消融对照（**不参与择优**） |

CNN1D 架构速记：

```
Conv1d(52→32, k=5) → BN → ReLU → MaxPool(2)
  → Conv1d(32→64, k=5) → BN → ReLU → MaxPool(2)
  → Conv1d(64→128, k=5) → BN → ReLU
  → AdaptiveAvgPool(8) → Flatten
  → Linear(1024, 128) → ReLU → Dropout(0.3)
  → Linear(128, 2)
```

MLP / DLSmall 走 58 维手工特征：时域统计 / 差分能量 / FFT 频带 / 谱熵 / 自相关 / 形态矩，每组特征同时在原始幅度和子载波中位数归一化的相对幅度上计算。

### 主模型指标（CNN1D，Widar 数据集）

> 这组数字是 **`--no-normalize`** 跑出来的。默认（归一化开）在 Widar 上会掉到窗 AUC 0.844，
> 原因见下一节——**Widar 和自采数据的归一化默认值是相反的**。

| 指标 | 窗级 | 文件级（多数投票） |
|---|---|---|
| 准确率 | **0.856** | **0.875** |
| F1 | 0.856 | 0.871 |
| AUC | 0.935 | — |
| **跌倒召回** | — | **0.844**（27/32） |

按活动看（文件级）：`fall=0.844, jump=1.000, sit=0.750, stand=0.875, walk=1.000`。`sit` 仍是数据中最难与跌倒区分的动作。

> **这 0.935 里有一部分是"房间指纹"给的。** Widar 全程同一房间、同一套摆位，静态 |H(f)| 在不同
> 活动之间本身就不一样（跌倒在地面、站/走在另一高度），CNN 吃了裸幅度就顺手用了它。换到新房间
> 会缩水——关掉这部分线索后 Widar 窗 AUC 是 0.844。报数字时值得说清口径。

> **随机方差提示**：MLP / DLSmall 跨次训练指标会有 ±0.05 波动（PyTorch 算子顺序 + AdamW 内部状态）；
> CNN1D 完全确定，跨次数字一致。如要复现精确数字，优先看 CNN 指标。

### 历史基线（v0.2.x，RandomForest）

| 指标 (RandomForest) | 窗级 | 文件级（多数投票） |
|---|---|---|
| 准确率 | 0.797 | 0.844 |
| F1 | 0.817 | 0.861 |
| AUC | 0.860 | 0.935 |
| 跌倒召回 | 0.886 | **0.969**（31/32） |

CNN1D 在文件级 Acc 和 F1 上显著超过 RF；唯一弱项是跌倒召回（RF 0.969 → CNN 0.844），
如果业务侧重"宁可误报也要抓全跌倒"，可临时加载 `outputs/models/rf.joblib` 旧模型。

### 自采数据（ESP32-S3，92 样本 / 4 人）—— 归一化后

| 划分（val=ID11, `--no-balance`） | CNN 窗 AUC（归一化前 → 后） |
|---|---|
| test=ID12 用户均衡 | 0.845 → **0.862** |
| test=ID12 不均衡 | 0.909 → 0.868 |
| test=ID13 用户均衡 | 0.526 → **0.698** |
| test=ID13 不均衡 | 0.718 → **0.804** |
| test=ID14 用户均衡 | **0.194** → **0.919** |
| test=ID14 不均衡 | **0.097** → **0.921** |
| **均值 / 极差** | **0.548（极差 0.812）** → **0.845（极差 0.223）** |

收益主要不在均值，而在**下限**：归一化前最差的两组是 0.097 / 0.194——比随机还差、方向反了，
是 CNN 把静态 |H(f)| 当"这是谁"记住后的典型崩塌（ID14 只有 3 条 fall）。归一化后这类崩塌消失。

---

## 逐子载波归一化（关键：两个数据集默认值相反）

主模型 CNN1D 吃的是裸幅度，`|H(f)|` 的静态形状原封不动进网络。给 H(f) 加一点扰动，
CNN 窗级准确率 0.975 → 0.525（且是**漏报**方向）。自家板子频率选择性 2.5~2.9×、Widar 5.6×，
等于每个人一套 H(f)，跨人就翻车。

**做法**：`amp_norm[t, s] = amp[t, s] / median(amp[t0:t1, s])`，参考量是**逐子载波**的滑动中位数
（`wifalldemo/preprocessing.py :: normalize_per_subcarrier`）。对**乘性**信道是数学上精确不变的
——`amp → amp·g(s)` 时参考量同样乘 `g(s)`，比值分毫不变（实测相对误差 1.5e-07）。

| 开关 | 默认 | 说明 |
|---|---|---|
| `config.NORMALIZE_PER_SUBCARRIER` | `True` | 总开关 |
| `config.NORM_WINDOW_S` | `1.5` | 参考窗长度（秒）；实测 1.0/1.5 s 最好，3.5 s 明显变差 |
| `config.NORM_CAUSAL` | `False` | `False`=居中窗（离线评估）；`True`=只用历史帧（**部署**） |
| CLI `--no-normalize` | | 关掉（**复现 Widar 基线必须用**） |
| CLI `--norm-window SEC` | | 改参考窗长度 |
| CLI `--norm-causal` | | 因果窗（部署前训一次） |

**只在 CNN 的原始窗通路生效**：MLP / DLSmall 的 58 维手工特征里本来就有一半是逐子载波相对量
`rel_amp`，再归一化会让它退化成常数、白丢一半特征。

**⚠ 为什么 Widar 要关、自采要开**

| | Widar | 自采（ESP32-S3） |
|---|---|---|
| 采集条件 | 同一房间、同一套摆位、全程不变 | 每次摆位都可能不同 |
| 静态 \|H(f)\| 的含义 | 不同活动在不同位置/高度 → **真的带标签信息** | 只跟"这是谁/哪次摆位"相关 → **纯干扰** |
| 归一化 | **掉点**（窗 AUC 0.935 → 0.844） | **救命**（均值 0.548 → 0.845） |

**训练/推理口径**：离线训练用居中窗（质量略好），`run_demo_realtime.py` 流式推理天然只能用
因果窗。实测因果窗平均掉 0.042 AUC（ID12 +0.006 / ID13 −0.009 / ID14 −0.123，92 条样本分辨不了
这个量级）。但口径一致是**正确性问题不是调参问题**，所以：

- 评估/对比数字 → 居中窗
- 部署模型 → 用 `--norm-causal` 重训

**⚠ 归一化口径跟模型走，不跟 config 走。** 归一化是模型输入表示的一部分，训练时那套
（enabled / window_s / causal）会**存档进 `metrics.json` 的 `normalization` 字段**，
两个 demo 都从那里读（`load_norm_config()`），**改 `config.py` 不会改变已训好模型的推理行为**。
`metrics.json` 里没这个键（老版本训练的产物）时会退回 config 并打警告——看到警告就该重训。
流式物理上只能用因果窗，所以模型若是居中窗训的，`run_demo_realtime.py` 会明确提示口径不一致。

完整实测见 `outputs/归一化与重训报告.md`。

---

## 音频特征（第 4 路输入，`--audio`）

CSI 看到"人体运动改变了多径"，对"落地那一刻的撞击声"是**间接**的；声音对撞击是
**直接**的。开启 `--audio` 后，每条样本的 `.wav` 会变成第 4 路特征：58 维 CSI 手工
特征之后再拼 31 维声学特征（:mod:`wifalldemo/audio.py`），融合模型同时拿到运动学
证据和撞击证据。

### 怎么对齐（这不是"大概同时"）

音频和 CSI 是两台独立时钟，采集端用 `sync.json` 记下了两边的时间轴映射：

```
第 i 个 CSI 窗（第 start..start+128 行）
  → PC 墙钟 [csi_start_pc + start/83, csi_start_pc + (start+128)/83]
  → 音频采样区间 (t - audio_start_pc) × 16000
```

在 404 条自采样本上用逐行声音能量交叉验证：**映射相关系数最小值 1.0000**
（含 4 条做过互相关校正的样本）。单测 `test_audio_features` 会再造一个"已知时刻的
咔哒声"验证它确实落在正确的窗里。

### 31 维声学特征（:data:`AUDIO_FEATURE_NAMES`）

| 组 | 维数 | 内容 | 物理含义 |
|---|---|---|---|
| 包络/起振 | 11 | dBFS 均值/峰值/波动、峰均比、最大跳变、起振/衰减时长、峰值位置 | 跌倒的"砰"一声 → 峰值高、起振陡 |
| 频谱形状 | 10 | 谱质心/带宽/滚降/平坦度/通量/过零率/主频/谱熵 | 撞击是宽带的，走路是低频节律 |
| 四频带占比 | 4 | <250 / 250-1k / 1-4k / >4 kHz | 撞击能量往高频散 |
| 峰值帧局部 | 4 | 峰值帧的质心/低频占比/通量/信噪比 | "最响那一下"长什么样 |
| 元信息 | 2 | `audio_present` / `audio_coverage` | 让模型知道这行音频可不可信 |

设计取舍：

* **纯 numpy + 标准库 wave**，不引入 librosa/scipy —— 采集现场的轻量环境也能算。
* **缺音频不报错**：没有 wav/sync 的样本补全零 + `audio_present=0`，融合模型可以
  同时在"有声音/没声音"的数据上训练；麦克风坏了系统降级成纯 CSI，而不是崩掉。
* **特征顺序是接口**：改动 `AUDIO_FEATURE_NAMES` 必须同步 `AUDIO_FEATURE_VERSION`，
  缓存与旧模型都会因版本号失效，不会静默错位。
* **绝对电平逐文件居中（v2，`AUDIO_FEATURE_VERSION=2`）**：`audio_logrms_mean/max/p90`
  三个 dBFS 绝对量减去**该文件**的稳健中位电平。不同采集批次的录音增益/麦克风
  距离不同，绝对 dBFS 不可比——ID15 那批撞击声偏轻、整折失效的直接原因。居中后
  特征表达"相对本底的突出量"，批次差异被消掉；crest/onset/频带占比等增益不变量
  不动。测试 `tests/test_audio_features.py::test_per_file_level_centering` 用 ±14 dB
  增益差的两条录音验证居中后可比。
* 训练时另有 **±6 dB 增益抖动增广**（`--gain-aug-db`，只作用于训练窗），进一步
  强迫模型学相对形状而不是绝对电平。

### 用法

```bash
python run_train.py --data-dir realdata --test-users ID13 \
    --val-users ID11 --no-balance --norm-causal --audio
# 会额外训练两个模型：
#   mlp_av   CSI+音频融合（参与主模型择优）
#   audio    仅音频单模态（消融对照，不参与择优）
python run_audio_ablation.py          # 跨人消融：CSI-only vs 融合 vs 仅音频
```

### 实测（realdata，4 人各做一次留出，val=ID11，MLP 族，--no-cnn）

| 测试人 | CSI-only AUC/F1 | **CSI+Audio AUC/F1** | Audio-only AUC/F1 |
|---|---|---|---|
| ID12 | 0.604 / 0.392 | **0.875 / 0.656** | 0.904 / 0.656 |
| ID13 | 0.679 / 0.486 | **0.897 / 0.571** | 0.919 / 0.800 |
| ID14 | 0.456 / 0.065 | **0.746 / 0.682** | 0.753 / 0.308 |
| ID15 | 0.419 / 0.286 | **0.442 / 0.000** | 0.562 / 0.000 |
| **均值** | 0.540 / 0.307 | **0.740 / 0.477** | 0.784 / 0.441 |

### 五人留一交叉验证（LOPO，"这批数据整体准确率"的标准口径）

上面 4 行每人只被测了一次，方差大；把 ID11~ID15 **每个人轮流当测试人**（其余 4 人训练，
验证人自动避开测试人），5 折汇总才是"把 ID11~ID15 全部数据交给模型"的公允数字。
一键复现：

```bash
python run_audio_ablation.py --lopo        # 已有各折 metrics_ablation_ID*.json 时加 --reuse 直接汇总
```

结果（窗级 AUC / 文件级 Acc / 文件级 F1）：

（v2 特征 + top-k 文件级投票，2026-10-17 重跑。**最新口径（音频 v2 居中 + 增益增广）
的数字见上面"修复批次 v3"一节**，本表保留作历史对照）

| 测试人 | CSI-only | **CSI+Audio（融合）** | Audio-only |
|---|---|---|---|
| ID11 | 0.673 / 0.750 / 0.333 | **0.892 / 0.912 / 0.837** | 0.937 / 0.950 / 0.889 |
| ID12 | 0.613 / 0.506 / 0.494 | **0.882 / 0.627 / 0.563** | 0.906 / 0.855 / 0.769 |
| ID13 | 0.704 / 0.588 / 0.522 | **0.917 / 0.750 / 0.667** | 0.908 / 0.738 / 0.656 |
| ID14 | 0.480 / 0.395 / 0.329 | **0.718 / 0.765 / 0.457** | 0.744 / 0.765 / 0.240 |
| ID15 | 0.421 / 0.212 / 0.259 | **0.535 / 0.750 / 0.000** | 0.562 / 0.750 / 0.000 |
| **5 折均值** | 0.578 / 0.490 / 0.388 | **0.789 / 0.761 / 0.505** | 0.812 / 0.812 / 0.512 |

同一批模型换成 **top-k 窗投票**（每文件取概率最高的 3 个窗求平均）后，融合模型的
文件级指标显著改善：

| 测试人 | ID11 | ID12 | ID13 | ID14 | ID15 | **均值** |
|---|---|---|---|---|---|---|
| top-k Acc | 0.938 | 0.783 | 0.963 | 0.778 | 0.750 | **0.842**（+8.1） |
| top-k F1 | 0.857 | 0.690 | 0.930 | 0.700 | 0.000 | **0.635**（+13.1） |

> **top-k 必须配自己的阈值**。分数分布整体上移后沿用全窗均值的阈值，文件级 Acc
> 会从 0.761 掉到 0.558。`run_train.py` 现在在同一验证人上按同一口径重选
> `threshold_topk`，两者都存进 `metrics.json`（逐窗用 `threshold`，文件级报警用
> `threshold_topk`）。

即：**跨人场景下融合模型文件级准确率约 77%（窗级 Acc 73.1%、AUC 0.764）**。
注意读法：

* **别只看 Acc**。Audio-only 的文件级 Acc（0.824）看着最高，但它在 ID14/ID15 的
  跌倒召回是 0——把所有人都判成"没跌倒"就能在 3:1 的类别比下刷高 Acc；
  F1（0.542 vs 融合 0.534）和各折召回才是真话。
* **ID15 是明显的短板**（融合 AUC 0.44、F1 0）：那批录音跌倒撞击声偏轻，
  三种模型一起塌。补录更接近真实跌倒的样本是当前最划算的改进。
* 如果把 5 人数据**全部混进训练、再在同一批数据上测**，窗级 Acc 能刷到 95%+，
  但那是同一条录音的相邻窗同时出现在训练和测试里的**泄漏假象**，不能写进任何报告。

### 全量训练与泄漏对照（run_train_full.py）

部署时应该用上全部数据（`outputs/models/final_all.joblib` 就是 5 人 404 样本全量训出的
融合模型），但**全量训练之后就没有诚实的测试集了**。下面的对照（同一批数据、同一模型、
唯一变量是评估协议）量化了口径差异有多大：

| 评估协议 | CSI-only | CSI+Audio | Audio-only |
|---|---|---|---|
| LOPO（诚实口径） | 0.58 / 0.49 | **0.79 / 0.76** | 0.81 / 0.81 |
| 随机窗划分 80/20（泄漏） | 0.76 / 0.78 | 0.94 / 0.91 | 0.84 / 0.84 |
| 训练集内（最虚高） | 0.75 / 0.77 | 0.97 / 0.97 | 0.88 / 0.87 |

（每格：窗级 AUC / 文件级 Acc；完整指标见 `outputs/metrics_full_train.json`，
图见 `figures/train_full_leakage.png`）

三个结论：

1. **泄漏值多少钱：融合模型 +13~17 个点的文件级 Acc 全是假的**（0.767 → 0.904 → 0.933）。
   随机窗划分之所以几乎等于背答案，是因为窗步进 0.39s、窗长 1.54s——相邻窗重叠 75%，
   打乱随机切等于把同一段录音的"复印件"放进考场。
2. **CSI-only 连背答案都只有 0.80**（训练集内文件 Acc），说明 58 维手工 CSI 特征在这批
   自采数据上表达能力不足——这不是评估问题，是特征容量问题。
3. 全量模型 `final_all` 可以做部署候选，但它的性能预期只能引用 LOPO 数字（≈77%），
   任何来自协议 A/B 的指标都不能写进报告。

### 误报归因诊断（diag_false_alarm.py）

`per_activity_file_acc` 的读法要小心：它是**该活动的判对比例**，不是"被判为 fall 的
比例"。部署模型（test=ID13）按活动拆开是：

| 活动 | 判对比例（全窗均值） | 判对比例（top-k 投票） | 结论 |
|---|---|---|---|
| fall | 1.000 | 1.000 | 无漏报 |
| walk | 1.000 | 1.000 | 无误报（脚步声没有骗到模型） |
| stand | **0.200** | **0.533** | 误报 80% → 47%，**仍是最大问题源** |
| jump | 0.467 | 0.600 | 跳完落地被当成跌倒 |
| sit | 0.733 | **0.933** | 快速坐下基本分清了 |

修复主要来自两处：v2 的窗内"运动→骤停"转换特征，以及 top-k 文件级投票
（stand/sit 的误报通常是零星高分窗，top-k + 配套阈值能压住）。stand 从 93%
误报降到 47%，但依然是全系统最弱的一环。

用 `diag_false_alarm.py` 做的归因（CSI×音频 特征交换矩阵）：

```
CSI\AUD    fall   walk    sit  stand   jump     行=CSI 来源, 列=音频来源
fall      0.858  0.119  0.284  0.132  0.170
walk      0.735  0.102  0.255  0.109  0.123
sit       0.692  0.176  0.244  0.190  0.233
stand     0.674  0.544  0.448  0.554  0.599   ← 整行偏高
jump      0.824  0.243  0.377  0.266  0.293
```

结论：**stand 的误报由 CSI 通路 + 融合交互引入**——纯 CSI 模型给 stand 的得分只有
0.235（fall vs stand AUC 0.977，分得很好），融合模型反而抬到 0.554（AUC 掉到
0.907）；且 stand 的 CSI 无论配哪种音频都 0.54~0.67。根源是窗级手工特征里
"静止站立"与"跌倒过程中的窗"大量重叠（`raw_t_std` 10.57 vs 10.66，几乎相同）。

对症的改进方向（按优先级）：

1. **加"运动→静止转换"特征**：跌倒的本质是前一窗剧烈运动、当前窗骤停；
   stand 没有跳变。用前后窗能量差 / 因果运动能量是最对症的一刀。
2. **查 stand 的采集**：stand 音频 `active_ratio≈0.92`（静止时不该这么高）、
   `xcorr_corr≈0.1`（对齐未证实）——怀疑 stand 录音时有人在旁边活动或说话。
3. 文件级判定改"top-k 窗均值"替代全窗均值（跌倒是短时事件），配合冷却时间压误报。
4. 训练时把 fall 文件首尾的纯静止窗降权或改标负类，削弱"静止=fall"的伪相关。

> ⚠ 工程教训：换划分跑实验（消融/扫参）**必须加 `--no-save`**，否则最后一次实验
> 的权重会覆盖 `outputs/models/`，而 `metrics.json` 还停在上一次 —— 推理侧拿到
> 一个来历不明的模型，诊断时表现为"模型测它见过的数据"，AUC 虚高到 1.0。

读法：

* **融合对纯 CSI 是全用户一致的正增益**（AUC +0.20 / 文件 F1 +0.17 均值，
  4 个人没有一个变差）—— 这是"音频有增量信息"最硬的证据。
* **仅音频的 AUC 比融合还高，但别用它部署**：它在 ID14/ID15 的文件级 F1 崩掉
  （0.308 / 0.000）。声音单模态只会喊"有撞击"，而跳跃落地也有撞击声；CSI 恰好
  补上"这次撞击前有没有人体运动"这一层。两路证据互补，这正是做融合的理由。
* **ID15 对所有模型都差**：那批录音里跌倒的撞击声明显偏轻（人在软垫上倒），
  三个模型一起塌 —— 换句话说，音频通路的短板是"没有声音的跌倒"，这和 CSI 通路的
  短板（区分度低的慢倒）正好不同轴，也是多模态的价值所在。
* ⚠ **口径提醒**：仅音频 AUC 高有一个可能的混杂因素——不同人/不同批录音的麦克风
  增益和环境噪声不同。当前数据量（4 人 × 20 fall）还分不开"真的听到了撞击"和
  "这批录音整体更响"。上更多人、并有意变化录音音量后再下结论。

推理侧：`mlp_av` 的特征名（89 个）随模型存档在 `models/mlp_av_features.json`，
两个 demo 都按这个名字现算音频特征——**改了特征工程必须重训**，否则特征名对不上
会直接报错，而不是静默跑错。

```
┌─① 数据加载 data_loader ─ 扫描 ID*/<action>/*.csv -> 样本清单 (ID/活动/标签)
│                           -> 默认从 outputs/data_sequence.pkl 读 (120MB, 928 样本)
├─② 预处理     preprocessing ─ data 列 -> 复 CSI -> 幅度/相位 -> 去离群(Hampel)
│                        └─ 逐子载波归一化(仅 CNN 通路, 切窗前做): amp /= 滑动中位数
│        ┌─ 音频 audio（--audio）─ <sample>.wav + .sync.json
│        └─ 25ms 帧/10ms 跳 STFT -> 逐帧电平/起振/频谱/四频带 -> 按行区间取帧
├─③ 滑窗       dataset ─ 1.54s 窗 / 0.39s 步进，对每个样本生成若干重叠窗
│                              -> 输出三套并行:
│                                 (a) build_feature_dataset -> (N, 58) 特征矩阵 (给 MLP/DLsmall)
│                                 (b) 同上 + 音频 -> (N, 89) 融合特征 (给 MLP-Fusion)
│                                 (c) build_raw_window_dataset -> (N, 128, 52) 张量 (给 CNN)
├─④ 建模       models / cnn ─ 同步训练 MLP / DLSmall / CNN1D
│                              (+ --audio 时: MLP-Fusion / MLP-AudioOnly 消融)
│                              (PyTorch + AdamW + 类别权重 + CosineAnnealing + 早停)
├─⑤ 评估       run_train  ─ 跨人留出 (test=ID8/9) + 验证留出 (val=ID6/7)
│                              -> 窗级 + 文件级多数投票 两套指标
│                              -> 阈值在验证人员上按 Youden-J 选
└─⑥ 可视化     visualize  ─ CSI 热力图 / 三模型对比 / 混淆矩阵 / ROC
                                特征对比 / 离线 / 流式概率时间线 / 音频三视图
```

关键点：

- **人员级划分**：测试用户完全不出现在训练中，指标反映跨人泛化能力。
- **用户均衡**：每用户每动作文件数截断至全用户中位数，防止单人（ID2 有 20 倍轮次）主导分布。
- **滑窗增广**：1 段 ~3s 样本切成若干重叠窗，让 CNN 的 (128, 52) 张量足够训练。
- **特征设计**：时域/频域/形态 6 大类，每类在原始幅度 + 子载波中位数归一化两种归一化下都计算，58 维 = 29 × 2。
- **音频对齐**：音频与 CSI 靠 `sync.json` 的墙钟映射对齐（数学上就是一次线性换算），
  不是"大概同时开始"；每窗取的是**同一时间段**的音频帧，实测 404 条样本映射相关系数 1.0000。
- **阈值自适配**：Youden-J 在验证人员上选阈值（不是验证集最优，是折中灵敏度/特异度的稳定点）。
- **推理复用**：`window_feature_vector()` 为训练与流式推理共用；CNN 走 `_raw_windows_for_file()` 同源，
  `features_for_sample()` 让离线 demo 与训练走同一套切窗+拼接，保证训练/推理窗口切片一致。

---

## 修复批次：窗内时序结构特征（v2）+ top-k 投票

针对误报归因的结论做的一轮修复，全部有 LOPO 数据支撑。

### 1. 窗内"运动→骤停"转换特征（58 → 70 维）

原来 29×2 维特征**全是全窗平均**，把转换抹平了：全程静止的 stand 窗与跌倒文件
里冲击前/后的静止窗在这些维度上几乎同分布。v2 在每个窗内把活动度序列切成 8 段，
新增 6×2 维 `*_act_*`：

| 特征 | 含义 | stand | fall 冲击窗 | walk |
|---|---|---|---|---|
| `act_early_late` | 前半/后半活动度 log 比 | ≈0 | 明显为正 | ≈0 |
| `act_max_drop` | 最大骤降幅度 | ≈0 | **大** | ≈0 |
| `act_drop_pos` | 骤降发生位置 | 随机 | 有结构 | 随机 |
| `act_max_rise` | 最大上升幅度 | ≈0 | 大 | ≈0 |
| `act_peak_pos` | 活动峰位置 | 随机 | 中段 | 随机 |
| `act_tail_quiet` | 末尾相对安静度 | ≈0 | **很负** | ≈0 |

单元测试 `tests/test_transition_features.py` 用合成信号验证：冲击窗 drop=3.75 /
tail=−0.92，匀速窗 drop=0.07 / tail=0.01。

### 2. top-k 文件级投票（主口径）

每文件取概率最高的 3 个窗求平均。**必须配自己的阈值**（见上）。

### 3. 顺手修掉的三个工程缺陷

| 缺陷 | 后果 | 修法 |
|---|---|---|
| 缓存 meta 不校验特征版本 | 特征列含义变了仍复用旧缓存，训练不报错只会变蠢 | meta 加 `feature_version`，`FEATURE_VERSION=2` |
| 主模型从 CNN 换成 MLP 后 `primary.pt` 残留 | `load_model` 优先读 `.pt`，demo 会加载**上一次的 CNN**，且与新特征名不匹配 | `save_model` 保存时清掉同 tag 的另一种格式 |
| `config.VAL_USERS` 被上次实验留在 `ID14` | 跑 Widar 时 ID14 不在数据集里，阈值退化成 0.5 | 恢复为文档默认 `ID6/ID7` |

### 修复前后对照（五折 LOPO 均值，融合模型）

| 口径 | 修复前 | 修复后 |
|---|---|---|
| 窗级 AUC | 0.764 | **0.789** |
| 文件级 Acc（全窗均值） | 0.767 | 0.761 |
| 文件级 F1（全窗均值） | 0.534 | 0.505 |
| 文件级 Acc（top-k） | — | **0.842** |
| 文件级 F1（top-k） | — | **0.635** |
| ID13 折 stand 判对率 | 0.067 | **0.533**（top-k） |

即：**主口径（top-k）文件级 Acc 77% → 84%，F1 53% → 64%**。CSI-only 的 AUC
也从 0.560 提到 0.587 —— 新特征确实增强了 CSI 通路的表达能力。

## 修复批次 v3：音频 v2 归一化 + 打分后处理实验（2026-10-17）

针对遗留四问题（ID15 整折塌 / stand 误报 / 音频批次混杂 / CNN 失效）的一轮修复。
核心代码在 `wifalldemo/postprocess.py`（新增：文件内相对打分、top-k 投票、
"坠落后静止"确认规则）与 `wifalldemo/audio.py`（v2 逐文件电平居中）。

### 1. 音频特征逐文件居中（`AUDIO_FEATURE_VERSION=2`，默认生效）

`audio_logrms_mean/max/p90` 减该文件稳健中位电平。这是四个问题里**唯一一个
治本且实测有效的**：ID15 折 audio-only 文件级从 0.75（F1=0 的假象）→ **0.963**，
五折 audio-only 均值 Acc 0.812 → 0.950 —— 批次增益混杂被从特征层消掉。

### 2. 打分后处理：三个候选里只保留 raw（重要负结论）

动机是"绝对阈值不跨批次迁移"，于是在**分数层**试了三种变换（阈值都在同一变换
空间的验证集上选）：

| score-mode | 融合 top-k Acc / F1（5 折均值） | 结论 |
|---|---|---|
| `rel`（文件内 z-score） | 0.280 / 0.261 | **惨败**。窗级 AUC 掉到 ~0.5：top-k 取的是组内极值，z 分极值由"该文件分数分布的形状"决定，跨文件趋于常数——绝对置信度信息被整体抹掉 |
| `hybrid`（z + 0.25×raw） | 0.317 / 0.270 | 同样失败：z 分量主导 top-k 极值，0.25 的 raw 权重救不回来 |
| `raw`（原始概率，**默认**） | **0.790 / 0.675** | 保留。MLP 的绝对校准在 4/5 的人身上本来就能迁移，ID15 交给特征层居中解决 |

rel/hybrid 保留在 `--score-mode` 里供研究，但**不要默认开**。这条负结论值得写进
报告："分数归一化不是免费的 lunch——它把模型已经学到的跨文件校准一并抹掉了"。

### 3. "坠落后静止"时序确认规则（实现完整，默认关闭，`--stillness` 开启）

top-k 触发后检查触发点后 ~1.5 s 的 CSI 活动度：没显著静下来（或触发能量相对
文件基线不够暴）就否决。参数在验证人上网格搜索（含"不用规则"候选），验证集
不支持时自动退回。但 LOPO 五折里它**对融合模型从未通过调参**（单验证人 × 少量
stand 文件，网格搜索太噪），对 CSI-only 还略有负收益 —— 故默认关闭。
stand 误报目前主要靠 top-k 投票 + 音频特征压着（见上节各折判对率）。

### 4. CNN 退出部署链路

`run_train.py` 主模型择优集合改为 {mlp, dlsmall, mlp_av}（`DEPLOY_EXCLUDE =
{"audio", "cnn"}`）。CNN 照常训练、照常进 metrics.json 作对照，但不可能再被
选为主模型，demo 也不会加载到它。

### 5. 修复前后对照（五折 LOPO 均值，融合模型，top-k 主口径）

| 口径 | v2 批次（旧） | v3 批次（本版） |
|---|---|---|
| 文件级 Acc | 0.842 | 0.790（↓ 但 honest：ID15 不再靠"全判负"刷分） |
| 文件级 F1 | 0.635 | **0.675** |
| 窗级 AUC | 0.789 | 0.731 |
| ID15 折 Acc / F1 | 0.750 / **0.000** | 0.550 / **0.357** |
| audio-only Acc / F1 | 0.812 / 0.512 | **0.950 / 0.912** |
| stand 各折判对率 | 0.533（ID13 折） | 1.0 / 0.41 / 1.0 / 0.8 / 0.4 |

读法：**F1 与最差折改善是真金白银**（跌倒检测是低基率问题，F1 比 Acc 诚实）；
Acc 与 AUC 的小幅回落来自 ID13/ID14 两折的高分位抖动和 ID15 从"弃权"变成
"作答"。音频增益在 5 个人身上依旧全部为正。

一键复现：

```bash
python run_audio_ablation.py --lopo                 # 本版默认口径（raw + 居中 + 增广）
python run_audio_ablation.py --lopo --score-mode rel --tag _rel   # 负结论对照
python run_audio_ablation.py --lopo --stillness --tag _stil       # 规则开启对照
```

## 工程结构

```
WiFall/
├── README.md
├── requirements.txt
├── run_build_sequence.py   # 把 ID*/<action>/*.csv 打包成 outputs/data_sequence.pkl
├── run_train.py            # 端到端训练 + 评估 + 出图
├── run_demo_offline.py     # 离线检测演示
├── run_demo_realtime.py    # 流式(实时)检测模拟
├── wifalldemo/             # 核心库
│   ├── config.py           # 全局配置（路径/类别/滑窗/归一化/音频/DL 超参）
│   ├── data_loader.py      # 数据集扫描与人员划分
│   ├── preprocessing.py    # CSI 解析与清洗
│   ├── audio.py            # 音频：wav 解析 + sync 对齐 + 31 维声学特征
│   ├── features.py         # 时/频域特征（58 维）
│   ├── dataset.py          # 特征数据集（CSI/CSI+音频）+ npz 缓存 + 原始窗张量
│   ├── models.py           # MLP / DLsmall 训练/评估/阈值
│   ├── cnn.py              # CNN1D 架构 + train_cnn + save_cnn/load_cnn
│   └── visualize.py        # matplotlib 出图（含音频三视图）
├── DATA/                   # 原始 CSI 数据：ID0/ ... ID9/（每人 32 个 csv）
├── outputs/
│   ├── data_sequence.pkl   # 928 样本预打包
│   ├── cache/              # 特征/原始窗缓存 (build_feature_dataset 持久化)
│   ├── models/             # mlp.joblib / dlsmall.joblib / cnn.pt / mlp_av.joblib
│   │                       #   / audio.joblib / primary.*
│   ├── figures/            # demo_*.png / cm_*.png / roc_*.png / audio_example_*.png ...
│   └── metrics.json        # 全部指标汇总（含 primary_model / audio 字段）
└── tests/                  # 单元测试（5 个，纯 assert，都不需要硬件）
    ├── __init__.py
    ├── test_models_interface.py
    ├── test_cnn_arch.py
    ├── test_data_loader_scan.py
    ├── test_subcarrier_normalization.py
    └── test_audio_features.py
```

---

## 输出（`outputs/`）

| 文件 | 说明 |
|---|---|
| `data_sequence.pkl` | 928 条 CSI 打包，预处理阶段已经包好；训练时默认读这个 |
| `cache/feat_*.npz` / `raw_*.npz` | 滑窗缓存；命中复用，加 `--no-cache` 强制重建 |
| `models/mlp.joblib` | MLP 序列化（`torch.save` state_dict + arch meta） |
| `models/dlsmall.joblib` | DLSmall 序列化 |
| `models/cnn.pt` | CNN1D 完整 PyTorch 模型（arch + state） |
| `models/mlp_av.joblib` | CSI+音频融合 MLP（`--audio` 时生成） |
| `models/audio.joblib` | 仅音频 MLP（消融对照） |
| `models/primary.*` | 主模型软链（按文件级 F1 择优，audio-only 除外），demo 自动加载 |
| `metrics.json` | 全部模型的窗级 + 文件级指标 + 阈值 + per-activity 矩阵 + `audio` 块 |
| `audio_ablation.json` | 跨人音频消融结果（`run_audio_ablation.py`） |
| `audio_ablation_lopo.json` | 五人留一交叉验证汇总（`--lopo`，ID11~ID15 各留出一次） |
| `metrics_full_train.json` | 全量训练 + 三种评估协议对照（`run_train_full.py`） |
| `models/final_all.joblib` | 全量 ID11~15 训出的融合模型（部署候选，性能以 LOPO 为准） |
| `figures/train_full_leakage.png` | 评估口径泄漏对照图（LOPO vs 随机窗划分 vs 训练集内） |
| `figures/cm_window_*.png` / `cm_file_*.png` | 三个模型的窗级/文件级混淆矩阵 |
| `figures/roc_*.png` | 三个模型的窗级 ROC |
| `figures/model_compare.png` | 三模型对比柱图 |
| `figures/feature_compare.png` | 特征对比（如果有意义） |
| `figures/csi_example_*.png` | 训练集单个 fall / 非fall 样本的 CSI 热力图 |
| `figures/demo_*.png` | 离线 demo 三样本（fall/stand/walk）打分时间线 + CSI 图 |
| `figures/audio_example_*.png` | 音频三视图（波形 / dB 包络 / 语谱图，绿区=CSI 覆盖段） |
| `figures/audio_ablation.png` | 跨人消融柱状图（CSI-only vs 融合 vs 仅音频） |
| `figures/audio_ablation_lopo.png` | 五人留一交叉验证柱状图 |
| `figures/realtime_timeline.png` | 流式 demo 的实时报警时间线 |

---

## 真实设备接入（两块 ESP32-S3）

原来这里只有一句"把生成器换成串口回调"。现在两块板的固件和桥接脚本都已补齐。

> ## 📦 采集的东西已搬到 `../ESP32-S3`
>
> **2026-09-23 起，固件、采集脚本、示例、采集包全部搬到仓库外面的兄弟目录
> `ESP32-S3/`**（与 `WiFall/` 同级）。本仓库只留训练 / 评估 / 演示。
> 采数据时请 `cd ../ESP32-S3` 再跑，那里的 `README.md` 是采集的完整上手说明。
>
> ```
> ESP32-S3/
> ├── tools/          11 个脚本：采集 / 串口体检 / 单条质检 / 整批体检 / 声音同步
> ├── wifalldemo/     config.py + preprocessing.py（不含 torch 模块）
> ├── firmware/       prebuilt(rx/tx + flash.bat) + rx/tx 源码 + micropython 参考
> ├── examples/       5 个示例（含无硬件回放）
> ├── realdata/       采出来的数据落这儿
> └── README.md       装依赖 → 烧固件 → 体检 → 摆位 → 采集 → 质检 → 拷回训练
> ```
>
> 本仓库的 `wifalldemo/config.py` 与 `preprocessing.py` **保留**（训练流水线要用），
> 所以采集包里那两份是副本，改了要同步。
>
> 下面几节的命令都假设你在 `ESP32-S3/` 目录下。

想在**自己的 Python 脚本**里驱动硬件，不用 subprocess 调 CLI：

```python
from tools.csi_easy import CsiSession

with CsiSession("COM5") as s:          # 打开 A 板（RX）串口
    rec = s.record(seconds=6.0)        # 录 6 秒（自动重采样到 83 Hz 均匀网格）
rec.save("realdata/ID10/fall/fall_001.csv", label="fall")   # 直接喂 run_train.py
```

### CSI + 声音同步采集（开发板采 CSI，PC 同时录麦克风）

```bash
python tools/csi_audio_sync.py --list-devices                  # 看麦克风
python tools/csi_audio_sync.py --port COM5 --label fall        # 一次录三件套
```

一次产出 `fall_001.csv`（CSI，可直接训练）+ `fall_001.wav`（16k/16bit）+ `fall_001.sync.json`
（对齐元数据）+ `fall_001_audio_env.csv`（逐 CSI 行的声音能量）。两个设备时钟不同源，
模块用"首帧到达时刻 + 声卡延迟"把 ESP32 的 `ts_us` 映射到 PC 时钟轴，并用
CSI 扰动能量与声音能量的互相关给出滞后量做校验。详见 `tools/csi_audio_sync.py` 文件头。

`tools/csi_easy.py` 本身也可以直接跑（倒计时 → 录一段 → 质检 → 自动编号存档）：

```bash
python tools/csi_easy.py --port COM5 --label fall          # 录一条 fall，存到 realdata/ID10/fall/
python tools/csi_easy.py --replay outputs/realdata/esp32_resample.csv   # 没板子先演练
```

> **板子能不能烧 Python？** 能。ESP32-S3 是 MicroPython 官方目标，且从 v1.29.0 起
> `network.WLAN` 自带 CSI API，连 LTF 参数（硬编码 `htltf_en=1`）都跟我们的 C 固件
> **一致**（早期我误判它"跟我们相反"，其实是当时自己的 LLTF 配置有 bug，见
> `../ESP32-S3/firmware/README.md` 坑 7）。真正的差别是 MicroPython **必须连上 AP**、没有
> promiscuous 退路，也拿不到 `first_word_invalid`，正式采集还是用 C 固件。
> 详见 `../ESP32-S3/firmware/micropython/README.md`。

### 两块板分别烧什么

CSI 是**接收机**对信道的估计，发射机拿不到，所以两块板**职责不同、不可互换**：

| 板 | 角色 | 目录 | 干什么 | 出数据吗 |
|---|---|---|---|---|
| **A 板** | 接收端 RX | `esp32s3_csi_rx/` | 锁 HT20 + 只开 **HT-LTF**，把 64 个 tone 剪成 52 个数据子载波，交换实/虚部，组 119 字节二进制帧从 UART 吐出 | **出**，USB 接 PC |
| **B 板** | 发射端 TX | `esp32s3_csi_tx/` | ESP-NOW 定频喷包（默认 200 pps / MCS0 / HT20 / 200 B），把空口报文密度顶上去 | 不出，独立供电即可 |

一句话：**A 板接电脑负责"听"，B 板负责"喊"**。B 板上跑的是普通 ESP-NOW 发包程序，
不含任何 WiFall 逻辑；A 板才是这套系统的核心。

```bash
# A 板
cd firmware/esp32s3_csi_rx && idf.py set-target esp32s3 && idf.py -p COM3 flash monitor

# B 板（另一个串口号）
cd firmware/esp32s3_csi_tx && idf.py set-target esp32s3 && idf.py -p COM8 flash monitor
```

两边**必须同信道**（默认都锁 ch 6）且**都是 HT20**。

> B 板不是必需的。有路由器的话，让 PC 持续 ping A 板、或跑 iperf 灌流，同样能提供
> 报文密度。B 板固件是"现场只有两块板、没有可用路由器"时的确定性方案。

### A 板固件必须做对的三件事

1. **强锁 HT20** — ESP32-S3 默认 HT40，会给 114 个子载波，本项目只认 52 个
2. **只开 HT-LTF**（不是 LLTF）— LLTF 只覆盖 −26..+26，按 −28..+28 取会让最外侧 4 路恒为 0；HT-LTF 覆盖 −28..+28（去 DC 共 56），剔除 DC(0) 与导频(±7, ±21) 后正好剩 52 个数据子载波
3. **交换实部/虚部顺序** — ESP32 `buf` 是"虚部在前"，本项目 `parse_data_field` 期望实部在前

采集与实时检测：

```bash
# 带标签采集（倒计时 + 逐条质检 + 自动编号，产物可直接 --data-dir 重训）
python tools/csi_collect_labeled.py --port COM3 --user ID10 --action fall --repeat 10

# 原始采集（重采样到 --target-hz，默认 83Hz，与训练数据一致）
python tools/csi_serial_to_csv.py dump --port COM3 --duration 10 \
    --label fall --out realdata/ID10/fall/esp32_001.csv

# 或者直接实时检测（用 outputs 里的主模型 + metrics.json 阈值）
python tools/csi_serial_to_csv.py live --port COM3
```

**两个必须知道的前提**（`firmware/README.md` 有详细说明）：

- **采样率由空口报文密度决定**，无法人为设定。路由器 beacon 只有 ~10 Hz，要接近训练数据的 83 Hz 必须有高密度发包源——这正是 B 板存在的理由。
- **存在跨域差距**：现有 CNN 权重是在 Widar3.0 / Nexmon 类数据上训练的，与 ESP32-S3 的 CSI 幅度量纲不同。几何上（52 子载波 + HT20）对得上，但数值分布对不上——**要达到上面那些指标，必须用这套固件采集自有数据后重新训练**。

---

## 调参记录

- **CNN 强化正则 → 反而欠拟合**：试过 `Dropout=0.5 + label_smoothing=0.05`，窗 Acc 从 0.856 → 0.763。已退回 `Dropout=0.3 + 无标签平滑`。小数据集上正则要比想象的弱。
- **DLSmall 升级**：单层 64 → `(128, 64)` + Dropout 0.3 + 80 epochs，窗 Acc 0.612 → ~0.65–0.70。
- **判定阈值靠验证人员 Youden-J 选**，不在测试集上自选，避免过拟合。
- **跨次方差**：MLP/DLSmall 每次训练指标波动 ±0.05；CNN 跨次完全一致。如要复现具体数字，跑 CNN。

---

_这份 README 跟着代码走。改了配置/模型/测试请同步更新；CLI 参数变化请在"安装与运行"补一句。_
