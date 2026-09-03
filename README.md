# WiFi 跌倒检测 Demo（Widar3.0 兼容 CSI 数据）

基于 Wi-Fi CSI（信道状态信息）的**跌倒/非跌倒二分类**完整演示工程。
输入为 Widar3.0 风格的多人员、多动作 CSI 数据集，输出包括模型、指标与多张可视化图。

## 数据说明

本工程默认数据布局（Widar3.0 常用整理形态，"人 / 动作"切片）：

```
<data_root>/
├── ID0/fall/2023-11-12_21-21-03-546_104_1.csv   # 一条跌倒动作 CSI 流
├── ID0/walk/...
├── ID0/sit/...
├── ID1/fall/...
└── ...
```

- 每个 `CSV` 是一段连续动作，逐行是一个 CSI 包（`type=CSI_DATA`）。
- `data` 列每行 104 个浮点数，按 **52 个子载波 [实部,虚部] 交织**编码；
  该格式与 Widar3.0 系单链路 CSI 序列（Nexmon/ESP32 类采集端导出）一致。
- 目录名映射二分类标签：`fall` → 跌倒(正类)；`jump/sit/stand/walk` → 非跌倒(负类)。
- 实测包速率 ≈ **83 Hz**（由 `local_timestamp` 间隔统计得出）。

如需切换为自己的数据根目录，修改 `wifalldemo/config.py` 中的 `DATA_DIR`；
若某条链路不是 52 子载波，修改 `DATA_LEN_PER_ROW / N_SUBCARRIERS`。

## 安装与运行

```bash
pip install -r requirements.txt

python run_train.py            # ① 端到端训练 + 评估 + 出图
python run_demo_offline.py     # ② 离线回放：单样本逐窗打分演示
python run_demo_realtime.py    # ③ 流式(实时)检测模拟：逐包推理+报警
```

`run_train.py` 可传参：`--test-users ID8 ID9`（跨人评估）、`--window-size 128`、
`--window-step 32`、`--no-cache`（强制重建特征缓存）、`--no-balance`（关闭用户均衡）等。

## 参考结果（本仓库默认实验，随机种子固定）

测试：跨人留出 **ID8 / ID9**；训练：其余 8 人并做“用户均衡”
（每用户每动作文件数截断至全用户中位数，避免单人 20 倍轮次主导模型）。

| 指标 (主模型 RandomForest) | 窗级 | 文件级(多数投票) |
| --- | --- | --- |
| 准确率 | 0.797 | 0.844 |
| F1 | 0.817 | 0.861 |
| AUC | 0.860 | 0.935 |
| 跌倒召回 | 0.886 | **0.969** (31/32) |

按活动看（文件级），`stand=1.00, jump=0.875, walk=0.75`，
`fall=0.969`；`sit` 是本数据中最难与跌倒区分的动作（详见混淆矩阵与
`metrics.json`）。逻辑回归基线召回更高(1.0)但误报略多，两者权重均已保存，
演示默认加载择优的主模型。

## 处理流程

```
┌─① 数据加载 data_loader ─ CSV 扫描 -> 样本清单(ID/活动/标签)
├─② 预处理     preprocessing ─ data列 -> 复CSI -> 幅度 -> 去离群点(Hampel)
├─③ 特征工程   features      ─ 滑窗(1.54s/0.39s步进) -> 时域统计/差分能量/
│                              FFT频带/谱熵/自相关/形态矩(每窗 66 维)
├─④ 建模       models        ─ RandomForest(主) + LogisticRegression(基线)
├─⑤ 评估       run_train     ─ 按人员 hold-out；窗级 & 文件多数投票级指标
└─⑥ 可视化     visualize     ─ CSI热力图/混淆矩阵/ROC/特征重要性/概率时间线
```

关键点：

- **人员级划分**：测试用户(默认 ID8/ID9)完全不出现在训练中，指标反映跨人泛化。
- **用户均衡**：每用户每动作文件数截断至全用户中位数，防止单用户
  (如 ID2 的 20 倍轮次) 主导训练分布。
- **滑窗增广**：1 个 ~3s 的样本切成若干重叠窗，训练/推理粒度一致。
- **特征设计**：每组特征同时在「原始幅度」与「子载波中位数归一化的相对幅度」上
  计算，既保留环境绝对强度，也提升跨人稳定性；共 58 维 (29×2)，包含反映跌倒的
  谱熵、突发高频能量、差分幅度峭度、自相关周期性等描述子。
- **阈值选择**：在训练人员中再留出验证人员，按 Youden-J 准则选取，避免过拟合。
- **推理复用**：`window_feature_vector()` 为训练与流式推理共用，保证特征一致。

## 输出

`outputs/` 目录：模型(joblib)、`metrics.json`（准确率/精确率/召回率/F1/AUC/阈值）、
以及图 `cm_window_rf.png`、`cm_file_rf.png`、`roc_rf.png`、
`feature_importance_rf.png`、`demo_csi_*.png`、`demo_timeline_*.png`、
`realtime_timeline.png` 等。

## 工程结构

```
WiFall/
├── wifalldemo/           核心库
│   ├── config.py         配置
│   ├── data_loader.py    数据集扫描与人员划分
│   ├── preprocessing.py  CSI 解析与清洗
│   ├── features.py       滑窗与时/频域特征
│   ├── dataset.py        特征数据集 + npz 缓存
│   ├── models.py         分类器训练/评估/阈值
│   └── visualize.py      matplotlib 出图
├── run_train.py          端到端训练与评估
├── run_demo_offline.py   离线检测演示
├── run_demo_realtime.py  实时(流式)检测演示
├── requirements.txt
└── outputs/              模型/指标/图片输出
```

## 真实设备接入提示

`run_demo_realtime.py` 的 `stream_rows()` 已模拟"逐包到达"。接入真实
ESP32 CSI 采集端时，仅需将该生成器替换为 socket/串口回调，并在每个 CSI 包上
调用同一套 `parse_data_field -> window_feature_vector -> predict_proba_pos`，
即可获得端到端实时跌倒报警。
