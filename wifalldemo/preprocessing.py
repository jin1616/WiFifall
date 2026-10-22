"""CSI 预处理：把单个 CSV 样本文件解析为干净的复/幅值 CSI 矩阵。

一个 CSV 文件的结构（Widar3.0 单链路 CSI 流的通用形态）:
    type, seq, timestamp, target, mac, rssi, ..., local_timestamp, ..., data
    CSI_DATA, 26816, ..., fall, ..., 334538338, ..., "[-9, -20, -5, ...]"

其中 data 列为 104 个浮点，按 52 个子载波 [实部, 虚部, 实部, 虚部, ...]
交织编码。本模块负责:
  1. 解析 data 字段 -> 复数 CSI 矩阵 (T, 52)
  2. 计算幅度谱 |CSI| (T, 52)
  3. 去异常 / 平滑（面向后续特征提取）

并向上层提供**统一接口** ``load_sample_amplitude`` / ``iter_sample_rows``，
支持两种样本引用：CSV 路径或 ``<pkl>::<idx>`` 编码（pickle 数据源）。
"""
from __future__ import annotations

import csv
import pickle
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import DATA_DIR, DATA_LEN_PER_ROW, N_SUBCARRIERS, SAMPLING_RATE_HZ


def parse_data_field(text: str) -> np.ndarray | None:
    """把 data 字段字符串解析为复数 CSI 向量，长度应为 52。

    返回 shape=(52,) 的 complex64；解析失败或长度不符时返回 None。
    """
    if not text:
        return None
    text = text.strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    parts = text.split(",")
    if len(parts) != DATA_LEN_PER_ROW:
        return None
    try:
        vals = np.asarray([float(x.strip()) for x in parts], dtype=np.float32)
    except ValueError:
        return None
    real = vals[0::2]
    imag = vals[1::2]
    return real.astype(np.float32) + 1j * imag.astype(np.float32)


def read_csi_csv(path: str | Path) -> np.ndarray:
    """读取单个 CSI 记录文件 -> 复数矩阵 (T, N_SUBCARRIERS)。"""
    path = Path(path)
    packets = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("type") != "CSI_DATA":
                continue
            csi = parse_data_field(row.get("data", ""))
            if csi is not None:
                packets.append(csi)
    if not packets:
        raise ValueError(f"文件 {path} 中没有解析到任何 CSI 包")
    return np.stack(packets, axis=0)  # (T, S)


def complex_to_amplitude(csi: np.ndarray) -> np.ndarray:
    """复 CSI -> 幅度谱；clip 下限防后续对数/除法数值不稳。"""
    amp = np.abs(csi)
    # 幅度接近 0 会导致后续对数/除法数值不稳，统一 clip 到小正数
    amp = np.clip(amp, 1e-3, None)
    return amp.astype(np.float32)


def clean_amplitude(amp: np.ndarray) -> np.ndarray:
    """轻度去噪：移除每列(子载波)时域上的**硬件毛刺级**离群值并线性插值。

    阈值取 ``med + 4·MAD``（与 ``|col - med|`` 比较）——比标准 Hampel
    （``4·1.4826·MAD``）宽松一个 med 的量级，这是**故意的**：跌倒冲击峰值
    可达中位数的 2~3 倍，标准 Hampel 会把这些真正的信号当毛刺插值掉。
    现在的阈值只抓远超动作幅度的硬件尖刺（≳2.3× 中位数），动作波形原样保留。
    """
    amp = amp.copy()
    T, S = amp.shape
    if T < 3:
        return amp
    for s in range(S):
        col = amp[:, s]
        med = np.median(col)
        mad = np.median(np.abs(col - med)) * 1.4826
        if mad < 1e-3:  # 该子载波近乎常数，无需清洗
            continue
        thr = med + 4.0 * mad
        bad = np.abs(col - med) > thr
        if not bad.any():
            continue
        good_idx = np.where(~bad)[0]
        col[bad] = np.interp(np.where(bad)[0], good_idx, col[good_idx])
        amp[:, s] = col
    return amp


def load_amplitude_series(path: str | Path) -> np.ndarray:
    """便捷接口：直接返回某个样本的干净幅度矩阵 (T, S)。"""
    csi = read_csi_csv(path)
    amp = complex_to_amplitude(csi)
    return clean_amplitude(amp)


def normalize_per_subcarrier(
    amp: np.ndarray,
    fs: float = SAMPLING_RATE_HZ,
    window_s: float = 1.5,
    causal: bool = False,
    eps: float = 1e-6,
) -> np.ndarray:
    """逐子载波归一化：除以**滑动窗内的逐子载波中位数**，去掉静态 |H(f)| 形状。

    ``amp_norm[t, s] = amp[t, s] / median(amp[t0:t1, s])``，其中 ``[t0, t1]`` 是
    长度为 ``window_s`` 秒的窗（默认以 ``t`` 为中心；``causal=True`` 时取 ``t`` 之前）。

    为什么它对域差距有用
    --------------------
    实测的 CSI 幅度 = ``|H_static(f)| × |人体扰动(f,t)|``。前者是房间/摆位决定的
    静态形状（用"频率选择性 max/min"度量），后者才是跌倒信号。主模型 CNN1D 吃裸
    幅度时，这个静态形状直接进网络 —— 自家板子选择性 2.5~2.9×、Widar 5.6×，等于
    每个人的 H(f) 都不一样，模型只能记住"这是谁"。

    对**乘性**信道是数学上精确不变的：``amp → amp · g(s)`` 时参考量同样乘 ``g(s)``，
    比值分毫不变（实测 0.0% 变化）。这正是选"逐子载波"而不是全局归一化的原因。

    参数
    ----
    amp : (T, S) 幅度矩阵
    fs : 采样率(Hz)，用于把 ``window_s`` 换算成帧数
    window_s : 参考窗长度(秒)，1~2 s 为宜
    causal : True = 只用当前时刻**之前**的帧做参考（流式推理用）；
             False = 用居中窗（离线训练/评估用，质量略好）
    eps : 防止除以 0

    返回
    ----
    (T, S) float32，基线在 1.0 附近
    """
    amp = np.asarray(amp, dtype=np.float32)
    if amp.ndim != 2:
        raise ValueError(f"期望 (T, S) 二维矩阵，实际 {amp.shape}")
    T, S = amp.shape
    if T == 0:
        return amp

    win = max(1, int(round(window_s * fs)))
    if win >= T:
        # 样本比参考窗还短：退化成"整条的中位数"，仍然去掉静态形状
        ref = np.median(amp, axis=0, keepdims=True)
        return (amp / np.maximum(ref, eps)).astype(np.float32)

    # 居中窗需要两端各补 win//2；因果窗只补开头（保证参考量只用历史帧）
    if causal:
        pad_head, pad_tail = win, 0
    else:
        pad_head, pad_tail = win // 2, win - 1 - win // 2
    padded = np.pad(amp, ((pad_head, pad_tail), (0, 0)), mode="edge")

    # (T, S, win) -> 沿最后一维取中位数 -> (T, S)
    ref = np.median(
        np.lib.stride_tricks.sliding_window_view(padded, win, axis=0)[:T], axis=-1
    )
    return (amp / np.maximum(ref, eps)).astype(np.float32)


def estimate_sampling_rate(path: str | Path) -> float:
    """用 local_timestamp(us) 的中位间隔估计真实 CSI 包速率(Hz)。

    若解析失败返回默认配置值 83Hz。
    """
    path = Path(path)
    deltas = []
    prev = None
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for row in csv.DictReader(f):
            if row.get("type") != "CSI_DATA":
                continue
            try:
                ts = int(row["local_timestamp"])
            except (KeyError, ValueError, TypeError):
                continue
            if prev is not None:
                d = ts - prev
                if d > 0:
                    deltas.append(d)
            prev = ts
    if len(deltas) < 5:
        return SAMPLING_RATE_HZ
    deltas = np.asarray(deltas)
    return float(1e6 / np.median(deltas))


# ---------------------------------------------------------------------------
# 统一样本接口：同时支持 CSV 路径与 pickle 引用
# ---------------------------------------------------------------------------
# pickle 路径编码：``<pkl_abs>::<idx>``。rpartition 保证只在最后一个 ``::``
# 切分；双冒号不会出现在合法文件系统路径里，因此与绝对路径无歧义。
PICKLE_DELIM = "::"
# padding 行幅度/相位的标记：NaN。NaN 穿过 np.abs/np.angle 仍为 NaN，因此用
# ``np.isnan`` 即可稳定识别；相比 -1000 不会被误当成"大值 CSI 包"。
_PAD_NAN_THRESHOLD = "nan"  # 仅作文档标识，逻辑统一用 np.isnan


@lru_cache(maxsize=4)
def _load_pickle_seq(pkl_abs: str) -> list:
    """读 data_sequence.pkl 一次并缓存；整个训练流程只读盘一次。"""
    with Path(pkl_abs).open("rb") as f:
        seq = pickle.load(f)
    if not isinstance(seq, list):
        raise ValueError(f"{pkl_abs} 内容不是 list，无法解析")
    return seq


def _parse_pickle_ref(ref: str) -> tuple[str, int]:
    """``<pkl>::<idx>`` -> (绝对 pkl 路径, 样本下标)；解析失败抛 ValueError。"""
    if PICKLE_DELIM not in ref:
        raise ValueError(f"不是合法的 pickle 样本引用 (缺少 '{PICKLE_DELIM}'): {ref}")
    pkl_path, _, idx_str = ref.rpartition(PICKLE_DELIM)
    if not idx_str.isdigit():
        raise ValueError(f"pickle 样本引用末尾应为整数 idx: {ref}")
    return pkl_path, int(idx_str)


def _load_amplitude_from_pickle(pkl_abs: str, idx: int) -> np.ndarray:
    """从 data_sequence.pkl 取第 idx 个样本的幅度矩阵，剔除 padding 行。

    与 CSV 路径一致：剔除 padding -> clip(1e-3, inf) -> Hampel 清洗，
    返回 ``(T, N_SUBCARRIERS)`` 的 ``float32`` 矩阵。
    """
    seq = _load_pickle_seq(pkl_abs)
    if not (0 <= idx < len(seq)):
        raise IndexError(f"pickle 样本 idx 越界: {idx} / {len(seq)}")
    magnitude = seq[idx]["magnitude"].astype(np.float32, copy=False)
    # 剔除 run_build_sequence.PAD_VALUE(NaN) 行：任意子载波为 NaN 即视为 padding
    keep = ~np.isnan(magnitude).any(axis=1)
    if not keep.all():
        magnitude = magnitude[keep]
    if magnitude.size == 0:
        raise ValueError(f"pickle 样本 #{idx} 在剔除 padding 后为空")
    # 与 load_amplitude_series 保持一致：clip 防 log/除法数值不稳 + Hampel 清洗
    magnitude = np.clip(magnitude, 1e-3, None)
    return clean_amplitude(magnitude)


def load_sample_amplitude(path: str | Path) -> np.ndarray:
    """统一接口：返回某个样本的干净幅度矩阵 ``(T, N_SUBCARRIERS)``。

    ``path`` 可以是:
      * ``*.csv`` 实际文件路径 -> 走 ``load_amplitude_series``
      * ``<pkl_abs>::<idx>`` pickle 引用 -> 走 ``_load_amplitude_from_pickle``

    二者返回的幅度矩阵在数值语义上等价（同样的 clip + Hampel），可被
    ``features.extract_window_features`` 直接消费。
    """
    s = str(path)
    if PICKLE_DELIM in s:
        pkl_abs, idx = _parse_pickle_ref(s)
        return _load_amplitude_from_pickle(pkl_abs, idx)
    return load_amplitude_series(s)


def iter_sample_rows(path: str | Path) -> Iterator[tuple[float, np.ndarray]]:
    """模拟流式采集源：逐包/逐行产出 ``(t_seconds, amplitude_vec)``。

    实际部署时把生成器替换为 socket/串口回调即可接入真实设备，
    上层流式推理循环（``run_demo_realtime.py``）保持不变。
    """
    s = str(path)
    if PICKLE_DELIM in s:
        pkl_abs, idx = _parse_pickle_ref(s)
        seq = _load_pickle_seq(pkl_abs)
        if not (0 <= idx < len(seq)):
            raise IndexError(f"pickle 样本 idx 越界: {idx} / {len(seq)}")
        item = seq[idx]
        time_us = np.asarray(item["time"], dtype=np.float64)  # 微秒
        magnitude = item["magnitude"].astype(np.float32, copy=False)
        # 流式回放不应把 padding 当作真实 CSI 包：遇到 padding 行直接跳过
        keep = ~np.isnan(magnitude).any(axis=1)
        time_us = time_us[keep]
        magnitude = magnitude[keep]
        t0 = float(time_us[0]) if time_us.size else 0.0
        for t_us, vec in zip(time_us, magnitude):
            yield (float(t_us) - t0) / 1e6, vec.astype(np.float32, copy=False)
        return

    # CSV 路径：与原 run_demo_realtime.stream_rows 等价
    p = Path(s)
    t0 = None
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        rd = csv.DictReader(f)
        for row in rd:
            if row.get("type") != "CSI_DATA":
                continue
            csi = parse_data_field(row.get("data", ""))
            if csi is None:
                continue
            ts_us = row.get("local_timestamp")
            t = 0.0
            if ts_us is not None and str(ts_us).isdigit():
                ts = int(ts_us)
                t0 = ts if t0 is None else t0
                t = (ts - t0) / 1e6
            yield t, complex_to_amplitude(csi)
