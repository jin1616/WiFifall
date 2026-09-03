"""CSI 预处理：把单个 CSV 样本文件解析为干净的复/幅值 CSI 矩阵。

一个 CSV 文件的结构（Widar3.0 单链路 CSI 流的通用形态）:
    type, seq, timestamp, target, mac, rssi, ..., local_timestamp, ..., data
    CSI_DATA, 26816, ..., fall, ..., 334538338, ..., "[-9, -20, -5, ...]"

其中 data 列为 104 个浮点，按 52 个子载波 [实部, 虚部, 实部, 虚部, ...]
交织编码。本模块负责:
  1. 解析 data 字段 -> 复数 CSI 矩阵 (T, 52)
  2. 计算幅度谱 |CSI| (T, 52)
  3. 去异常 / 平滑（面向后续特征提取）
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .config import DATA_LEN_PER_ROW, N_SUBCARRIERS, SAMPLING_RATE_HZ


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
    """复 CSI -> 幅度谱 (log1p 压缩以抑制极端尖峰)。"""
    amp = np.abs(csi)
    # 幅度接近 0 会导致后续对数/除法数值不稳，统一 clip 到小正数
    amp = np.clip(amp, 1e-3, None)
    return amp.astype(np.float32)


def clean_amplitude(amp: np.ndarray) -> np.ndarray:
    """轻度去噪：移除每列(子载波)时域上的离群值并线性插值。

    使用 Hampel 思路的简化版：以 (中位数, MAD) 为基准，超出阈值的位置
    用相邻正常值线性插值替换，避免硬件毛刺污染特征。
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
