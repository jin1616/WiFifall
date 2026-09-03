"""特征工程：滑窗 + 时/频域特征提取。

针对跌倒检测，我们关注三类物理线索:
  1. 幅值波动统计      —— 动作越剧烈，CSI 幅度时域方差/差分能量越大
  2. 频域能量分布      —— 静坐几乎无动态能量; 走路有 ~1-2Hz 节律;
                          跌倒包含突发宽带冲激
  3. 波形形态          —— 偏度/峭度/自相关(周期性) 区分节奏型(走/跳)与
                          突发型(跌倒)

为避免不同人员/环境静默基线不一致，特征在两类输入上各算一遍:
  * raw  : 原始幅度      (保留绝对强度, 对同一环境有效)
  * rel  : 幅度/子载波中位数归一化 (突出"相对变化", 跨人更鲁棒)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import N_SUBCARRIERS, WINDOW_SIZE, WINDOW_STEP
from .preprocessing import load_amplitude_series


def _moments(x: np.ndarray) -> tuple[float, float]:
    """x 的 (偏度, 超额峭度) —— 用中心矩计算，不依赖 scipy。"""
    n = x.size
    if n < 4:
        return 0.0, 0.0
    mu = x.mean()
    m2 = np.mean((x - mu) ** 2)
    if m2 < 1e-12:
        return 0.0, 0.0
    m3 = np.mean((x - mu) ** 3)
    m4 = np.mean((x - mu) ** 4)
    skew = m3 / (m2 ** 1.5)
    kurt = m4 / (m2 * m2) - 3.0
    return float(skew), float(kurt)


def _autocorr(x: np.ndarray, max_lag: int = 3) -> np.ndarray:
    """归一化自相关 (lag=1..max_lag)，周期性动作(行走/跳跃)会偏高。"""
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom < 1e-9:
        return np.zeros(max_lag, dtype=np.float32)
    out = np.empty(max_lag, dtype=np.float32)
    for k in range(1, max_lag + 1):
        out[k - 1] = float(np.dot(x[:-k], x[k:])) / denom
    return out


def _spectral_features(A: np.ndarray, fs: float, eps: float = 1e-10):
    """基于幅度矩阵时间维 FFT 的频带能量描述子。

    A: (T, S) 已去均值(逐子载波)。返回多个标量特征的元组。
    """
    T, S = A.shape
    win = np.hanning(T)[:, None]
    spec = np.fft.rfft(A * win, axis=0)          # (F, S)
    psd = (np.abs(spec) ** 2).mean(axis=1)        # 各子载波功率谱平均 -> (F,)
    freq = np.fft.rfftfreq(T, d=1.0 / fs)         # (F,)
    total = float(psd.sum())
    if total < eps:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    p = psd / total

    # 去除 DC/极低频后取剩余能量归一化，便于不同窗间可比
    f_mask = freq > 0.2
    pn = np.where(f_mask, p, 0.0)
    pn_sum = float(pn.sum())
    if pn_sum < eps:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    pn = pn / pn_sum

    centroid = float(np.sum(freq * pn))
    spread = float(np.sqrt(np.sum(((freq - centroid) ** 2) * pn)))
    dom_bin = int(np.argmax(pn))
    dom_freq = float(freq[dom_bin])
    dom_ratio = float(pn[dom_bin])

    # 人体运动相关频带占比 (Hz)
    def band_energy(lo: float, hi: float) -> float:
        m = (freq >= lo) & (freq < hi)
        return float(pn[m].sum())

    b_pulse = band_energy(0.5, 2.0)     # 节奏运动主带 (走/跳)
    b_mid = band_energy(2.0, 5.0)
    b_burst = band_energy(5.0, 12.0)    # 突发高频 (跌倒/剧烈抖动)
    b_high = band_energy(12.0, fs / 2)

    # 谱熵: 白噪/复杂运动高, 平稳节律低
    ent = float(-np.sum(pn * np.log(pn + eps)) / np.log(len(pn)))
    return centroid, spread, dom_freq, dom_ratio, b_pulse, b_mid, b_burst, b_high, ent, 1.0 / max(freq[dom_bin], 0.1)


def _feature_vector(A: np.ndarray, fs: float, prefix: str):
    """对一个 (T, S) 幅度窗计算一组基础特征，返回 (特征列表, 特征名列表)。"""
    T, S = A.shape
    names, vals = [], []

    if T < 4:
        raise ValueError("窗口过短，无法提取特征")

    # ---- 时域统计 ---------------------------------------------------------
    mu_s = A.mean(axis=0)                       # (S,) 每个子载波的时间均值
    sd_s = A.std(axis=0)                        # (S,) 每个子载波的时间波动
    names += [f"{prefix}_t_mean", f"{prefix}_t_std", f"{prefix}_t_range"]
    vals += [float(mu_s.mean()), float(sd_s.mean()), float(mu_s.max() - mu_s.min())]
    names += [f"{prefix}_sd_mean", f"{prefix}_sd_std", f"{prefix}_sd_p90"]
    vals += [float(sd_s.mean()), float(sd_s.std()),
             float(np.percentile(sd_s, 90))]

    names += [f"{prefix}_amp_max", f"{prefix}_amp_min"]
    vals += [float(A.max()), float(A.min())]

    # 逐子载波去均值后的整体残差能量 (波动总量)
    resid = A - mu_s[None, :]
    names += [f"{prefix}_resid_rms", f"{prefix}_resid_std"]
    vals += [float(np.sqrt(np.mean(resid ** 2))), float(resid.std())]

    # ---- 时间差分 (运动瞬时速度代理) ---------------------------------------
    dA = np.diff(A, axis=0)
    ad = np.abs(dA)
    names += [f"{prefix}_diff_mean", f"{prefix}_diff_std",
              f"{prefix}_diff_p95", f"{prefix}_diff_energy"]
    vals += [float(ad.mean()), float(ad.std()),
             float(np.percentile(ad, 95)), float(np.mean(dA ** 2))]

    # 形态: 差分幅度绝对值序列的偏度/峭度
    diff_seq = ad.mean(axis=1)                  # 每个时刻的跨子载波平均变化
    skew, kurt = _moments(diff_seq)
    names += [f"{prefix}_diff_skew", f"{prefix}_diff_kurt"]
    vals += [skew, kurt]

    # 自相关(均值序列的周期性)
    ac = _autocorr(A.mean(axis=1), max_lag=3)
    names += [f"{prefix}_ac1", f"{prefix}_ac2", f"{prefix}_ac3"]
    vals += [float(ac[0]), float(ac[1]), float(ac[2])]

    # ---- 频域特征 -----------------------------------------------------------
    cent = _spectral_features(resid, fs)
    names += [f"{prefix}_f_centroid", f"{prefix}_f_spread", f"{prefix}_f_dom",
              f"{prefix}_f_domratio", f"{prefix}_f_pulse", f"{prefix}_f_mid",
              f"{prefix}_f_burst", f"{prefix}_f_high", f"{prefix}_f_entropy",
              f"{prefix}_f_period"]
    vals += list(cent)

    return vals, names


# 每个活动文件的可重复特征名
def feature_names() -> list[str]:
    """返回最终特征顺序。raw/rel 两组各 29 个（合计 58）。"""
    dummy = np.zeros((WINDOW_SIZE, N_SUBCARRIERS), dtype=np.float32) + 10.0
    _, n1 = _feature_vector(dummy + np.random.randn(*dummy.shape) * 0.01, 83.0, "raw")
    _, n2 = _feature_vector(dummy + np.random.randn(*dummy.shape) * 0.01, 83.0, "rel")
    return n1 + n2


def extract_window_features(
    amp: np.ndarray, fs: float = 83.0, win: int = WINDOW_SIZE, step: int = WINDOW_STEP
):
    """把单个样本的幅度矩阵切窗并逐窗提取特征。

    返回:
        feat (W, F)  特征矩阵
        names [F]    特征名
        rels [W]     每窗起点索引
    """
    T, S = amp.shape
    if T < win:
        raise ValueError(f"样本包数 {T} 小于滑窗 {win}，请调小 WINDOW_SIZE")
    # 相对幅度: 用整个样本的逐子载波中位数归一化
    base = np.median(amp, axis=0, keepdims=True) + 1e-6
    rel_amp = amp / base

    rows = []
    starts = list(range(0, T - win + 1, step))
    names = []
    for st in starts:
        seg_raw = amp[st:st + win]
        seg_rel = rel_amp[st:st + win]
        v1, n1 = _feature_vector(seg_raw, fs, "raw")
        v2, n2 = _feature_vector(seg_rel, fs, "rel")
        rows.append(v1 + v2)
        if not names:
            names = n1 + n2
    return np.asarray(rows, dtype=np.float32), names, np.asarray(starts)


def windows_from_file(
    path: str | Path,
    fs: float = 83.0,
    win: int = WINDOW_SIZE,
    step: int = WINDOW_STEP,
):
    """便捷封装: 读文件 -> 预处理 -> 切窗特征。"""
    amp = load_amplitude_series(path)
    return extract_window_features(amp, fs=fs, win=win, step=step)


def window_feature_vector(
    amp_win: np.ndarray,
    baseline_s: np.ndarray | None = None,
    fs: float = 83.0,
):
    """对单个幅度窗计算特征（训练/推理共用同一套特征描述子）。

    参数:
        amp_win:   (W, S) 幅度窗
        baseline_s:(S,)  每子载波参考基线，用于相对幅度归一化;
                     None 时用本窗中位数近似（流式降级策略）
    返回长度为 2*33 的特征向量。
    """
    if baseline_s is None:
        baseline_s = np.median(amp_win, axis=0)
    v1, n1 = _feature_vector(amp_win, fs, "raw")
    rel = amp_win / (baseline_s[None, :] + 1e-6)
    v2, n2 = _feature_vector(rel, fs, "rel")
    return np.asarray(v1 + v2, dtype=np.float32), n1 + n2
