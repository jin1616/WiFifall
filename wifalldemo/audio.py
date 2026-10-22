"""音频通路：把采集端的 ``<sample>.wav`` 对齐到 CSI 时间轴，并提取窗级声学特征。

为什么需要这个模块
------------------
采集脚本（``csi_audio_sync.py``，已随采集包迁到 ``../ESP32-S3``）每条样本会产出
四件套::

    fall_001.csv           CSI（训练主通路）
    fall_001.wav           麦克风录音 16 kHz / 16 bit
    fall_001.sync.json     两个设备的时间轴映射 + 对齐质量
    fall_001_audio_env.csv 逐 CSI 行的声音能量（旁证）

CSI 只能看到"人体运动改变了多径"，对"落地那一刻的撞击声"是间接的；而声音对
撞击是直接的。把两者拼起来，模型既有运动学证据、又有撞击证据，这正是本模块
存在的理由。

对齐怎么做的
------------
两台设备时钟不同源，但 ``sync.json`` 已经把映射解出来了：

* ``csi_start_pc``  —— CSI 第 0 行对应的 PC 墙钟时刻（秒）
* ``csi_fs``        —— CSI 行率（83 Hz，采集端已重采样到均匀网格）
* ``audio_start_pc``—— **音频第 0 个采样点**对应的 PC 墙钟时刻
* ``lag_applied_ms``—— 采集端用互相关做过的校正量（已烘进上面两条时间轴）

于是第 ``i`` 个 CSI 窗（覆盖第 ``start..start+win`` 行）对应的音频采样区间是::

    s0 = (csi_start_pc + start / csi_fs - audio_start_pc) * audio_fs
    s1 = (csi_start_pc + (start+win) / csi_fs - audio_start_pc) * audio_fs

在 404 条自采样本上用 ``_audio_env.csv`` 逐行 RMS 交叉验证过：**相关系数最小值
1.0000**（含 4 条 ``lag_applied_ms != 0`` 的样本），说明这个映射在全部数据上成立。

特征怎么算
----------
先在整条音频上做一次逐帧分析（25 ms 帧 / 10 ms 跳，Hann 窗 + rFFT），得到逐帧的
电平、起振、频谱形状、四个频带占比；再按窗取帧、聚合成 **31 维**定长向量
（:data:`AUDIO_FEATURE_NAMES`）。一次分析整条音频、多窗复用，避免每个窗都重算 STFT。

设计取舍
--------
* **纯 numpy + 标准库 wave**：不引入 librosa / scipy，采集现场的轻量环境也能跑。
* **固定维度 + 存在标志**：没有音频的样本（例如 Widar 数据集、或早期未录声音的
  自采样本）返回全零 + ``audio_present = 0``，而不是报错或改变维度 —— 融合模型
  因此可以同时在"有声音/没声音"的数据上训练。
* **物理量纲保留**：dBFS、秒、归一化频率（÷ Nyquist），配合首层 BatchNorm 足够稳，
  同时特征名本身可读、便于在报告里解释。
"""
from __future__ import annotations

import json
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------
# 参数
# ----------------------------------------------------------------------------
AUDIO_FRAME_MS = 25.0      # 分析帧长：25 ms（400 点 @16 kHz），兼顾时间/频率分辨率
AUDIO_HOP_MS = 10.0        # 帧移：10 ms，足够定位落地冲击的起振时刻
DB_FLOOR = -90.0           # 数字静音的电平下限，防止 log(0) 与极端离群值
ONSET_DB = 3.0             # 判为"起振"的单帧跳变阈值（10 ms 相邻帧）
BAND_EDGES_HZ = (250.0, 1000.0, 4000.0)  # 低 / 中低 / 中高 / 高 四个频带的分界

# 31 维特征的**固定顺序**。改动必须同步 AUDIO_FEATURE_VERSION，否则旧缓存/旧模型
# 会静默错位 —— 特征顺序错了不会报错，只会让指标莫名其妙地掉。
AUDIO_FEATURE_VERSION = 2   # v2: logrms_mean/max/p90 逐文件居中（减该文件中位电平）
AUDIO_FEATURE_NAMES: tuple[str, ...] = (
    # ---- 包络 / 起振（11）------------------------------------------------
    "audio_logrms_mean",     # 窗内平均电平 dBFS
    "audio_logrms_std",      # 电平波动
    "audio_logrms_max",      # 峰值电平（撞击最强处）
    "audio_logrms_p90",      # 90 分位电平
    "audio_crest_db",        # 峰值 - 均值：冲击性（跌倒"砰"一声 → 高）
    "audio_onset_max_db",    # 相邻帧最大正跳变：起振强度
    "audio_onset_ratio",     # 跳变 > 3 dB 的帧占比：瞬态密度
    "audio_attack_s",        # 从 -10 dB(相对峰值) 升到峰值所用时间
    "audio_decay_s",         # 从峰值衰减 -10 dB 所用时间
    "audio_active_ratio",    # 高于(峰值-20 dB)的帧占比：有声时长
    "audio_peak_pos",        # 峰值帧在窗内的相对位置 0~1
    # ---- 频谱形状（10）---------------------------------------------------
    "audio_centroid_mean",   # 谱质心（÷Nyquist，0~1）
    "audio_centroid_std",
    "audio_bandwidth_mean",  # 谱带宽
    "audio_rolloff85_mean",  # 85% 能量滚降点
    "audio_flatness_mean",   # 谱平坦度：噪声性 vs 谐波性
    "audio_flux_mean",       # 谱通量：频谱变化快慢
    "audio_flux_max",
    "audio_zcr_mean",        # 过零率
    "audio_domfreq_mean",    # 主频（÷Nyquist）
    "audio_entropy_mean",    # 谱熵
    # ---- 四频带能量占比（4）----------------------------------------------
    "audio_band_low",        # < 250 Hz：撞击"闷响"主要落在这里
    "audio_band_lowmid",     # 250 Hz ~ 1 kHz：人声基频
    "audio_band_high",       # 1 ~ 4 kHz
    "audio_band_veryhigh",   # > 4 kHz：摩擦/衣物声
    # ---- 峰值帧局部特征（4）----------------------------------------------
    "audio_impact_centroid", # 峰值帧的谱质心
    "audio_impact_low",      # 峰值帧的低频占比
    "audio_impact_flux",     # 峰值帧的谱通量
    "audio_peak_snr_db",     # 峰值 - 中位数电平：撞击相对本底的信噪比
    # ---- 元信息（2）-------------------------------------------------------
    "audio_present",         # 1=这条样本有可用音频且覆盖 > 50%
    "audio_coverage",        # 窗对应音频区间真正落在录音内的比例 0~1
)
AUDIO_DIM = len(AUDIO_FEATURE_NAMES)
_PRESENT_IDX = AUDIO_FEATURE_NAMES.index("audio_present")


def audio_feature_names() -> list[str]:
    """返回音频特征的顺序（与 :func:`extract_audio_window_features` 的列一一对应）。"""
    return list(AUDIO_FEATURE_NAMES)


def audio_feature_dim() -> int:
    return AUDIO_DIM


# ----------------------------------------------------------------------------
# 读取 wav / sync
# ----------------------------------------------------------------------------
def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """读 16-bit PCM wav -> ``(float32 单声道 [-1,1], fs)``。

    只用标准库 ``wave``：采集端固定输出 16 kHz / 16 bit / 单声道，但这里也接受
    8/16/32 bit 与多声道（取均值），免得换录音设备就崩。
    """
    path = Path(path)
    with wave.open(str(path), "rb") as w:
        fs = w.getframerate()
        n_ch = w.getnchannels()
        sampwidth = w.getsampwidth()
        n_frames = w.getnframes()
        raw = w.readframes(n_frames)
    if not raw:
        return np.zeros(0, dtype=np.float32), fs

    if sampwidth == 1:      # 8-bit 无符号
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sampwidth == 2:    # 16-bit 有符号（采集端默认）
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sampwidth == 4:    # 32-bit 有符号
        x = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"不支持的 wav 采样位宽: {sampwidth} 字节 ({path})")

    if n_ch > 1:
        x = x.reshape(-1, n_ch).mean(axis=1)
    return np.ascontiguousarray(x, dtype=np.float32), int(fs)


def read_sync(path: str | Path) -> dict | None:
    """读 ``<sample>.sync.json``；缺失或损坏时返回 ``None``。"""
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def sidecar_path(sample_path: str | Path, suffix: str) -> Path | None:
    """把样本路径的扩展名换成 ``suffix``（``fall_001.csv`` -> ``fall_001.wav``）。

    pickle 引用（``<pkl>::<idx>``）没有音频旁车文件，直接返回 ``None``。
    """
    s = str(sample_path)
    if "::" in s:
        return None
    return Path(s).with_suffix(suffix)


# ----------------------------------------------------------------------------
# 逐帧声学分析
# ----------------------------------------------------------------------------
@dataclass
class AudioFrames:
    """整条音频的逐帧分析结果（一次计算、多窗复用）。"""

    t_center: np.ndarray     # (F,) 每帧中心时刻（相对音频第 0 个采样，秒）
    db: np.ndarray           # (F,) 帧电平 dBFS
    centroid: np.ndarray     # (F,) 谱质心（÷Nyquist）
    bandwidth: np.ndarray    # (F,) 谱带宽（÷Nyquist）
    rolloff: np.ndarray      # (F,) 85% 滚降点（÷Nyquist）
    flatness: np.ndarray     # (F,) 谱平坦度 0~1
    flux: np.ndarray         # (F,) 谱通量（归一化谱的相邻帧 L2 差）
    zcr: np.ndarray          # (F,) 过零率
    domfreq: np.ndarray      # (F,) 主频（÷Nyquist）
    entropy: np.ndarray      # (F,) 谱熵 0~1
    band_low: np.ndarray     # (F,) <250 Hz 能量占比
    band_lowmid: np.ndarray  # (F,) 250~1000 Hz
    band_high: np.ndarray    # (F,) 1000~4000 Hz
    band_veryhigh: np.ndarray  # (F,) >4000 Hz
    hop_s: float             # 帧移（秒）
    fs: int

    def __len__(self) -> int:
        return int(self.db.size)


def _frame_signal(x: np.ndarray, fs: int, frame_ms: float, hop_ms: float):
    """切成 (F, n) 帧矩阵，返回 ``(frames, hop_s, frame_s)``；过短则补零成一帧。"""
    n = max(1, int(round(frame_ms * fs / 1000.0)))
    h = max(1, int(round(hop_ms * fs / 1000.0)))
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    frames = np.lib.stride_tricks.sliding_window_view(x, n)[::h]
    return np.ascontiguousarray(frames), h / fs, n / fs


def analyse_audio(
    samples: np.ndarray,
    fs: int,
    frame_ms: float = AUDIO_FRAME_MS,
    hop_ms: float = AUDIO_HOP_MS,
) -> AudioFrames:
    """对整条音频做逐帧分析，输出 :class:`AudioFrames`。

    所有频率类特征都除以 Nyquist（0~1），这样换采样率也不会改变量纲；
    电平类特征保留 dBFS，便于解释"撞击有多响"。
    """
    x = np.asarray(samples, dtype=np.float32)
    if x.size == 0:
        return _empty_frames(fs, hop_ms / 1000.0, frame_ms / 1000.0)

    frames, hop_s, _frame_s = _frame_signal(x, fs, frame_ms, hop_ms)
    win = np.hanning(frames.shape[1]).astype(np.float32)
    spec = np.abs(np.fft.rfft(frames * win, axis=1)).astype(np.float32)  # (F, B)
    power = spec ** 2
    freqs = np.fft.rfftfreq(frames.shape[1], d=1.0 / fs).astype(np.float32)
    nyq = float(fs) / 2.0
    total = power.sum(axis=1) + 1e-12
    p = power / total[:, None]

    # ---- 电平 ---------------------------------------------------------------
    rms = np.sqrt(np.mean(frames.astype(np.float32) ** 2, axis=1))
    db = np.clip(20.0 * np.log10(rms + 1e-8), DB_FLOOR, 0.0)

    # ---- 频谱形状 -----------------------------------------------------------
    centroid = (p @ freqs) / nyq
    var = np.maximum((p @ (freqs ** 2)) / nyq ** 2 - centroid ** 2, 0.0)
    bandwidth = np.sqrt(var)
    cum = np.cumsum(p, axis=1)
    rolloff_idx = np.argmax(cum >= 0.85, axis=1)
    rolloff = freqs[rolloff_idx] / nyq
    # 平坦度 = 几何均值 / 算术均值（对数域求几何均值，避免下溢）
    geo = np.exp(np.mean(np.log(power + 1e-12), axis=1))
    ari = np.mean(power, axis=1) + 1e-12
    flatness = np.clip(geo / ari, 0.0, 1.0)
    domfreq = freqs[np.argmax(power, axis=1)] / nyq
    entropy = -np.sum(p * np.log(p + 1e-12), axis=1) / np.log(p.shape[1])
    entropy = np.clip(entropy, 0.0, 1.0)

    # 谱通量：归一化谱逐帧差分的 L2 范数（首帧记 0）
    d = np.diff(p, axis=0)
    flux = np.concatenate([[0.0], np.sqrt(np.sum(d ** 2, axis=1))]).astype(np.float32)

    # ---- 过零率 -------------------------------------------------------------
    sgn = np.signbit(frames)
    zcr = np.mean(sgn[:, 1:] != sgn[:, :-1], axis=1).astype(np.float32)

    # ---- 四频带占比 ---------------------------------------------------------
    e0, e1, e2 = BAND_EDGES_HZ
    edges = [(-np.inf, e0), (e0, e1), (e1, e2), (e2, np.inf)]
    bands = []
    for lo, hi in edges:
        m = (freqs >= lo) & (freqs < hi)
        bands.append(p[:, m].sum(axis=1).astype(np.float32))

    hop_s_eff = hop_s
    t_center = (np.arange(len(db), dtype=np.float32) * hop_s_eff
                + (frames.shape[1] / 2.0) / fs)
    return AudioFrames(
        t_center=t_center.astype(np.float32),
        db=db.astype(np.float32),
        centroid=centroid.astype(np.float32),
        bandwidth=bandwidth.astype(np.float32),
        rolloff=rolloff.astype(np.float32),
        flatness=flatness.astype(np.float32),
        flux=flux,
        zcr=zcr,
        domfreq=domfreq.astype(np.float32),
        entropy=entropy.astype(np.float32),
        band_low=bands[0], band_lowmid=bands[1],
        band_high=bands[2], band_veryhigh=bands[3],
        hop_s=hop_s_eff, fs=int(fs),
    )


def _empty_frames(fs: int, hop_s: float, frame_s: float) -> AudioFrames:
    """空音频的占位帧（长度 1），保证后续聚合不崩。"""
    z = np.zeros(1, dtype=np.float32)
    return AudioFrames(
        t_center=np.zeros(1, dtype=np.float32),
        db=np.full(1, DB_FLOOR, dtype=np.float32),
        centroid=z.copy(), bandwidth=z.copy(), rolloff=z.copy(),
        flatness=z.copy(), flux=z.copy(), zcr=z.copy(), domfreq=z.copy(),
        entropy=z.copy(), band_low=np.ones(1, dtype=np.float32),
        band_lowmid=z.copy(), band_high=z.copy(), band_veryhigh=z.copy(),
        hop_s=hop_s, fs=int(fs),
    )


# ----------------------------------------------------------------------------
# 时间轴
# ----------------------------------------------------------------------------
@dataclass
class AudioTrack:
    """一条样本的音频 + 它到 CSI 时间轴的映射。"""

    samples: np.ndarray
    fs: int
    csi_start_pc: float
    audio_start_pc: float
    csi_fs: float
    path: Path
    sync: dict = field(default_factory=dict, repr=False)
    _frames: AudioFrames | None = field(default=None, repr=False)

    # ---- 基础量 -----------------------------------------------------------
    @property
    def duration(self) -> float:
        return float(len(self.samples)) / float(self.fs)

    def frames(self) -> AudioFrames:
        if self._frames is None:
            self._frames = analyse_audio(self.samples, self.fs)
        return self._frames

    # ---- 行号 -> 采样点 ---------------------------------------------------
    def sample_span(self, start_row: int, n_rows: int) -> tuple[int, int]:
        """CSI 行区间 ``[start_row, start_row+n_rows)`` 对应的音频采样区间。"""
        t0 = self.csi_start_pc + float(start_row) / self.csi_fs
        t1 = self.csi_start_pc + float(start_row + n_rows) / self.csi_fs
        s0 = int(round((t0 - self.audio_start_pc) * self.fs))
        s1 = int(round((t1 - self.audio_start_pc) * self.fs))
        return s0, s1

    def coverage(self, s0: int, s1: int) -> float:
        """音频采样区间真正落在录音内的比例（录音比 CSI 短时 < 1）。"""
        span = max(s1 - s0, 1)
        covered = min(s1, len(self.samples)) - max(s0, 0)
        return float(np.clip(covered / span, 0.0, 1.0))

    # ---- 行区间 -> 特征 ---------------------------------------------------
    def features_for_rows(self, start_row: int, n_rows: int) -> np.ndarray:
        s0, s1 = self.sample_span(start_row, n_rows)
        cov = self.coverage(s0, s1)
        return self.features_for_samples(s0, s1, cov)

    def features_for_samples(self, s0: int, s1: int, coverage: float | None = None) -> np.ndarray:
        """音频采样区间 -> 31 维特征。区间与录音无交集时退化成"无音频"。"""
        cov = self.coverage(s0, s1) if coverage is None else float(coverage)
        fr = self.frames()
        t0 = s0 / float(self.fs)
        t1 = s1 / float(self.fs)
        idx = np.where((fr.t_center >= t0) & (fr.t_center < t1))[0]
        if idx.size == 0:
            return _missing_audio_row(coverage=cov)
        return _aggregate_frames(fr, int(idx[0]), int(idx[-1]) + 1, cov)


def read_audio_track(sample_path: str | Path) -> AudioTrack | None:
    """读取一条 CSI 样本对应的音频轨道；没有 wav / sync 时返回 ``None``。

    缺任何一件都算"没有音频"：只有 wav 没有 sync 无法对齐（凭空猜偏移会让特征
    整体错位，比不用更糟）；只有 sync 没有 wav 自然也无从谈起。
    """
    wav_p = sidecar_path(sample_path, ".wav")
    sync_p = sidecar_path(sample_path, ".sync.json")
    if wav_p is None or sync_p is None or not wav_p.exists():
        return None
    sync = read_sync(sync_p)
    if not sync:
        return None
    try:
        csi_start = float(sync["csi_start_pc"])
        audio_start = float(sync["audio_start_pc"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (np.isfinite(csi_start) and np.isfinite(audio_start)):
        return None
    csi_fs = float(sync.get("csi_fs", 83.0) or 83.0)
    if not np.isfinite(csi_fs) or csi_fs <= 0:
        csi_fs = 83.0
    try:
        samples, fs = read_wav(wav_p)
    except (wave.Error, OSError, ValueError):
        return None
    if samples.size == 0 or fs <= 0:
        return None
    return AudioTrack(
        samples=samples, fs=fs, csi_start_pc=csi_start,
        audio_start_pc=audio_start, csi_fs=csi_fs, path=wav_p, sync=sync,
    )


# ----------------------------------------------------------------------------
# 窗级聚合
# ----------------------------------------------------------------------------
def _aggregate_frames(fr: AudioFrames, a: int, b: int, coverage: float) -> np.ndarray:
    """把 ``[a, b)`` 的逐帧结果聚合成一个 31 维向量。"""
    a = max(0, a)
    b = min(len(fr), b)
    if b <= a:
        return _missing_audio_row(coverage=coverage)

    db = fr.db[a:b].astype(np.float64)
    n = db.size
    peak = int(np.argmax(db))
    mean_db = float(db.mean())
    max_db = float(db.max())
    std_db = float(db.std())
    p90_db = float(np.percentile(db, 90))
    crest = max_db - mean_db

    if n >= 2:
        d = np.diff(db)
        onset_max = float(max(0.0, float(d.max())))
        onset_ratio = float(np.mean(d > ONSET_DB))
    else:
        onset_max, onset_ratio = 0.0, 0.0

    hop = float(fr.hop_s)
    # 起振时间：峰值往前第一次落到 (峰值-10 dB) 之后到峰值经历的时间
    above = np.where(db[: peak + 1] >= max_db - 10.0)[0]
    attack_s = float((peak - int(above[0])) * hop) if above.size else 0.0
    # 衰减时间：峰值往后一直保持在 (峰值-10 dB) 以上的时长
    post = np.where(db[peak:] >= max_db - 10.0)[0]
    decay_s = float(int(post[-1]) * hop) if post.size else 0.0
    active_ratio = float(np.mean(db >= max_db - 20.0))
    peak_pos = float(peak / max(n - 1, 1))
    peak_snr = max_db - float(np.median(db))

    ce = fr.centroid[a:b]
    bw = fr.bandwidth[a:b]
    ro = fr.rolloff[a:b]
    fl = fr.flatness[a:b]
    fx = fr.flux[a:b]
    zc = fr.zcr[a:b]
    df = fr.domfreq[a:b]
    en = fr.entropy[a:b]

    present = 1.0 if coverage >= 0.5 else 0.0
    vals = [
        mean_db, std_db, max_db, p90_db, crest,
        onset_max, onset_ratio, attack_s, decay_s, active_ratio, peak_pos,
        float(ce.mean()), float(ce.std()), float(bw.mean()), float(ro.mean()),
        float(fl.mean()), float(fx.mean()), float(fx.max()),
        float(zc.mean()), float(df.mean()), float(en.mean()),
        float(fr.band_low[a:b].mean()), float(fr.band_lowmid[a:b].mean()),
        float(fr.band_high[a:b].mean()), float(fr.band_veryhigh[a:b].mean()),
        float(fr.centroid[a + peak]), float(fr.band_low[a + peak]),
        float(fr.flux[a + peak]), float(peak_snr),
        present, float(np.clip(coverage, 0.0, 1.0)),
    ]
    return np.nan_to_num(np.asarray(vals, dtype=np.float32), nan=0.0,
                         posinf=0.0, neginf=0.0)


def _missing_audio_row(coverage: float = 0.0) -> np.ndarray:
    """无音频时的占位向量：全零 + ``present=0``（维度与正常样本一致）。"""
    out = np.zeros(AUDIO_DIM, dtype=np.float32)
    out[_PRESENT_IDX] = 0.0
    out[_PRESENT_IDX + 1] = float(np.clip(coverage, 0.0, 1.0))
    # 谱质心/主频等比例特征取 0 有歧义，但 present=0 已经告诉模型"这一行不可信"
    return out


# ----------------------------------------------------------------------------
# 对外主入口
# ----------------------------------------------------------------------------
def extract_audio_window_features(
    sample_path: str | Path,
    starts,
    win: int,
    csi_fs: float = 83.0,
) -> np.ndarray:
    """对一条样本，按 CSI 窗起点 ``starts`` 逐窗提取音频特征。

    参数
    ----
    sample_path : CSI 样本路径（CSV）。pickle 引用没有音频，返回全零 + present=0
    starts      : 各窗的起始**行号**（与 ``extract_window_features`` 的返回值一致）
    win         : 窗长（行数）
    csi_fs      : CSI 行率，默认 83 Hz

    返回
    ----
    (len(starts), 31) float32；没有音频时全部为占位行（不抛异常）
    """
    starts = np.asarray(list(starts), dtype=np.int64)
    n = len(starts)
    if n == 0:
        return np.zeros((0, AUDIO_DIM), dtype=np.float32)

    track = read_audio_track(sample_path)
    if track is None:
        return np.stack([_missing_audio_row(0.0) for _ in range(n)]).astype(np.float32)

    # 一次分析整条音频，多窗复用（STFT 只做一次）
    track.frames()
    out = np.empty((n, AUDIO_DIM), dtype=np.float32)
    for i, st in enumerate(starts):
        out[i] = track.features_for_rows(int(st), int(win))
    return _center_per_file(out)


# 逐文件居中要动的列：绝对电平类（dBFS，随录音增益/麦克风距离整体平移）；
# crest/onset/peak_snr 等"相对量"、频带占比、质心、时长类特征天然增益不变，
# 不动。audio_present/audio_coverage 是元信息，不动。
_LEVEL_IDX = tuple(
    AUDIO_FEATURE_NAMES.index(n)
    for n in ("audio_logrms_mean", "audio_logrms_max", "audio_logrms_p90")
)
_BASE_IDX = AUDIO_FEATURE_NAMES.index("audio_logrms_mean")


def _center_per_file(out: np.ndarray) -> np.ndarray:
    """把绝对电平特征减去**该文件**的稳健中位电平（逐文件归一化）。

    不同采集批次的录音增益/距离不同，绝对 dBFS 不可比：ID15 那批撞击声偏轻，
    在别批数据选出的绝对尺度下整折失效。减去本文件中位电平后，特征表达的是
    "相对本底有多突出"，批次差异被消掉。无音频的占位行（present=0）不动，
    免得占位值被平移后与"极安静"混淆 —— present=0 本身已告诉模型该行不可信。
    """
    present = out[:, _PRESENT_IDX] >= 0.5
    if not present.any():
        return out
    base = float(np.median(out[present, _BASE_IDX]))
    for c in _LEVEL_IDX:
        out[present, c] -= np.float32(base)
    return out



def has_audio(sample_path: str | Path) -> bool:
    """快速判断一条样本是否有 wav + sync（不做解码）。"""
    wav_p = sidecar_path(sample_path, ".wav")
    sync_p = sidecar_path(sample_path, ".sync.json")
    return bool(wav_p and sync_p and wav_p.exists() and sync_p.exists())


__all__ = [
    "AUDIO_DIM",
    "AUDIO_FEATURE_NAMES",
    "AUDIO_FEATURE_VERSION",
    "AudioFrames",
    "AudioTrack",
    "analyse_audio",
    "audio_feature_dim",
    "audio_feature_names",
    "extract_audio_window_features",
    "has_audio",
    "read_audio_track",
    "read_sync",
    "read_wav",
    "sidecar_path",
]
