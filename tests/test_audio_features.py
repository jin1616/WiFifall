#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""音频通路（第 4 路特征）: 锁住对齐、维度、缺失降级三件事。

    python -m tests.test_audio_features

为什么值得测
------------
音频特征最容易出的不是崩溃，而是**静默错位**：wav 和 CSI 差了半秒，特征照样
算得出来、维度也对，只是每一列都在描述错误的时刻，训练指标莫名其妙地掉。
所以这里的核心用例是"在已知时刻放一声咔哒，看它是否落在正确的窗里"。

覆盖:
  1. wav 读写与电平计算
  2. sync.json 时间轴 -> 窗特征的定位精度（合成信号，真值已知）
  3. 缺 wav / 缺 sync / 坏文件时退化成占位行（present=0，维度不变）
  4. 特征定长、有限、可复现
  5. 与 dataset 拼接后 CSI + 31 维音频，且 CSI 部分逐位不变
"""
from __future__ import annotations

import json
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from wifalldemo.audio import (  # noqa: E402
    AUDIO_DIM,
    AUDIO_FEATURE_NAMES,
    extract_audio_window_features,
    read_audio_track,
    read_wav,
)
from wifalldemo.dataset import features_for_sample  # noqa: E402

FS_AUDIO = 16000
CSI_FS = 83.0
WIN = 128


# ---------------------------------------------------------------------------
# 合成数据
# ---------------------------------------------------------------------------
def write_wav(path: Path, x: np.ndarray, fs: int = FS_AUDIO) -> None:
    """把 float32 [-1,1] 写成 16-bit PCM 单声道 wav。"""
    pcm = np.clip(x, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(fs)
        w.writeframes(pcm.tobytes())


def make_clip(duration_s: float = 7.0, click_at_s: float = 3.0,
              click_amp: float = 0.8, tone_amp: float = 0.01) -> np.ndarray:
    """底噪+微弱正弦，在 ``click_at_s`` 处插入 60 ms 的衰减冲击（模拟落地声）。"""
    n = int(duration_s * FS_AUDIO)
    t = np.arange(n) / FS_AUDIO
    x = tone_amp * np.sin(2 * np.pi * 220.0 * t)
    x += 0.001 * np.random.default_rng(0).standard_normal(n)

    i0 = int(click_at_s * FS_AUDIO)
    ln = int(0.06 * FS_AUDIO)
    env = np.exp(-np.arange(ln) / (0.012 * FS_AUDIO))
    burst = click_amp * env * np.random.default_rng(1).standard_normal(ln)
    x[i0:i0 + ln] += burst
    return x.astype(np.float32)


def write_sync(path: Path, csi_start_pc: float, audio_start_pc: float,
               n_rows: int = 498) -> None:
    path.write_text(json.dumps({
        "csi_rows": n_rows, "csi_fs": CSI_FS, "audio_fs": FS_AUDIO,
        "csi_start_pc": csi_start_pc, "audio_start_pc": audio_start_pc,
        "lag_applied_ms": 0.0,
    }), encoding="utf-8")


def make_csi_csv(path: Path, n_rows: int = 300, seed: int = 0) -> None:
    """造一条最小可解析的 CSI CSV（104 个浮点 = 52 组实虚部）。"""
    rng = np.random.default_rng(seed)
    base = 20.0 + 15.0 * rng.random(52)          # 静态 |H(f)| 形状
    lines = ["type,local_timestamp,data"]
    for i in range(n_rows):
        amp = base * (1.0 + 0.04 * rng.standard_normal(52))
        vals = np.empty(104, dtype=np.float32)
        vals[0::2] = amp
        vals[1::2] = 0.0
        payload = ",".join(f"{v:.3f}" for v in vals)
        lines.append(f'CSI_DATA,{1700000000000000 + i * 12048},"[{payload}]"')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. wav 读写
# ---------------------------------------------------------------------------
def test_wav_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "a.wav"
        x = make_clip(duration_s=1.0, click_at_s=0.5)
        write_wav(p, x)
        y, fs = read_wav(p)
        assert fs == FS_AUDIO, fs
        assert y.dtype == np.float32 and y.ndim == 1
        assert abs(len(y) - len(x)) <= 1, (len(y), len(x))
        # 16-bit 量化误差应远小于信号幅度（削顶样本除外，浮点峰值可能 >1）
        ok = np.abs(x) < 0.9
        assert np.max(np.abs(y[ok] - x[ok])) < 2e-3
    print("  [1/5] wav 读写 OK")


# ---------------------------------------------------------------------------
# 2. 时间轴对齐：咔哒声必须落在正确的窗里
# ---------------------------------------------------------------------------
def test_alignment_locates_click() -> None:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        csv_p = td / "fall_001.csv"
        make_csi_csv(csv_p, n_rows=300)

        # 音频第 0 采样对应 PC 时刻 999.5；CSI 第 0 行对应 1000.0（head pad 0.5s）
        csi_start, audio_start = 1000.0, 999.5
        click_at = 3.0                      # 音频绝对时刻（秒）
        write_wav(td / "fall_001.wav", make_clip(7.0, click_at_s=click_at))
        write_sync(td / "fall_001.sync.json", csi_start, audio_start, n_rows=300)

        # 咔哒声落在第几行： (click - (csi_start - audio_start)) * csi_fs
        click_row = (click_at - (csi_start - audio_start)) * CSI_FS   # ≈207.5
        starts = np.arange(0, 300 - WIN + 1, 32)
        # 相邻窗有 75% 重叠，一声咔哒通常落在 2 个窗里 —— 这些窗都算"命中"，别拿来互相比较
        inside = [i for i, s in enumerate(starts) if s <= click_row < s + WIN]
        assert inside, "合成数据没把冲击放进任何窗，检查用例参数"
        hit = inside[0]

        feats = extract_audio_window_features(csv_p, starts, WIN, csi_fs=CSI_FS)
        assert feats.shape == (len(starts), AUDIO_DIM), feats.shape

        local = [AUDIO_FEATURE_NAMES.index(n) for n in
                 ("audio_onset_max_db", "audio_crest_db", "audio_logrms_max")]
        others = [i for i in range(len(starts)) if i not in inside]
        hit_onset = float(feats[hit, local[0]])
        other_onset = float(np.max(feats[others, local[0]]))
        assert hit_onset > other_onset + 10.0, (
            f"冲击窗的起振 {hit_onset:.2f} dB 没有明显高于无冲击窗 {other_onset:.2f} dB "
            f"—— 时间轴大概率错位"
        )

        # 平移 0.5 s（约 41 行）后，冲击应落到别的窗里；原来那批窗不再命中
        write_sync(td / "fall_001.sync.json", csi_start, audio_start + 0.5, n_rows=300)
        feats2 = extract_audio_window_features(csv_p, starts, WIN, csi_fs=CSI_FS)
        shifted_row = click_row - 0.5 * CSI_FS
        shifted_inside = [i for i, s in enumerate(starts) if s <= shifted_row < s + WIN]
        assert shifted_inside, "平移后冲击落在窗序列之外，用例参数需要调整"
        assert set(shifted_inside) != set(inside), "平移量太小，命中窗没有变化"
        peak_onset = int(np.argmax(feats2[:, local[0]]))
        assert peak_onset in shifted_inside, (
            "改 sync.json 后特征没有跟着移动 —— 对齐参数被忽略了"
        )
    print("  [2/5] sync.json 时间轴对齐 OK（含 0.5 s 平移复检）")


# ---------------------------------------------------------------------------
# 3. 缺失 / 损坏音频的降级
# ---------------------------------------------------------------------------
def test_missing_audio_degrades() -> None:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        csv_p = td / "walk_001.csv"
        make_csi_csv(csv_p, n_rows=300)
        starts = [0, 32, 64]

        # (a) 什么都没有
        a = extract_audio_window_features(csv_p, starts, WIN, csi_fs=CSI_FS)
        assert a.shape == (3, AUDIO_DIM)
        assert np.all(a[:, AUDIO_FEATURE_NAMES.index("audio_present")] == 0.0)
        assert np.allclose(a[:, :-2], 0.0), "无音频时非元信息列应全为 0"

        # (b) 有 wav 没 sync：无法对齐，同样按"无音频"处理
        write_wav(td / "walk_001.wav", make_clip(7.0))
        b = extract_audio_window_features(csv_p, starts, WIN, csi_fs=CSI_FS)
        assert np.all(b[:, AUDIO_FEATURE_NAMES.index("audio_present")] == 0.0)

        # (c) 补上 sync 后恢复成正常特征
        write_sync(td / "walk_001.sync.json", 1000.0, 999.5, n_rows=300)
        c = extract_audio_window_features(csv_p, starts, WIN, csi_fs=CSI_FS)
        assert np.all(c[:, AUDIO_FEATURE_NAMES.index("audio_present")] == 1.0)

        # (d) 损坏的 wav 不抛异常（读取失败 -> 降级）
        (td / "walk_001.wav").write_bytes(b"not a wav file")
        d = extract_audio_window_features(csv_p, starts, WIN, csi_fs=CSI_FS)
        assert d.shape == (3, AUDIO_DIM)
        assert np.all(d[:, AUDIO_FEATURE_NAMES.index("audio_present")] == 0.0)
    print("  [3/5] 缺失/损坏音频降级 OK")


# ---------------------------------------------------------------------------
# 4. 特征定长、有限、可复现
# ---------------------------------------------------------------------------
def test_feature_sanity() -> None:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        csv_p = td / "fall_002.csv"
        make_csi_csv(csv_p, n_rows=300)
        write_wav(td / "fall_002.wav", make_clip(7.0, click_at_s=2.5))
        write_sync(td / "fall_002.sync.json", 1000.0, 999.5, n_rows=300)

        track = read_audio_track(csv_p)
        assert track is not None and track.duration > 6.0
        f1 = track.features_for_rows(100, WIN)
        f2 = track.features_for_rows(100, WIN)
        assert f1.shape == (AUDIO_DIM,)
        assert np.array_equal(f1, f2), "同一窗两次提取结果不一致（特征不确定）"
        assert np.isfinite(f1).all()

        # 越界的行号不应崩，覆盖率应下降
        f3 = track.features_for_rows(10 ** 6, WIN)
        assert np.isfinite(f3).all()
        assert f3[AUDIO_FEATURE_NAMES.index("audio_present")] == 0.0

        # 静音：电平应贴近下限而不是 NaN
        import wifalldemo.audio as A
        silent = A.analyse_audio(np.zeros(FS_AUDIO, dtype=np.float32), FS_AUDIO)
        assert np.isfinite(silent.db).all()
        assert float(silent.db.max()) <= -60.0
    print("  [4/5] 特征定长/有限/可复现 OK")


# ---------------------------------------------------------------------------
# 5. 与 dataset 拼接：维度正确且 CSI 部分不变
# ---------------------------------------------------------------------------
def test_dataset_concat() -> None:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        csv_p = td / "fall_003.csv"
        make_csi_csv(csv_p, n_rows=300)
        write_wav(td / "fall_003.wav", make_clip(7.0, click_at_s=3.0))
        write_sync(td / "fall_003.sync.json", 1000.0, 999.5, n_rows=300)

        csi = features_for_sample(str(csv_p), win=WIN, step=32, fs=CSI_FS, audio=False)
        both = features_for_sample(str(csv_p), win=WIN, step=32, fs=CSI_FS, audio=True)
        assert csi is not None and both is not None
        X_csi, names_csi, starts_csi = csi
        X_both, names_both, starts_both = both

        assert names_csi == names_both[:len(names_csi)]
        assert len(names_both) - len(names_csi) == AUDIO_DIM
        assert X_csi.shape[1] + AUDIO_DIM == X_both.shape[1]
        assert np.array_equal(starts_csi, starts_both), "音频接入改变了切窗位置"
        assert np.array_equal(X_csi, X_both[:, :X_csi.shape[1]]), (
            "拼音频后 CSI 特征被改动了 —— 两条通路必须互不影响"
        )
        assert np.all(X_both[:, -2] == 1.0), "音频存在标志应为 1"
    print("  [5/5] dataset 拼接 CSI+音频 OK")


# ---------------------------------------------------------------------------
# 6. 逐文件电平居中（v2）：不同增益的两条录音，居中后电平特征应可比
# ---------------------------------------------------------------------------
def test_per_file_level_centering() -> None:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        base = make_clip(7.0, click_at_s=3.0)
        feats = []
        for name, gain in (("quiet", 0.2), ("loud", 1.0)):
            csv_p = td / f"{name}.csv"
            make_csi_csv(csv_p, n_rows=300)
            write_wav(td / f"{name}.wav", (base * gain).astype(np.float32))
            write_sync(td / f"{name}.sync.json", 1000.0, 999.5, n_rows=300)
            f = extract_audio_window_features(csv_p, [0, 64, 128], WIN, csi_fs=CSI_FS)
            feats.append(f)

        fq, fl = feats
        # 居中后：每条录音内 logrms_mean 的中位数应≈0
        for tag, f in (("quiet", fq), ("loud", fl)):
            med = float(np.median(f[:, AUDIO_FEATURE_NAMES.index("audio_logrms_mean")]))
            assert abs(med) < 1.0, f"{tag}: 居中后 logrms_mean 中位数应≈0，实际 {med:.2f}"
        # 相对量（crest / 频带占比）本就不随增益变；绝对量居中后与增益无关
        for n in ("audio_logrms_mean", "audio_logrms_max"):
            c = AUDIO_FEATURE_NAMES.index(n)
            assert np.allclose(fq[:, c], fl[:, c], atol=1.5), (
                f"居中后 {n} 仍随增益漂移 —— 逐文件归一化没生效"
            )
        cr = AUDIO_FEATURE_NAMES.index("audio_crest_db")
        assert np.allclose(fq[:, cr], fl[:, cr], atol=1.0), "crest 是增益不变量，不应受影响"
    print("  [6/6] 逐文件电平居中 OK（±14 dB 增益差被消掉）")


def main() -> None:
    print("音频通路测试:")
    test_wav_roundtrip()
    test_alignment_locates_click()
    test_missing_audio_degrades()
    test_feature_sanity()
    test_dataset_concat()
    test_per_file_level_centering()
    print("全部通过。")


if __name__ == "__main__":
    main()
