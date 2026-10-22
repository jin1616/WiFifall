#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""逐子载波归一化（R45 的修复）: 锁住"对乘性 H(f) 精确不变"这个核心性质。

    python -m tests.test_subcarrier_normalization

背景（2026-09-23 实测）：主模型 CNN1D 吃的是**裸幅度**，|H(f)| 的静态形状直接
进网络。给 H(f) 加一点扰动，CNN 窗级准确率 0.975 → 0.525（且是漏报方向）；
自家板子选择性 2.5~2.9× vs Widar 5.6×，等于每个人一套 H(f)，模型只能记住"这是谁"。

这里锁住的是**表示本身**的性质，与模型无关：
  * 乘性信道增益必须被完全除掉（这是选"逐子载波"而非全局归一化的理由）
  * 两个人同样的动作、不同的静态 H(f)，归一化后应当得到同一个张量
  * 因果模式（流式）只能用历史帧
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from wifalldemo.preprocessing import normalize_per_subcarrier  # noqa: E402

FS = 83.0
N_TONES = 52
RNG = np.random.default_rng(0)


def _sample(t: int = 500, burst: bool = True) -> np.ndarray:
    """造一条样本：静态 H(f) 形状 + 逐帧扰动 + 可选的跌倒瞬态。"""
    shape = 10.0 + 40.0 * RNG.random(N_TONES)          # 静态 |H(f)|，选择性 ~5x
    amp = shape[None, :] * (1.0 + 0.05 * RNG.standard_normal((t, N_TONES)))
    if burst:
        b = slice(t // 2, t // 2 + 40)                 # 0.5 s 的跌倒扰动
        amp[b] *= 1.0 + 2.0 * RNG.random(N_TONES)[None, :]
    return np.clip(amp, 1e-3, None).astype(np.float32)


def test_multiplicative_h_invariant() -> None:
    amp = _sample()
    gain = (0.2 + 3.0 * RNG.random(N_TONES)).astype(np.float32)   # 每个子载波一个增益
    n1 = normalize_per_subcarrier(amp, fs=FS, window_s=1.5)
    n2 = normalize_per_subcarrier(amp * gain[None, :], fs=FS, window_s=1.5)
    err = float(np.abs(n1 - n2).max())
    rel = err / max(float(np.abs(n1).max()), 1e-9)
    assert rel < 1e-4, f"乘性 H(f) 没被除干净: 相对误差 {rel:.2e}"
    print(f"  乘性 H(f)（每子载波 0.2~3.2x）：归一化后相对误差 {rel:.2e}（≈0）, OK")


def test_two_people_same_motion_same_tensor() -> None:
    """两个人不同房间（不同静态 H(f)）、做同一个动作 → 归一化后应当一致。"""
    motion = 1.0 + 0.3 * RNG.standard_normal((300, N_TONES))
    motion[150:190] *= 2.0                       # 同一段跌倒扰动
    motion = np.clip(motion, 1e-3, None)
    h_a = (5.0 + 10.0 * RNG.random(N_TONES)).astype(np.float32)   # 选择性 ~3x
    h_b = (20.0 + 60.0 * RNG.random(N_TONES)).astype(np.float32)  # 选择性 ~4x，形状完全不同
    a = normalize_per_subcarrier((motion * h_a[None, :]).astype(np.float32), fs=FS)
    b = normalize_per_subcarrier((motion * h_b[None, :]).astype(np.float32), fs=FS)
    err = float(np.abs(a - b).max())
    assert err < 1e-3, f"不同 H(f) 没被对齐: 最大差 {err:.3e}"
    print(f"  两个房间（选择性 3x vs 4x）同一动作：归一化后最大差 {err:.1e}, OK")


def test_baseline_near_one() -> None:
    amp = _sample()
    n = normalize_per_subcarrier(amp, fs=FS, window_s=1.5)
    base = float(np.median(n))
    assert 0.7 < base < 1.4, f"基线应落在 1.0 附近，实际中位 {base:.3f}"
    assert n.shape == amp.shape and n.dtype == np.float32
    print(f"  基线中位 {base:.3f}（≈1.0），形状/ dtype 保持, OK")


def test_short_sample_falls_back() -> None:
    """样本比参考窗还短：退化成整条中位数，仍然对乘性 H(f) 不变。"""
    amp = _sample(t=60)
    gain = (0.5 + 2.0 * RNG.random(N_TONES)).astype(np.float32)
    a = normalize_per_subcarrier(amp, fs=FS, window_s=1.5)
    b = normalize_per_subcarrier(amp * gain[None, :], fs=FS, window_s=1.5)
    assert float(np.abs(a - b).max()) < 1e-4
    print("  短样本（60 帧 < 1.5s 窗）：退化到整条中位数且仍不变, OK")


def test_causal_only_uses_history() -> None:
    """因果模式：改未来帧不能影响当前时刻的输出。"""
    amp = _sample(t=300)
    n1 = normalize_per_subcarrier(amp, fs=FS, window_s=1.5, causal=True)
    amp2 = amp.copy()
    amp2[200:] *= 5.0                      # 把"未来"整段抬 5 倍
    n2 = normalize_per_subcarrier(amp2, fs=FS, window_s=1.5, causal=True)
    assert float(np.abs(n1[:200] - n2[:200]).max()) < 1e-4, "因果模式泄漏了未来"
    # 对照：非因果（居中）模式必然会被未来影响，这正好说明两者确实不同
    m1 = normalize_per_subcarrier(amp, fs=FS, window_s=1.5)
    m2 = normalize_per_subcarrier(amp2, fs=FS, window_s=1.5)
    assert float(np.abs(m1[:180] - m2[:180]).max()) > 1e-3
    print("  因果模式只用历史（未来帧改动不影响过去）；居中模式会受影响（符合预期）, OK")


def test_causal_also_removes_h() -> None:
    """因果模式（部署路径）同样必须对乘性 H(f) 精确不变——否则流式等于白做。"""
    amp = _sample(t=400)
    gain = (0.2 + 3.0 * RNG.random(N_TONES)).astype(np.float32)
    a = normalize_per_subcarrier(amp, fs=FS, window_s=1.5, causal=True)
    b = normalize_per_subcarrier(amp * gain[None, :], fs=FS, window_s=1.5, causal=True)
    rel = float(np.abs(a - b).max()) / max(float(np.abs(a).max()), 1e-9)
    assert rel < 1e-4, f"因果模式没除掉 H(f): 相对误差 {rel:.2e}"
    print(f"  因果模式同样对乘性 H(f) 不变（相对误差 {rel:.2e}）, OK")


def test_removes_static_shape_keeps_transient() -> None:
    """归一化要去掉静态形状，但**不能**把跌倒瞬态也抹平。"""
    amp = _sample(t=500, burst=True)
    n = normalize_per_subcarrier(amp, fs=FS, window_s=1.5)
    quiet = np.r_[0:200, 300:500]
    burst = slice(250, 290)
    # 归一化后静态选择性应当塌到 1 附近
    sel_before = float(np.median(amp, axis=0).max() / np.median(amp, axis=0).min())
    sel_after = float(np.median(n, axis=0).max() / np.median(n, axis=0).min())
    assert sel_after < sel_before / 2, f"静态选择性没被去掉: {sel_before:.2f}→{sel_after:.2f}"
    # 瞬态必须还在
    dev_q = float(np.abs(n[quiet] - 1.0).mean())
    dev_b = float(np.abs(n[burst] - 1.0).mean())
    assert dev_b > 3 * dev_q, f"瞬态被抹平了: 安静段 {dev_q:.3f} vs 瞬态 {dev_b:.3f}"
    print(f"  静态选择性 {sel_before:.2f}x → {sel_after:.2f}x；"
          f"瞬态偏离 {dev_b:.3f} 是安静段 {dev_q:.3f} 的 {dev_b/dev_q:.1f} 倍（保留下来）, OK")


def test_norm_config_follows_model_not_config() -> None:
    """归一化口径必须跟**模型**走，而不是当前 config。

    归一化是模型输入表示的一部分：训练用因果窗、推理用居中窗（或反过来），模型
    吃到的就是它没见过的分布。所以这个口径要像 threshold 一样存档在 metrics.json，
    **改 config 不应该改变已训好模型的推理行为**。
    """
    import tempfile
    from pathlib import Path

    from wifalldemo import config as cfg
    from wifalldemo.models import load_norm_config, norm_config_snapshot

    snap = norm_config_snapshot()
    assert set(snap) == {"enabled", "window_s", "causal"}, f"快照字段不对: {set(snap)}"

    # 造一份"训练时存档"的 metrics.json：因果窗 2.5s
    d = Path(tempfile.mkdtemp())
    (d / "metrics.json").write_text(
        __import__("json").dumps(
            {"threshold": 0.3,
             "normalization": {"enabled": True, "window_s": 2.5, "causal": True}}),
        encoding="utf-8")
    got = load_norm_config(output_dir=d, verbose=False)
    assert got["source"] == "metrics.json"
    assert got["causal"] is True and got["window_s"] == 2.5, f"没读到训练口径: {got}"

    # 把 config 改成**完全相反**的口径，读出来的仍必须是训练时那份
    old = (cfg.NORMALIZE_PER_SUBCARRIER, cfg.NORM_WINDOW_S, cfg.NORM_CAUSAL)
    try:
        cfg.NORMALIZE_PER_SUBCARRIER, cfg.NORM_WINDOW_S, cfg.NORM_CAUSAL = False, 9.0, False
        got2 = load_norm_config(output_dir=d, verbose=False)
        assert got2 == got, f"config 改了不该影响已训好模型的口径: {got2} != {got}"
    finally:
        cfg.NORMALIZE_PER_SUBCARRIER, cfg.NORM_WINDOW_S, cfg.NORM_CAUSAL = old

    # 老产物（metrics.json 里没这个键）→ 退回 config 并标出来
    empty = Path(tempfile.mkdtemp()) / "sub"
    empty.mkdir()
    got3 = load_norm_config(output_dir=empty, verbose=False)
    assert got3["source"] == "config", f"缺记录时应标记 config 来源: {got3}"
    print("  归一化口径随模型存档：config 改成相反值后仍读出训练时的因果/2.5s；"
          "老产物退回 config 并标记, OK")


if __name__ == "__main__":
    test_multiplicative_h_invariant()
    test_two_people_same_motion_same_tensor()
    test_baseline_near_one()
    test_short_sample_falls_back()
    test_causal_only_uses_history()
    test_causal_also_removes_h()
    test_removes_static_shape_keeps_transient()
    test_norm_config_follows_model_not_config()
    print("\n[PASS] 逐子载波归一化 8/8")
