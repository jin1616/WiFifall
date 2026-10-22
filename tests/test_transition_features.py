"""窗内时序结构特征（FEATURE_VERSION=2 新增）的单元测试。

背景：误报归因发现"全程静止的 stand 窗"与"跌倒文件里冲击前/后的静止窗"
在原有 29×2 统计量上几乎同分布 —— 那些特征都是全窗平均，把"运动→骤停"
这个转换抹平了。v2 新增的 6×2 维（`*_act_*`）专门刻画窗内活动度序列的
**形状**。本测试用三种合成信号验证它们确实分得开：

  * burst_quiet : 静止 -> 突发剧烈运动 -> 骤停（跌倒冲击窗的形状）
  * uniform     : 全程均匀运动（walk/jump 的形状）
  * still       : 全程静止（stand 的形状）

运行:  python -m tests.test_transition_features
"""

from __future__ import annotations

import numpy as np

from wifalldemo.features import FEATURE_VERSION, feature_names, window_feature_vector

FS = 83.0
WIN = 128
S = 52


def _make(kind: str, seed: int = 0) -> np.ndarray:
    """构造 (WIN, S) 幅度窗：基线 10 + 噪声，运动段叠加大幅波动。"""
    rng = np.random.RandomState(seed)   # 每次调用重置，保证可复现
    t = np.arange(WIN)
    base = 10.0 + rng.randn(WIN, S) * 0.05
    if kind == "still":
        return base.astype(np.float32)
    if kind == "uniform":
        wave = np.sin(2 * np.pi * 1.2 * t / FS)[:, None] * 1.5
        return (base + wave + rng.randn(WIN, S) * 0.4).astype(np.float32)
    if kind == "burst_quiet":
        # 中段 0.3s 的剧烈冲击，前后静止 —— 跌倒冲击窗的典型形状
        sig = base.copy()
        m = (t >= WIN // 2 - 12) & (t < WIN // 2 + 12)
        sig[m] += rng.randn(m.sum(), S) * 3.0
        return sig.astype(np.float32)
    raise ValueError(kind)


def _act_feats(vec: np.ndarray, names: list[str], prefix: str) -> dict:
    return {n: float(vec[names.index(n)]) for n in names if n.startswith(f"{prefix}_act_")}


def main() -> None:
    names = feature_names()
    assert FEATURE_VERSION >= 2, "FEATURE_VERSION 未随新特征递增"
    assert len(names) == 70, f"v2 应为 35×2=70 维，实际 {len(names)}"
    for p in ("raw", "rel"):
        for suffix in ("act_early_late", "act_max_drop", "act_drop_pos",
                       "act_max_rise", "act_peak_pos", "act_tail_quiet"):
            assert f"{p}_{suffix}" in names, f"缺少特征 {p}_{suffix}"
    print("  [1/3] 特征名录：70 维、6×2 个 act_* 特征齐全")

    vec_b, _ = window_feature_vector(_make("burst_quiet"), fs=FS)
    vec_u, _ = window_feature_vector(_make("uniform"), fs=FS)
    vec_s, _ = window_feature_vector(_make("still"), fs=FS)

    for prefix in ("raw", "rel"):
        fb = _act_feats(vec_b, names, prefix)
        fu = _act_feats(vec_u, names, prefix)
        fs_ = _act_feats(vec_s, names, prefix)

        # 1) 冲击窗的"骤降"必须显著大于匀速运动窗和静止窗
        assert fb[f"{prefix}_act_max_drop"] > fu[f"{prefix}_act_max_drop"] + 0.5, (
            f"{prefix}: burst 的 max_drop={fb[f'{prefix}_act_max_drop']:.2f} "
            f"未明显超过 uniform 的 {fu[f'{prefix}_act_max_drop']:.2f}"
        )
        # 2) 冲击窗末尾相对安静（骤停后静止），匀速运动窗末尾不安静
        assert fb[f"{prefix}_act_tail_quiet"] < fu[f"{prefix}_act_tail_quiet"] - 0.5, (
            f"{prefix}: burst 的 tail_quiet={fb[f'{prefix}_act_tail_quiet']:.2f} "
            f"未明显低于 uniform 的 {fu[f'{prefix}_act_tail_quiet']:.2f}"
        )
        # 3) 冲击窗的活动峰应落在窗中段附近（0.25~0.75），而不是随机位置
        assert 0.2 < fb[f"{prefix}_act_peak_pos"] < 0.8, (
            f"{prefix}: burst 的 peak_pos={fb[f'{prefix}_act_peak_pos']:.2f} 不在中段"
        )
        # 4) 静止窗的 max_drop 应接近 0（全程平稳，没有转换）
        assert fs_[f"{prefix}_act_max_drop"] < fb[f"{prefix}_act_max_drop"], (
            f"{prefix}: still 的 max_drop 应小于 burst"
        )
        print(f"  [act/{prefix}] burst drop={fb[f'{prefix}_act_max_drop']:.2f} "
              f"tail={fb[f'{prefix}_act_tail_quiet']:.2f} | "
              f"uniform drop={fu[f'{prefix}_act_max_drop']:.2f} "
              f"tail={fu[f'{prefix}_act_tail_quiet']:.2f} | "
              f"still drop={fs_[f'{prefix}_act_max_drop']:.2f}")
    print("  [2/3] 冲击/匀速/静止三种形状在 act_* 特征上可分")

    # 5) 确定性：同一输入两次计算必须逐位一致（缓存安全）
    vec_b2, _ = window_feature_vector(_make("burst_quiet"), fs=FS)
    assert np.array_equal(vec_b, vec_b2), "特征计算不具备确定性"
    print("  [3/3] 确定性 OK")


if __name__ == "__main__":
    main()
    print("test_transition_features: ALL PASS")
