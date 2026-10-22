#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""打分后处理：文件内相对打分、top-k 投票、"坠落后静止"确认规则。

    python -m tests.test_postprocess

覆盖:
  1. relative_scores：文件内 median≈0、MAD 缩放正确；两组的尺度互不串扰
  2. topk_file_scores：取每文件最高 k 个窗的均值
  3. 静止规则：fall 模式（暴 trigger + 触发后安静）通过；stand 模式
     （触发后继续活动 / 触发不够暴）被否决；无后置窗时不否决
  4. tune_stillness_rule：规则能提升验证 F1 时返回参数，纯噪声时退回 None
  5. 与 run_train 的口径耦合：find_activity_index 能在真实特征名里找到列
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from wifalldemo.features import feature_names  # noqa: E402
from wifalldemo.postprocess import (  # noqa: E402
    apply_stillness_rule,
    find_activity_index,
    relative_scores,
    topk_file_scores,
    tune_stillness_rule,
)


def test_relative_scores() -> None:
    # 文件 A：低尺度信号；文件 B：同形状但整体 +100（模拟批次增益差）
    g = np.array([0] * 20 + [1] * 20)
    pA = np.linspace(0.0, 1.0, 20)
    p = np.concatenate([pA, pA + 100.0])
    z = relative_scores(p, g)
    zA, zB = z[:20], z[20:]
    assert np.allclose(zA, zB, atol=1e-9), "相对打分应消掉两文件的绝对尺度差"
    assert abs(float(np.median(zA))) < 1e-9, "稳健 z-score 后中位数应≈0"
    # MAD 缩放：z 的稳健 std 应≈1
    mad = np.median(np.abs(zA - np.median(zA)))
    assert abs(1.4826 * mad - 1.0) < 0.2, f"稳健 std 应≈1，实际 {1.4826 * mad:.2f}"
    # 单窗文件 MAD=0 不应产生 inf
    z1 = relative_scores(np.array([0.3, 0.7]), np.array([0, 1]))
    assert np.isfinite(z1).all()
    print("  [1/5] 文件内相对打分 OK")


def test_topk_file_scores() -> None:
    g = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.9, 0.5, 0.2, 0.4])
    y = np.array([1, 1, 1, 0, 0])
    p_f, y_f = topk_file_scores(p, g, y, k=2)
    # 文件 0: top2 = {0.9, 0.5} -> 0.7；文件 1: top2 = {0.4, 0.2} -> 0.3
    assert np.allclose(p_f, [0.7, 0.3]), p_f
    assert list(y_f) == [1, 0]
    print("  [2/5] top-k 文件投票 OK")


def _file_pattern(kind: str, n: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """造单文件的 (分数, 活动度) 时间序列：fall / stand 两种模式。"""
    rng = np.random.default_rng(0)
    if kind == "fall":       # 冲击窗高分高活动，随后长时间静止
        p = np.full(n, -0.2)
        act = np.full(n, 0.02)
        p[4:7] = [3.0, 4.0, 2.5]
        act[4:7] = [0.5, 0.8, 0.4]
        act[7:] = 0.02
    elif kind == "stand":    # 站起瞬间有活动，随后继续走动（不归零）
        p = np.full(n, -0.2)
        act = np.full(n, 0.02)
        p[4:7] = [2.0, 2.8, 1.8]
        act[4:7] = [0.3, 0.4, 0.3]
        act[7:] = 0.25
    else:
        raise ValueError(kind)
    return p + rng.normal(0, 0.01, n), act + rng.normal(0, 0.002, n)


def test_stillness_rule() -> None:
    rule = {"still_ratio": 0.6, "burst_ratio": 3.0, "post_n": 4,
            "threshold_topk": 1.0}
    p_f, act_f = _file_pattern("fall")
    p_s, act_s = _file_pattern("stand")
    p = np.concatenate([p_f, p_s])
    act = np.concatenate([act_f, act_s])
    g = np.array([0] * len(p_f) + [1] * len(p_s))
    pred = apply_stillness_rule(p, g, act, rule)
    assert pred[0] == 1, "fall（触发后静止）应被确认"
    assert pred[1] == 0, "stand（触发后继续活动）应被否决"
    assert (pred >= 0).all() and (pred <= 1).all()

    # 触发后没有后置窗（冲击发生在文件末尾）：不否决
    p2, act2 = _file_pattern("fall")
    p2[7:], act2[7:] = -0.2, 0.02   # 已无关
    p_short = p2[:7]                # 只留到触发结束
    act_short = act2[:7]
    pred2 = apply_stillness_rule(
        p_short, np.zeros(len(p_short), dtype=int), act_short, rule)
    assert pred2[0] == 1, "无后置窗时不应否决"

    # 触发不够暴（stand 的触发能量相对基线不足 burst_ratio）：否决
    rule_lo = dict(rule, burst_ratio=30.0)
    pred3 = apply_stillness_rule(p, g, act, rule_lo)
    assert pred3[0] == 0 and pred3[1] == 0
    print("  [3/5] 坠落后静止规则 OK（fall 通过 / stand 否决 / 无后置窗不否决）")


def test_tune_stillness_rule() -> None:
    # 场景 1：stand 误报存在且规则能救 -> 应返回参数
    rng = np.random.default_rng(1)
    ps, acts = [], []
    ys, gs = [], []
    for i in range(6):
        p_f, a_f = _file_pattern("fall")
        p_s, a_s = _file_pattern("stand")
        ps += [p_f, p_s]; acts += [a_f, a_s]
        ys += [1] * len(p_f) + [0] * len(p_s)
        gs += [i * 2] * len(p_f) + [i * 2 + 1] * len(p_s)
    p = np.concatenate(ps); act = np.concatenate(acts)
    g = np.array(gs); y = np.array(ys)
    p_f_file, y_f_file = topk_file_scores(p, g, y, k=3)
    thr = 1.0   # top-k 分数阈值：fall≈3.2 过线，stand≈2.2 过线（误报）
    assert (p_f_file >= thr).sum() == len(p_f_file)
    rule = tune_stillness_rule(p, g, act, y, thr)
    assert rule is not None, "规则能提升 F1 时不应退回 None"
    hard = apply_stillness_rule(p, g, act, rule)
    f1_before = 2 * (6 / 12) * (6 / 12) / (6 / 12 + 6 / 12)  # 全判正 = P=R=0.5
    tp = int(((hard == 1) & (y_f_file == 1)).sum()); fp = int(((hard == 1) & (y_f_file == 0)).sum())
    fn = int(((hard == 0) & (y_f_file == 1)).sum())
    f1_after = 2 * tp / (2 * tp + fp + fn)
    assert f1_after > f1_before, f"规则应提升 F1（{f1_after:.2f} vs {f1_before:.2f}）"

    # 场景 2：没有误报（stand 分数低于阈值），规则没有可救的 -> 退回 None
    p2 = p.copy()
    p2[g == 1] -= 5.0   # 把第一组 stand 压到阈值下（后续组类似）
    for gg in np.unique(g)[1::2]:
        p2[g == gg] -= 5.0
    rule2 = tune_stillness_rule(p2, g, act, y, thr)
    assert rule2 is None, "无误报时规则应自动关闭"
    print("  [4/5] 规则参数网格搜索 OK（有救则启用 / 无救则关闭）")


def test_activity_index_in_real_features() -> None:
    names = feature_names()
    idx = find_activity_index(names)
    assert idx is not None, "真实 CSI 特征里必须能找到活动度特征"
    assert names[idx].endswith("_diff_mean")
    print(f"  [5/5] 活动度特征定位 OK（列 {idx}: {names[idx]}）")


def main() -> None:
    print("打分后处理测试:")
    test_relative_scores()
    test_topk_file_scores()
    test_stillness_rule()
    test_tune_stillness_rule()
    test_activity_index_in_real_features()
    print("全部通过。")


if __name__ == "__main__":
    main()
