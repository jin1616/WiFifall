#!/usr/bin/env python
"""离线检测演示：用训练好的模型回放测试样本，逐窗打分并绘图。

展示两张图:
  1) CSI 幅度热力图（可看到动作引起的信号扰动）
  2) 随时间变化的跌倒概率曲线 + 判定阈值（触发即判定为跌倒）

用法:
    python run_train.py        # 先训练
    python run_demo_offline.py
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from wifalldemo import config
from wifalldemo.data_loader import scan_dataset, person_split
from wifalldemo.features import extract_window_features
from wifalldemo.models import load_model, predict_proba_pos
from wifalldemo.visualize import plot_csi_sample, plot_detection_timeline
from wifalldemo.preprocessing import load_amplitude_series


def load_threshold() -> float:
    p = config.OUTPUT_DIR / "metrics.json"
    if p.exists():
        return float(json.loads(p.read_text(encoding="utf-8"))["threshold"])
    return 0.5


def load_primary_model():
    """读取训练时的主模型名并加载对应权重。"""
    p = config.OUTPUT_DIR / "metrics.json"
    tag = "rf"
    if p.exists():
        tag = str(json.loads(p.read_text(encoding="utf-8")).get("primary_model", tag))
    return load_model(config.MODEL_DIR, tag), tag


def demo_on_file(path: str, tag: str, gt_label: int, model, threshold: float, fs: float):
    """对单个样本做滑窗打分，打印每窗结果并保存概率时间线图。"""
    amp = load_amplitude_series(path)
    feat, _, starts = extract_window_features(amp, fs=fs)
    prob = predict_proba_pos(model, feat)
    times = (np.asarray(starts, dtype=float) + config.WINDOW_SIZE / 2.0) / fs
    alarms = int((prob >= threshold).sum())

    print(f"\n  [{tag}] 真实标签={config.ACTIVITY_LABELS[gt_label]}, "
          f"滑窗数={len(prob)}, 触发报警窗={alarms}/{len(prob)}")
    for t, p in zip(times, prob):
        mark = "!! FALL ALARM !!" if p >= threshold else ""
        print(f"    t={t:6.2f}s  P(fall)={p:.3f}  {mark}")

    # 与训练期文件级评估一致：窗平均概率 >= 决策阈值即判跌倒
    outcome = "检出跌倒" if float(prob.mean()) >= threshold else "判定非跌倒"
    print(f"  -> 多数投票结论: {outcome} (真实: {config.ACTIVITY_LABELS[gt_label]})")

    plot_csi_sample(
        amp, f"demo_csi_{tag}.png",
        title=f"CSI of {'Fall' if gt_label else 'Non-Fall'} sample [{tag}]",
    )
    plot_detection_timeline(
        times, prob, gt_label, threshold, fs,
        name=f"demo_timeline_{tag}.png",
        title=f"Fall probability over windows — {tag} "
              f"(true={config.ACTIVITY_LABELS[gt_label]})",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="离线检测演示")
    ap.add_argument("--model", default=None,
                    help="rf / lr / primary(默认，训练择优结果)")
    ap.add_argument("--test-users", nargs="+", default=["ID9"])
    args = ap.parse_args()

    if args.model:
        model, feat_names = load_model(config.MODEL_DIR, args.model)
        tag = args.model
    else:
        (model, feat_names), tag = load_primary_model()
    threshold = load_threshold()
    fs = config.SAMPLING_RATE_HZ
    print(f"已加载模型 {tag}; 决策阈值={threshold:.3f}")

    records = scan_dataset(config.DATA_DIR)
    _, test = person_split(records, tuple(args.test_users))
    assert len(test) > 0, "请选择存在于数据集中的测试用户"

    fall_file = test[test.label == 1]["path"].iloc[0]
    # 从多个非跌倒活动中各挑一个做对比（sit/stand 中 sit 最难，可在
    # outputs/metrics.json 与混淆矩阵中查看 sit 的更多误报统计）
    targets = [("stand", 0), ("walk", 0)]
    chosen = []
    for act, lab in targets:
        rows = test[(test.activity == act) & (test.label == lab)]
        if len(rows):
            chosen.append((rows["path"].iloc[0], f"{act}", lab))

    print("\n================ 离线回放演示 ================")
    demo_on_file(fall_file, "fall", 1, model, threshold, fs)
    for path, tag, lab in chosen:
        demo_on_file(path, tag, lab, model, threshold, fs)

    print(f"\n图片已保存到 {config.FIG_DIR} (demo_*.png)")


if __name__ == "__main__":
    main()
