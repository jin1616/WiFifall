#!/usr/bin/env python
"""离线检测演示：用训练好的模型回放测试样本，逐窗打分并绘图。

展示两张图:
  1) CSI 幅度热力图（可看到动作引起的信号扰动）
  2) 随时间变化的跌倒概率曲线 + 判定阈值（触发即判定为跌倒）

用法:
    python run_train.py        # 先训练
    python run_demo_offline.py

该版本同时支持三种深度学习模型：
  * MLP / DLSmall — 输入 70 维手工特征
  * CNN1D         — 输入 (T, S) 原始滑窗张量
按模型类型自动选择特征通路。
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from wifalldemo import config
from wifalldemo.cnn import CNNClassifier
from wifalldemo.data_loader import scan_dataset, person_split
from wifalldemo.dataset import build_raw_window_dataset, features_for_sample
from wifalldemo.models import load_model, load_norm_config, predict_proba_pos
from wifalldemo.postprocess import (
    apply_stillness_rule,
    find_activity_index,
    hybrid_scores,
    relative_scores,
)
from wifalldemo.preprocessing import load_sample_amplitude, normalize_per_subcarrier
from wifalldemo.visualize import plot_csi_sample, plot_detection_timeline


def load_threshold() -> float:
    p = config.OUTPUT_DIR / "metrics.json"
    if p.exists():
        return float(json.loads(p.read_text(encoding="utf-8"))["threshold"])
    return 0.5


def load_primary_model():
    """读取训练时的主模型名并加载对应权重。"""
    p = config.OUTPUT_DIR / "metrics.json"
    tag = "mlp"  # 默认主训练后保存的别名
    if p.exists():
        tag = str(json.loads(p.read_text(encoding="utf-8")).get("primary_model", tag))
    return load_model(config.MODEL_DIR, tag), tag


def _raw_windows_for_file(path: str, win: int, step: int,
                          fs: float = config.SAMPLING_RATE_HZ,
                          norm: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """复用 dataset._process_raw_windows 的同一切窗约定；返回 (raw, starts)。

    训练和推理必须走同一套表示：模型是用归一化后的幅度训的，这里也得归一化，
    否则喂进去的是裸幅度（带着静态 |H(f)| 形状），打分会完全不同。

    ``norm`` 是 :func:`load_norm_config` 取到的**训练时**口径（居中/因果 + 窗长）。
    不传则退回当前 config —— 那只在没有 metrics.json 时才该发生。
    """
    amp = load_sample_amplitude(path)
    if norm is None:
        norm = load_norm_config()
    if norm["enabled"]:
        amp = normalize_per_subcarrier(
            amp, fs=fs, window_s=norm["window_s"], causal=norm["causal"]
        )
    T = amp.shape[0]
    starts = list(range(0, T - win + 1, step))
    raw = np.stack([amp[s : s + win] for s in starts], axis=0).astype(np.float32)
    return raw, np.asarray(starts)


def build_model_input(path: str, feat_names: list[str] | None, win: int, step: int,
                     fs: float):
    """按**模型存档的特征名**构造输入矩阵，返回 ``(X, starts)``。

    模型文件名旁边的 ``<tag>_features.json`` 记录了它训练时吃哪些列：纯 CSI 模型
    是 70 个 raw_*/rel_*，融合模型是 101 个（多 31 个 audio_*），仅音频模型只有
    audio_*。这里据此决定要不要现算音频特征、以及切哪几列 —— 靠"猜维度"迟早会
    静默喂错分布。
    """
    names = list(feat_names or [])
    need_audio = any(n.startswith("audio_") for n in names)
    res = features_for_sample(path, win=win, step=step, fs=fs, audio=need_audio)
    if res is None:
        raise ValueError(f"样本 {path} 的包数不足一个窗 ({win})")
    X, all_names, starts = res
    if not names:
        return X, starts  # CNN 没有特征名概念
    try:
        idx = [all_names.index(n) for n in names]
    except ValueError:
        if X.shape[1] == len(names):
            return X, starts
        raise ValueError(
            f"模型需要的特征与当前工程算出的特征对不上（缺 "
            f"{[n for n in names if n not in all_names][:3]}…），请用同一版本重训"
        )
    return X[:, idx], starts


def demo_on_file(path: str, tag: str, gt_label: int, model, threshold: float,
                 fs: float, win: int = config.WINDOW_SIZE,
                 step: int = config.WINDOW_STEP, norm: dict | None = None,
                 feat_names: list[str] | None = None, meta: dict | None = None):
    """对单个样本做滑窗打分，根据模型类型自动挑选输入维度。

    * CNN1D       → (n_win, win, S) 原始张量
    * MLP/DLSmall → 按模型存档的特征名构造（纯 CSI 70 维 / 融合 101 维 / 仅音频 31 维）

    文件级判定用**训练主口径**（见 metrics.json 的 postprocess 字段）：相对打分
    → top-k 窗投票 → "坠落后静止"确认。旧版 metrics.json 没有这些字段时退回
    "全窗均值 >= 阈值"的老口径。
    """
    if isinstance(model, CNNClassifier):
        raw, starts = _raw_windows_for_file(path, win, step, fs=fs, norm=norm)
        amp_for_plot = load_sample_amplitude(path)
        feat, feat_starts = None, starts
    else:
        amp = load_sample_amplitude(path)
        amp_for_plot = amp
        feat, feat_starts = build_model_input(path, feat_names, win, step, fs)
    prob = predict_proba_pos(model, feat if feat is not None else raw)

    meta = meta or {}
    post = meta.get("postprocess", {}) if isinstance(meta.get("postprocess"), dict) else {}
    mode = post.get("score_mode")
    if mode is None:                       # 旧版 metrics.json 只有布尔位
        mode = "rel" if post.get("relative_scoring") else "raw"
    groups = np.zeros(len(prob), dtype=int)          # 单文件：全部窗同属一组
    if mode == "rel":
        prob_eval = relative_scores(prob, groups)
    elif mode == "hybrid":
        prob_eval = hybrid_scores(prob, groups)
    else:
        prob_eval = prob
    thr_tk = float(meta.get("threshold_topk", threshold))

    k = min(3, len(prob_eval))
    topk_score = float(np.sort(prob_eval)[-k:].mean())
    pred_fall = topk_score >= thr_tk
    # 静止确认规则：需要 CSI 活动度特征（CNN 原始张量通路没有，跳过）
    rule = None
    rules_by_tag = meta.get("stillness_rules") or {}
    rule_key = meta.get("primary_model") if tag == "primary" else tag
    if rule_key in rules_by_tag:
        rule = rules_by_tag[rule_key]
    act = None
    if rule is not None and feat is not None and feat_names:
        ai = find_activity_index(list(feat_names))
        if ai is not None:
            act = feat[:, ai]
    veto = ""
    if rule is not None and pred_fall and act is not None:
        if not bool(apply_stillness_rule(prob_eval, groups, act, rule)[0]):
            pred_fall = False
            veto = "（触发后仍在活动，被静止规则否决）"
    elif rule is not None and act is None:
        veto = "（无活动度特征，静止规则跳过）"

    times = (np.asarray(feat_starts, dtype=float) + win / 2.0) / fs
    alarms = int((prob_eval >= threshold).sum())

    audio_note = ""
    if feat is not None and feat_names and any(n.startswith("audio_") for n in feat_names):
        n_audio = sum(1 for n in feat_names if n.startswith("audio_"))
        audio_note = f", 输入 {feat.shape[1]} 维 (含 {n_audio} 维音频)"

    print(f"\n  [{tag}] 真实标签={config.ACTIVITY_LABELS[gt_label]}, "
          f"滑窗数={len(prob)}, 触发报警窗={alarms}/{len(prob)}{audio_note}")
    for t, p in zip(times, prob_eval):
        mark = "!! FALL ALARM !!" if p >= threshold else ""
        tag_mode = "" if mode == "raw" else f" ({mode})"
        print(f"    t={t:6.2f}s  P(fall)={p:+.3f}{tag_mode:9s}  {mark}")

    outcome = "检出跌倒" if pred_fall else "判定非跌倒"
    print(f"  -> top-k 投票={topk_score:.3f} (阈值 {thr_tk:.3f}): {outcome}"
          f"{veto} (真实: {config.ACTIVITY_LABELS[gt_label]})")

    plot_csi_sample(
        amp_for_plot, f"demo_csi_{tag}.png",
        title=f"CSI of {'Fall' if gt_label else 'Non-Fall'} sample [{tag}]",
    )
    plot_detection_timeline(
        times, prob, gt_label, threshold, fs,
        name=f"demo_timeline_{tag}.png",
        title=f"Fall probability over windows — {tag} "
              f"(true={config.ACTIVITY_LABELS[gt_label]})",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="离线检测演示 (DL)")
    ap.add_argument("--model", default=None,
                    help="mlp / dlsmall / cnn / primary(默认，训练择优结果)")
    ap.add_argument("--data-dir", default=None,
                    help="数据源（目录或 .pkl）。默认读 metrics.json 里训练时的"
                         " data_dir（保证演示数据和模型同源），没有再退回 config.DATA_DIR")
    ap.add_argument("--test-users", nargs="+", default=None,
                    help="演示用的测试人员；默认读 metrics.json 的 test_users")
    args = ap.parse_args()

    meta: dict = {}
    meta_p = config.OUTPUT_DIR / "metrics.json"
    if meta_p.exists():
        meta = json.loads(meta_p.read_text(encoding="utf-8"))

    if args.model:
        model, feat_names = load_model(config.MODEL_DIR, args.model)
        tag = args.model
    else:
        (model, feat_names), tag = load_primary_model()
    threshold = load_threshold()
    # 归一化口径跟模型走，不跟当前 config 走：config 改了不该影响已训好的模型
    norm = load_norm_config()
    fs = config.SAMPLING_RATE_HZ
    model_kind = "CNN1D" if isinstance(model, CNNClassifier) else "MLP-family"
    uses_audio = bool(feat_names) and any(n.startswith("audio_") for n in feat_names)
    print(f"已加载模型 {tag} ({model_kind}); 决策阈值={threshold:.3f}")
    print(f"输入特征: {len(feat_names) if feat_names else '原始 (T,S) 张量'}"
          + ("（含音频特征：CSI+音频融合）" if uses_audio else ""))
    print(f"归一化口径(来自 {norm['source']}): "
          f"enabled={norm['enabled']}, 参考窗={norm['window_s']}s, "
          f"{'因果窗' if norm['causal'] else '居中窗'}")

    # 演示数据默认跟训练同源：metrics.json 记了 data_dir 和 test_users；
    # 否则退回 config.DATA_DIR + ID9（Widar 时代的历史默认）。
    data_src = args.data_dir or meta.get("data_dir") or str(config.DATA_DIR)
    test_users = args.test_users or meta.get("test_users") or ["ID9"]
    print(f"数据源: {data_src}  演示用户: {test_users}")
    records = scan_dataset(data_src)
    _, test = person_split(records, tuple(test_users))
    assert len(test) > 0, f"测试用户 {test_users} 不在数据源 {data_src} 里"

    fall_rows = test[test.label == 1]
    assert len(fall_rows) > 0, f"测试用户 {test_users} 没有 fall 样本，换 --test-users"
    fall_file = fall_rows["path"].iloc[0]
    targets = [("stand", 0), ("walk", 0)]
    chosen = []
    for act, lab in targets:
        rows = test[(test.activity == act) & (test.label == lab)]
        if len(rows):
            chosen.append((rows["path"].iloc[0], f"{act}", lab))

    print("\n================ 离线回放演示 ================")
    demo_on_file(fall_file, "fall", 1, model, threshold, fs, norm=norm,
                 feat_names=feat_names, meta=meta)
    for path, t, lab in chosen:
        demo_on_file(path, t, lab, model, threshold, fs, norm=norm,
                     feat_names=feat_names, meta=meta)

    print(f"\n图片已保存到 {config.FIG_DIR} (demo_*.png)")


if __name__ == "__main__":
    main()
