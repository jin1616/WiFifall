#!/usr/bin/env python
"""误报归因诊断：非跌倒动作（尤其 walk / stand）为什么会被判成 fall。

训练指标只告诉你"误报有多少"，不会告诉你"哪一路输入在骗模型"。本脚本对
**同一折训练出来的三个模型**（纯 CSI / CSI+音频 / 仅音频）喂**同一批特征**，
比较它们在各活动上的文件级得分：

* 三个模型都高      → 两路信号里都有"像跌倒"的成分（动作本身易混）
* 只有 audio 高     → 声音把非跌倒动作（脚步、椅子响）当成了撞击
* 只有 mlp 高       → CSI 的运动能量类特征在起作用（步态扰动大）
* 都低但融合仍报    → 融合层交互，需要看单窗得分曲线（--plot）

> 前提：``outputs/models/`` 必须和 ``outputs/metrics.json`` 是同一次训练的产物。
> 换划分跑实验时务必加 ``--no-save``，否则最后一个实验的权重会顶掉部署模型，
> 而 metrics.json 还停在上一次 —— 诊断就会变成"模型测它见过的数据"。

用法::

    python diag_false_alarm.py                       # 默认用 metrics.json 的 test_users
    python diag_false_alarm.py --test-users ID13     # 指定测试人
    python diag_false_alarm.py --plot                # 额外画 walk/stand 的窗得分曲线

产物（加 --plot 时）：outputs/figures/diag_false_alarm.png
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from wifalldemo import config
from wifalldemo.data_loader import person_split, scan_dataset
from wifalldemo.dataset import features_for_sample
from wifalldemo.models import load_model, predict_proba_pos

MODELS = [("mlp", "CSI-only"), ("mlp_av", "CSI+Audio"), ("audio", "Audio-only")]
ACTS = ["fall", "walk", "sit", "stand", "jump"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="误报归因诊断")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--test-users", nargs="+", default=None)
    ap.add_argument("--plot", action="store_true", help="画误报文件的窗得分曲线")
    ap.add_argument("--swap", action="store_true",
                    help="CSI/音频特征交换矩阵：定位误报由哪一路输入驱动")
    return ap.parse_args()


def score_files(samples, all_names, tag):
    """按模型存档的特征名切列打分，返回 (文件级均值概率, 窗级概率列表)。"""
    model, feat_names = load_model(config.MODEL_DIR, tag)
    idx = [all_names.index(n) for n in feat_names]
    win_probs = [predict_proba_pos(model, X[:, idx]) for _, X in samples]
    return np.array([float(p.mean()) for p in win_probs]), win_probs


def main() -> None:
    args = parse_args()
    meta: dict = {}
    mp = config.OUTPUT_DIR / "metrics.json"
    if mp.exists():
        meta = json.loads(mp.read_text(encoding="utf-8"))
    data_src = args.data_dir or meta.get("data_dir") or str(config.DATA_DIR)
    test_users = args.test_users or meta.get("test_users") or ["ID13"]
    thr = float(meta.get("threshold", 0.5))
    fs = float(meta.get("fs_hz", config.SAMPLING_RATE_HZ))

    recs = scan_dataset(data_src)
    _, test = person_split(recs, tuple(test_users))
    print(f"数据源={data_src}  测试人={test_users}  样本数={len(test)}  阈值={thr:.3f}")

    samples, acts, all_names = [], [], []
    for act in ACTS:
        for p in test.loc[test.activity == act, "path"]:
            res = features_for_sample(p, fs=fs, audio=True)
            if res is None:
                continue
            X, names, _ = res
            samples.append((p, X))
            acts.append(act)
            all_names = names
    acts = np.asarray(acts)

    per_model, win_by_model = {}, {}
    for tag, _ in MODELS:
        per_model[tag], win_by_model[tag] = score_files(samples, all_names, tag)

    # ---- 1. 各活动的文件级得分 ----
    print(f"\n文件级平均 P(fall)（阈值 {thr:.3f}，>= 即报警）")
    head = f"{'活动':<7}{'n':>4}" + "".join(f"{name:>13}" for _, name in MODELS)
    print("-" * len(head)); print(head); print("-" * len(head))
    for act in ACTS:
        m = acts == act
        line = f"{act:<7}{int(m.sum()):>4}"
        for tag, _ in MODELS:
            line += f"{per_model[tag][m].mean():>13.3f}"
        print(line)

    # ---- 2. 主模型的实际误报/漏报文件数 ----
    primary = str(meta.get("primary_model", "mlp_av"))
    pred = (per_model[primary] >= thr).astype(int)
    true = (acts == "fall").astype(int)
    print(f"\n主模型 {primary} 的文件级混淆（n={len(acts)}）")
    for act in ACTS:
        m = acts == act
        n, k = int(m.sum()), int(pred[m].sum())
        label = "误报" if act != "fall" else "召回"
        print(f"  {act:<6} n={n:<3} 报警 {k:<3} ({k / max(n, 1):5.1%})  -> {label}")

    # ---- 3. 判别力：fall vs 各非跌倒动作能否分开 ----
    print("\nfall vs 各动作 的文件级得分 AUC（1.0=完全分开，0.5=分不开）")
    for act in [a for a in ACTS if a != "fall"]:
        m_pos, m_neg = acts == "fall", acts == act
        line = f"  fall vs {act:<6}"
        for tag, _ in MODELS:
            a, b = per_model[tag][m_pos], per_model[tag][m_neg]
            auc = float((a[:, None] > b[None, :]).mean() + 0.5 * (a[:, None] == b[None, :]).mean())
            line += f"  {tag}={auc:.3f}"
        print(line)

    # ---- 4. 特征层：误报活动与 fall 最接近的维度 ----
    X_fall = np.vstack([X for (_, X), a in zip(samples, acts) if a == "fall"])
    worst = max([a for a in ACTS if a != "fall"],
                key=lambda a: float((per_model[primary][acts == a]).mean()))
    X_bad = np.vstack([X for (_, X), a in zip(samples, acts) if a == worst])
    mu_f, mu_b, sd = X_fall.mean(0), X_bad.mean(0), np.vstack([X_fall, X_bad]).std(0) + 1e-9
    z = (mu_b - mu_f) / sd
    print(f"\n{worst}（最容易误报的活动）与 fall 最接近的 10 个特征（|z| 小 = 难分）")
    for i in np.argsort(np.abs(z))[:10]:
        print(f"  {all_names[i]:<24} fall={mu_f[i]:9.3f}  {worst}={mu_b[i]:9.3f}  z={z[i]:+.2f}")

    if args.swap:
        swap_matrix(test, fs)
    if args.plot:
        plot(samples, acts, win_by_model[primary], pred, primary, test_users[0])


def swap_matrix(test, fs: float) -> None:
    """CSI×音频 交换矩阵：行 = CSI 来源活动，列 = 音频来源活动。

    把 A 活动窗的 CSI 特征和 B 活动窗的音频特征拼起来打分。若某行的分数对列
    "换什么都高"，误报就由**该行活动的 CSI** 驱动；若某列普遍抬高分数，则该列
    活动的音频像跌倒。注意直接把特征置零是无效的——全零是训练时没见过的分布，
    融合模型会外推到极端概率（实测 0.99+），交换才是在有效分布内的归因。
    """
    acts = ACTS
    model, feat_names = load_model(config.MODEL_DIR, "mlp_av")
    na = sum(1 for n in feat_names if n.startswith("audio_"))
    csi, aud, names_all = {}, {}, None
    for act in acts:
        xs, xa = [], []
        for p in test.loc[test.activity == act, "path"]:
            res = features_for_sample(p, fs=fs, audio=True)
            if res is None:
                continue
            X, names_all, _ = res
            X = X[:, [names_all.index(n) for n in feat_names]]
            xs.append(X[:, :-na])
            xa.append(X[:, -na:])
        csi[act], aud[act] = np.vstack(xs), np.vstack(xa)
    n = min(min(len(v) for v in csi.values()), min(len(v) for v in aud.values()))
    print(f"\nCSI×音频交换矩阵（每格 {n} 窗，平均 P(fall)；行=CSI 来源，列=音频来源）")
    print(f"{'CSI\\AUD':<8}" + "".join(f"{a:>8}" for a in acts))
    for a in acts:
        line = f"{a:<8}"
        for b in acts:
            X = np.hstack([csi[a][:n], aud[b][:n]])
            line += f"{predict_proba_pos(model, X).mean():>8.3f}"
        print(line)
    print("读法：某行整体偏高 → 该活动的 CSI 误导模型；某列整体偏高 → 该活动的音频像跌倒。")


def plot(samples, acts, win_probs, pred, tag, user) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 误报最集中的活动 + fall 各取 3 条，看窗级曲线是"整段都高"还是"局部尖峰"
    fp_acts = [a for a in ACTS if a != "fall"]
    bad = max(fp_acts, key=lambda a: int(pred[acts == a].sum()))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8), sharey=True)
    for ax, act in zip(axes, ("fall", bad)):
        idx = np.where(acts == act)[0][:3]
        for i in idx:
            t = (np.arange(len(win_probs[i])) * config.WINDOW_STEP
                 + config.WINDOW_SIZE / 2) / config.SAMPLING_RATE_HZ
            ax.plot(t, win_probs[i], lw=1.0, label=f"#{i}")
        ax.set_title(f"{act} — window P(fall), {user} ({tag})")
        ax.set_xlabel("time (s)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("P(fall)")
    fig.tight_layout()
    out = config.FIG_DIR / "diag_false_alarm.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"\n图: {out}")


if __name__ == "__main__":
    main()
