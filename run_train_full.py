#!/usr/bin/env python
"""用全部 ID11~ID15 数据训练，并用三种评估协议对照——把"窗级泄漏"量化出来。

动机：把 5 个人的数据全部交给模型训练是部署时的正常做法（数据越多越好），
但**评估**必须分清楚三种口径，否则数字会骗人：

  A. 训练集内评估（in-sample）——模型打它背过的答案，最虚高；
  B. 随机窗划分（random window split）——把全部窗打乱后 80/20 切分。
     这是论文里最常见的坑：同一条录音的相邻窗（步进只有 0.39s，重叠 75%）
     同时落进训练和测试，模型等于"开卷考试"，指标同样虚高；
  C. 留一交叉验证（LOPO）——测试人完全不参与训练，是唯一诚实的口径，
     结果直接读 ``outputs/audio_ablation_lopo.json``（run_audio_ablation.py --lopo）。

用法::

    python run_train_full.py                # 复用特征缓存
    python run_train_full.py --no-cache     # 强制重建特征

产物::

    outputs/models/final_all.joblib          全量数据训出的融合模型（部署候选）
    outputs/metrics_full_train.json          三种协议 × 三种输入的完整指标
    outputs/figures/train_full_leakage.png   协议对比柱状图
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from wifalldemo import config
from wifalldemo.data_loader import scan_dataset
from wifalldemo.dataset import build_feature_dataset
from wifalldemo.models import (
    evaluate,
    predict_proba_pos,
    save_model,
    train_mlp,
    youden_threshold,
)
from wifalldemo.postprocess import topk_file_scores

# (tag, 显示名, 列选择): csi=非 audio_ 列, aud=audio_ 列, all=全部
INPUTS = [("mlp", "CSI-only", "csi"), ("mlp_av", "CSI+Audio", "all"), ("audio", "Audio-only", "aud")]


TOPK = 3


def file_vote(y: np.ndarray, p: np.ndarray, groups: np.ndarray, thr: float,
              topk: bool = False) -> dict:
    """文件级投票。

    topk=False: 全窗均值（旧口径）
    topk=True : 取概率最高的 min(3, n) 个窗求平均 —— 主口径。跌倒是短时事件，
                一个文件里只有少数窗含冲击，全窗均值会把证据稀释掉；而
                stand/sit 这类误报往往只有零星高分窗，top-k 同样能压住。
    """
    gids = np.unique(groups)
    y_f = np.array([int(round(float(y[groups == g].mean()))) for g in gids])
    if topk:
        p_f, _ = topk_file_scores(p, groups, y, k=TOPK)
    else:
        p_f = np.array([float(p[groups == g].mean()) for g in gids])
    return evaluate(y_f, p_f, thr)


def select_cols(X: np.ndarray, names: list[str], kind: str) -> np.ndarray:
    is_aud = np.array([n.startswith("audio_") for n in names])
    if kind == "csi":
        return X[:, ~is_aud]
    if kind == "aud":
        return X[:, is_aud]
    return X


def main() -> None:
    ap = argparse.ArgumentParser(description="全量 ID11~15 训练 + 泄漏口径对照")
    ap.add_argument("--data-dir", default="realdata")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--seed", type=int, default=42, help="随机窗划分的种子")
    args = ap.parse_args()

    records = scan_dataset(args.data_dir)
    users = sorted(records["user"].unique())
    print(f"样本 {len(records)} 条，用户 {users}，fall={int((records.label == 1).sum())}")

    cache_fp = (
        config.CACHE_DIR
        / f"features_w{config.WINDOW_SIZE}_s{config.WINDOW_STEP}_fs{config.SAMPLING_RATE_HZ:.1f}.npz"
    )
    X, y, rids, names = build_feature_dataset(
        records, fs=config.SAMPLING_RATE_HZ, cache_fp=cache_fp, force=args.no_cache,
        audio=True,
    )
    print(f"滑窗 {X.shape[0]} 条 × {X.shape[1]} 维（fall={int((y == 1).sum())}）")

    report: dict = {"data_dir": args.data_dir, "users": users, "n_files": len(records),
                    "n_windows": int(X.shape[0]), "protocols": {}}

    # ---------------- 协议 A：训练集内评估（背答案） ----------------
    print("\n=== 协议 A：训练集内评估（模型评它训练时见过的窗，最虚高） ===")
    prot_a = {}
    final_models = {}
    for tag, label, kind in INPUTS:
        Xs = select_cols(X, names, kind)
        model = train_mlp(Xs, y, groups=rids)
        p = predict_proba_pos(model, Xs)
        thr = youden_threshold(y, p)          # 阈值也在训练数据上选——同样泄漏
        p_tk_f, y_f = topk_file_scores(p, rids, y, k=TOPK)
        thr_tk = youden_threshold(y_f, p_tk_f)   # top-k 分数分布不同，阈值必须重选
        met_w = evaluate(y, p, thr)
        met_f = file_vote(y, p, rids, thr)
        met_ft = file_vote(y, p, rids, thr_tk, topk=True)
        prot_a[tag] = {"threshold": thr, "threshold_topk": thr_tk,
                       "window": met_w, "file": met_f, "file_topk": met_ft}
        final_models[tag] = model
        print(f"  [{label:<10}] thr={thr:.3f} 窗 Acc={met_w['accuracy']:.4f} AUC={met_w['auc']:.4f} "
              f"| 文件 Acc={met_f['accuracy']:.4f} F1={met_f['f1']:.4f} "
              f"| top-k Acc={met_ft['accuracy']:.4f} F1={met_ft['f1']:.4f}")
    report["protocols"]["in_sample"] = prot_a

    # ---------------- 协议 B：随机窗划分 80/20（开卷考试） ----------------
    print("\n=== 协议 B：随机窗划分 80/20（同一条录音的相邻窗跨训练/测试，仍泄漏） ===")
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(len(y))
    n_tr = int(len(y) * 0.8)
    tr_i, te_i = perm[:n_tr], perm[n_tr:]
    prot_b = {}
    for tag, label, kind in INPUTS:
        Xs = select_cols(X, names, kind)
        model = train_mlp(Xs[tr_i], y[tr_i], groups=rids[tr_i])
        p_tr = predict_proba_pos(model, Xs[tr_i])
        thr = youden_threshold(y[tr_i], p_tr)
        p = predict_proba_pos(model, Xs[te_i])
        p_tr_f, y_tr_f = topk_file_scores(p_tr, rids[tr_i], y[tr_i], k=TOPK)
        thr_tk = youden_threshold(y_tr_f, p_tr_f)
        met_w = evaluate(y[te_i], p, thr)
        met_f = file_vote(y[te_i], p, rids[te_i], thr)
        met_ft = file_vote(y[te_i], p, rids[te_i], thr_tk, topk=True)
        prot_b[tag] = {"threshold": thr, "threshold_topk": thr_tk,
                       "window": met_w, "file": met_f, "file_topk": met_ft}
        print(f"  [{label:<10}] thr={thr:.3f} 窗 Acc={met_w['accuracy']:.4f} AUC={met_w['auc']:.4f} "
              f"| 文件 Acc={met_f['accuracy']:.4f} F1={met_f['f1']:.4f} "
              f"| top-k Acc={met_ft['accuracy']:.4f} F1={met_ft['f1']:.4f}")
    report["protocols"]["random_window_split"] = prot_b

    # ---------------- 协议 C：LOPO（读已有结果，唯一诚实口径） ----------------
    lopo_fp = config.OUTPUT_DIR / "audio_ablation_lopo.json"
    prot_c = None
    if lopo_fp.exists():
        lopo = json.loads(lopo_fp.read_text(encoding="utf-8"))
        prot_c = {}
        for tag, label, _ in INPUTS:
            auc = float(np.mean([r[tag]["auc"] for r in lopo["rows"]]))
            acc = float(np.mean([r[tag]["file_acc"] for r in lopo["rows"]]))
            f1 = float(np.mean([r[tag]["file_f1"] for r in lopo["rows"]]))
            prot_c[tag] = {"auc": auc, "file_acc": acc, "file_f1": f1}
        report["protocols"]["lopo"] = prot_c
        print("\n=== 协议 C：LOPO 留一交叉验证（测试人完全不参与训练，诚实口径） ===")
        for tag, label, _ in INPUTS:
            c = prot_c[tag]
            print(f"  [{label:<10}] 窗 AUC={c['auc']:.4f} | 文件 Acc={c['file_acc']:.4f} F1={c['file_f1']:.4f}")
    else:
        print("\n[!] 未找到 audio_ablation_lopo.json，跳过 LOPO 对照（先跑 run_audio_ablation.py --lopo --reuse）")

    # ---------------- 落盘：全量融合模型（部署候选） ----------------
    save_model(final_models["mlp_av"], names, config.MODEL_DIR, "final_all")
    # 部署候选的阈值一并落盘：全量训练没有留出人，阈值只能在训练集内选（虚高），
    # 所以这里只是"能用"的起点，真实预期仍以 LOPO 为准。
    thr_fp = config.MODEL_DIR / "final_all_thresholds.json"
    thr_fp.write_text(json.dumps({
        "window": float(prot_a["mlp_av"]["threshold"]),
        "file_mean": float(prot_a["mlp_av"]["threshold"]),
        "file_topk": float(prot_a["mlp_av"]["threshold_topk"]),
        "score_mode": "raw",        # 与 LOPO 主口径一致：原始概率打分
        "note": "训练集内 Youden-J 选出（无留出人），仅作起点；性能预期以 LOPO 为准",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    out = config.OUTPUT_DIR / "metrics_full_train.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    # ---------------- 画图 ----------------
    fig_path = _plot(report, prot_c)
    print(f"\n融合模型已存 {config.MODEL_DIR / 'final_all.joblib'}（全量数据，部署候选）")
    print(f"指标: {out}\n图: {fig_path}")


def _plot(report: dict, prot_c) -> Path:
    import matplotlib.pyplot as plt

    from wifalldemo.visualize import save_fig

    tags = [t for t, _, _ in INPUTS]
    labels = [l for _, l, _ in INPUTS]
    protocols = [("lopo", "LOPO\n(诚实口径)"), ("random_window_split", "随机窗划分\n(泄漏)"),
                 ("in_sample", "训练集内\n(最虚高)")]
    protos = [p for p, _ in protocols if p in report["protocols"]]

    def val(proto, tag, metric):
        d = report["protocols"][proto][tag]
        if proto == "lopo":
            return d[{"win_auc": "auc", "file_acc": "file_acc"}[metric]]
        return d["window"]["auc"] if metric == "win_auc" else d["file"]["accuracy"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = {"mlp": "tab:blue", "mlp_av": "tab:red", "audio": "tab:orange"}
    for ax, metric, title in (
        (axes[0], "win_auc", "Window-level AUC"),
        (axes[1], "file_acc", "File-level Accuracy"),
    ):
        for j, (tag, lab) in enumerate(zip(tags, labels)):
            vals = [val(p, tag, metric) for p in protos]
            x = np.arange(len(protos)) + (j - 1) * 0.26
            ax.bar(x, vals, width=0.24, color=colors[tag], label=lab)
            for xi, v in zip(x, vals):
                ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
        ax.set_xticks(np.arange(len(protos)))
        ax.set_xticklabels([lab for p, lab in protocols if p in protos], fontsize=9)
        ax.set_ylim(0, 1.12)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("同一批数据、同一模型：评估口径不同，指标天差地别", fontsize=12)
    fig.tight_layout()
    return save_fig(fig, "train_full_leakage.png")


if __name__ == "__main__":
    main()
