#!/usr/bin/env python
"""音频 / CSI 单模态与融合的跨人消融实验。

回答三个问题：

1. **声音本身能不能判跌倒？**      → ``audio``（仅 31 维音频特征）
2. **加了声音比只用 CSI 强多少？**  → ``mlp``（58 维 CSI）vs ``mlp_av``（58+31 融合）
3. **这个增益是稳定的还是某个人碰巧？** → 对多个测试人各跑一遍，看均值和极差

之所以要按人跑多遍：自采数据一人才 20 条 fall，单次划分的方差极大，
"某一次融合比 CSI 高 0.1"说明不了任何事；只有跨人一致才算数。

用法::

    python run_audio_ablation.py                      # 默认 ID12~ID15，val=ID11
    python run_audio_ablation.py --test-users ID12 ID13
    python run_audio_ablation.py --no-cache           # 强制重建特征
    python run_audio_ablation.py --lopo               # 全部 N 人各留出一次（留一交叉验证）

``--lopo`` 是回答"这批数据整体准确率是多少"的正确口径：ID11~ID15 一共才 5 个人，
任何单次留出都只测了一个人（80 来个文件），方差大到没有参考价值；让每个人轮流
当测试人、其余人训练，再把 N 折的预测汇总，才等价于"把这批数据交给模型"的真实
泛化水平。验证人自动从**其余人**里取（不能和测试人重合，否则阈值是在测试集上选的）。

产物：
    outputs/audio_ablation.json         每人的三项指标
    outputs/figures/audio_ablation.png  分组柱状图（AUC / 文件级 F1）
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from wifalldemo import config

ROOT = Path(__file__).resolve().parent
CONFIGS = [
    ("mlp", "CSI-only", "#7f8c8d"),
    ("mlp_av", "CSI + Audio", "#c0392b"),
    ("audio", "Audio-only", "#2980b9"),
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="音频 vs CSI 跨人消融")
    ap.add_argument("--data-dir", default="realdata")
    ap.add_argument("--test-users", nargs="+", default=None,
                    help="留出的测试人；--lopo 时默认取数据目录下全部用户")
    ap.add_argument("--val-users", nargs="+", default=None,
                    help="选阈值用的人；--lopo 时自动从测试人之外取")
    ap.add_argument("--lopo", action="store_true",
                    help="留一交叉验证：每个人各当一次测试人，val 自动避开测试人")
    ap.add_argument("--reuse", action="store_true",
                    help="已有 metrics_ablation_<user>.json 就直接汇总，不重训")
    ap.add_argument("--no-cache", action="store_true", help="透传给 run_train.py")
    ap.add_argument("--score-mode", choices=("rel", "hybrid", "raw"), default=None,
                    help="透传：打分后处理（默认沿用 run_train 的 rel）")
    ap.add_argument("--stillness", action="store_true",
                    help="透传：启用坠落后静止确认规则（默认关，LOPO 上未通过验证集调参）")
    ap.add_argument("--gain-aug-db", type=float, default=None, metavar="DB",
                    help="透传：音频增益增广幅度（默认沿用 run_train 的 6 dB）")
    ap.add_argument("--tag", default="", help="结果文件名后缀，便于区分不同设置")
    args = ap.parse_args()
    if args.lopo and args.test_users is None:
        args.test_users = discover_users(args.data_dir)
    if args.test_users is None:
        args.test_users = ["ID12", "ID13", "ID14", "ID15"]
        args.val_users = args.val_users or ["ID11"]
    return args


def discover_users(data_dir: str) -> list[str]:
    """数据目录下所有 ``ID*`` 子目录，按名字排序（ID11, ID12, …）。"""
    root = Path(data_dir)
    if not root.is_absolute():
        root = ROOT / root
    return sorted(p.name for p in root.glob("ID*") if p.is_dir())


def resolve_val(user: str, args: argparse.Namespace) -> list[str]:
    """该折的验证人。LOPO 下必须避开测试人，否则阈值是在测试集上挑的。"""
    if args.val_users:
        vals = [u for u in args.val_users if u != user]
        if vals:
            return vals
    others = [u for u in args.test_users if u != user]
    return others[:1]


def file_counts(data_dir: str) -> dict[str, int]:
    """每个测试人的文件数，用于把各折指标按样本量加权（人多折权重大）。"""
    from wifalldemo.data_loader import scan_dataset

    root = Path(data_dir)
    if not root.is_absolute():
        root = ROOT / root
    recs = scan_dataset(str(root))
    return {u: int((recs["user"] == u).sum()) for u in recs["user"].unique()}


def run_one(user: str, args: argparse.Namespace) -> dict:
    """跑一次 run_train（跳过 CNN：消融只关心手工特征通路，跑得快 5 倍）。"""
    out = config.OUTPUT_DIR / f"metrics_ablation_{user}{args.tag}.json"
    if args.reuse and out.exists():
        print(f"\n>>> 测试人 {user} … 复用 {out.name}")
    else:
        cmd = [
            sys.executable, str(ROOT / "run_train.py"),
            "--data-dir", args.data_dir,
            "--test-users", user,
            "--val-users", *resolve_val(user, args),
            "--no-balance", "--no-cnn", "--audio", "--no-save",
            "--metrics-out", str(out),
        ]
        if args.no_cache:
            cmd.append("--no-cache")
        if args.score_mode:
            cmd += ["--score-mode", args.score_mode]
        if args.stillness:
            cmd.append("--stillness")
        if args.gain_aug_db is not None:
            cmd += ["--gain-aug-db", str(args.gain_aug_db)]
        print(f"\n>>> 测试人 {user} …")
        subprocess.run(cmd, cwd=str(ROOT), check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    meta = json.loads(out.read_text(encoding="utf-8"))
    row = {"user": user, "val": meta.get("val_users", [])}
    for tag, _, _ in CONFIGS:
        w = meta["window_metrics"].get(tag, {})
        f = meta["file_metrics"].get(tag, {})
        ft = meta.get("file_topk_metrics", {}).get(tag, {})
        ftc = meta.get("file_topk_confirmed", {}).get(tag, {})
        row[tag] = {
            "auc": float(w.get("auc", 0.0)),
            "win_acc": float(w.get("accuracy", 0.0)),
            "win_f1": float(w.get("f1", 0.0)),
            "win_rec": float(w.get("recall", 0.0)),
            "file_acc": float(f.get("accuracy", 0.0)),
            "file_f1": float(f.get("f1", 0.0)),
            "file_rec": float(f.get("recall", 0.0)),
            # top-k 窗投票口径（新版 run_train 才有；旧结果文件没有则为 0）
            "file_acc_topk": float(ft.get("accuracy", 0.0)),
            "file_f1_topk": float(ft.get("f1", 0.0)),
            # top-k + "坠落后静止"确认（主口径；规则未启用时与 top-k 相同）
            "file_acc_topk_c": float(ftc.get("accuracy", ft.get("accuracy", 0.0))),
            "file_f1_topk_c": float(ftc.get("f1", ft.get("f1", 0.0))),
        }
    return row


def print_table(rows: list[dict], weights: dict[str, int] | None = None) -> None:
    head = f"{'测试人':<8}" + "".join(
        f"{name + ' AUC/Acc/F1':>24}" for _, name, _ in CONFIGS
    )
    print("\n" + "=" * len(head))
    print("  音频消融结果（窗级 AUC / 文件级 Acc / 文件级 F1）")
    print("=" * len(head))
    print(head)
    print("-" * len(head))
    for r in rows:
        line = f"{r['user']:<8}"
        for tag, _, _ in CONFIGS:
            d = r[tag]
            line += f"{d['auc']:>10.3f}/{d['file_acc']:.3f}/{d['file_f1']:.3f}"
        print(line)
    print("-" * len(head))

    w = np.array([float((weights or {}).get(r["user"], 1)) for r in rows])
    for label in ("均值", "加权均值"):
        line = f"{label:<8}"
        for tag, _, _ in CONFIGS:
            auc = np.average([r[tag]["auc"] for r in rows], weights=w)
            acc = np.average([r[tag]["file_acc"] for r in rows], weights=w)
            f1 = np.average([r[tag]["file_f1"] for r in rows], weights=w)
            line += f"{auc:>10.3f}/{acc:.3f}/{f1:.3f}"
        print(line)
    line = f"{'极差':<8}"
    for tag, _, _ in CONFIGS:
        auc = np.ptp([r[tag]["auc"] for r in rows])
        acc = np.ptp([r[tag]["file_acc"] for r in rows])
        f1 = np.ptp([r[tag]["file_f1"] for r in rows])
        line += f"{auc:>10.3f}/{acc:.3f}/{f1:.3f}"
    print(line)

    # 主口径补充表：top-k + 静止确认
    print("\n  主口径（top-k 窗投票 + 坠落后静止确认，文件级 Acc/F1）：")
    for r in rows:
        line = f"  {r['user']:<8}"
        for tag, _, _ in CONFIGS:
            line += f"  {r[tag]['file_acc_topk_c']:.3f}/{r[tag]['file_f1_topk_c']:.3f}"
        print(line)
    line = f"  {'均值':<8}"
    for tag, _, _ in CONFIGS:
        acc = np.average([r[tag]["file_acc_topk_c"] for r in rows], weights=w)
        f1 = np.average([r[tag]["file_f1_topk_c"] for r in rows], weights=w)
        line += f"  {acc:.3f}/{f1:.3f}"
    print(line)


def plot(rows: list[dict], name: str = "audio_ablation.png") -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    users = [r["user"] for r in rows]
    x = np.arange(len(users))
    w = 0.26
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    for ax, key, title in ((axes[0], "auc", "Window AUC (cross-person)"),
                           (axes[1], "file_f1", "File-level F1")):
        for i, (tag, label, color) in enumerate(CONFIGS):
            vals = [r[tag][key] for r in rows]
            bars = ax.bar(x + (i - 1) * w, vals, width=w, label=label, color=color)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                        ha="center", fontsize=6.5)
        ax.set_xticks(x)
        ax.set_xticklabels(users)
        ax.set_ylim(0, 1.12)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("score")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Ablation: CSI-only vs CSI+Audio vs Audio-only", fontsize=11)
    fig.tight_layout()
    out = config.FIG_DIR / name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    args = parse_args()
    rows = [run_one(u, args) for u in args.test_users]
    counts = file_counts(args.data_dir)
    print_table(rows, counts)

    name = "audio_ablation" + ("_lopo" if args.lopo else "") + args.tag
    out_json = config.OUTPUT_DIR / f"{name}.json"
    out_json.write_text(json.dumps(
        {"val_users": args.val_users, "lopo": args.lopo,
         "data_dir": args.data_dir, "file_counts": counts, "rows": rows},
        ensure_ascii=False, indent=2), encoding="utf-8")
    fig = plot(rows, f"{name}.png")
    print(f"\n结果已保存: {out_json}\n图: {fig}")


if __name__ == "__main__":
    main()
