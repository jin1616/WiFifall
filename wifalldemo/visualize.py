"""可视化：CSI 热力图、特征对比、混淆矩阵、ROC、检测时间线。"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无显示环境下也能出图
import matplotlib.pyplot as plt
import numpy as np

from .config import ACTIVITY_LABELS, FIG_DIR


def _setup_fonts() -> None:
    """为中文标题选择可用字体（找不到时自动回落 DejaVu）。"""
    try:
        from matplotlib import font_manager

        installed = {f.name for f in font_manager.fontManager.ttflist}
        for cand in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                     "PingFang SC", "WenQuanYi Zen Hei"):
            if cand in installed:
                plt.rcParams["font.sans-serif"] = [cand, "DejaVu Sans"]
                break
        plt.rcParams["axes.unicode_minus"] = False  # 避免中文环境负号变方块
    except Exception:  # 字体探测失败不影响出图
        pass


_setup_fonts()
plt.rcParams.update(
    {
        "figure.dpi": 110,
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def save_fig(fig, name: str, subdir: Path | str = FIG_DIR) -> Path:
    """保存图形并关闭，返回文件路径。"""
    out = Path(subdir) / name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_csi_sample(
    amp: np.ndarray,
    name: str = "csi_heatmap.png",
    title: str = "CSI Amplitude (52 subcarriers over time)",
) -> Path:
    """展示单个样本的 CSI 幅度热力图 + 子载波均值波形。"""
    fig, axes = plt.subplots(
        2, 1, figsize=(11, 6), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )
    im = axes[0].imshow(
        amp.T,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        interpolation="nearest",
    )
    axes[0].set_ylabel("Subcarrier index")
    axes[0].set_title(title)
    fig.colorbar(im, ax=axes[0], label="|CSI|")
    t = np.arange(amp.shape[0]) / 83.0
    axes[1].plot(t, amp.mean(axis=1), lw=0.8, color="tab:blue")
    axes[1].set_ylabel("mean |CSI|")
    axes[1].set_xlabel("time (s)")
    axes[1].set_title("Mean CSI trace")
    fig.tight_layout()
    return save_fig(fig, name)


def plot_confusion(cm: np.ndarray, name: str, title: str = "Confusion Matrix") -> Path:
    """混淆矩阵热力图。"""
    fig, ax = plt.subplots(figsize=(5, 4.4))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, shrink=0.85)
    classes = [ACTIVITY_LABELS[0], ACTIVITY_LABELS[1]]
    ax.set_xticks([0, 1])
    ax.set_xticklabels(classes)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    total = int(cm.sum())
    for i in range(2):
        for j in range(2):
            v = int(cm[i, j])
            ax.text(j, i, f"{v}\n({v / total * 100:.1f}%)" if total else str(v),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() * 0.5 else "black")
    fig.tight_layout()
    return save_fig(fig, name)


def plot_roc(fpr: np.ndarray, tpr: np.ndarray, auc: float, name: str = "roc.png") -> Path:
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot(fpr, tpr, color="tab:red", lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC — window-level (fall vs non-fall)")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    return save_fig(fig, name)


def plot_feature_importance(
    importances: np.ndarray,
    names: list[str],
    name: str = "feature_importance.png",
    top_k: int = 20,
) -> Path:
    """随机森林特征重要性（top_k）。"""
    order = np.argsort(importances)[::-1][:top_k]
    fig, ax = plt.subplots(figsize=(8, max(3, top_k * 0.35)))
    ax.barh(np.arange(top_k)[::-1], importances[order][::-1], color="tab:blue")
    ax.set_yticks(np.arange(top_k)[::-1])
    ax.set_yticklabels([names[i] for i in order[::-1]], fontsize=8)
    ax.set_xlabel("Importance (Gini)")
    ax.set_title("Top CSI features for fall detection")
    ax.grid(axis="x")
    fig.tight_layout()
    return save_fig(fig, name)


def plot_detection_timeline(
    times: np.ndarray,
    prob_fall: np.ndarray,
    gt_label: int,
    threshold: float,
    fs: float,
    name: str = "detection_timeline.png",
    title: str = "",
) -> Path:
    """展示一条样本被逐窗打分的结果（离线/流式通用）。

    times:     每个滑窗中心时刻(秒)
    prob_fall: 每窗预测为跌倒的概率
    gt_label:  该样本真实标签(0/1)
    """
    fig, ax = plt.subplots(figsize=(10, 3.4))
    color = "tab:red" if gt_label == 1 else "tab:green"
    ax.plot(times, prob_fall, "-o", ms=4, lw=1.5, color=color,
            label=f"true={ACTIVITY_LABELS[gt_label]}")
    ax.axhline(threshold, color="black", ls="--", lw=1,
               label=f"decision threshold = {threshold:.3f}")
    ax.fill_between(times, threshold, prob_fall, where=prob_fall >= threshold,
                    color="red", alpha=0.12, interpolate=True)
    ax.set_xlabel(f"window center time (s)  (~{fs:.0f} packets/s)")
    ax.set_ylabel("P(fall)")
    ax.set_ylim(0, 1.05)
    ax.set_title(title or f"Fall probability over a {len(times)}-window sequence")
    ax.legend(loc="center left", fontsize=8)
    fig.tight_layout()
    return save_fig(fig, name)


def plot_feature_compare(
    X: np.ndarray, y: np.ndarray, names: list[str],
    top: int = 6, name: str = "feature_compare.png",
) -> Path:
    """在测试集上展示区分度最强的特征中 fall / non-fall 分布对比。"""
    # 用均值差/散布比挑 top 特征: |mean1-mean0| / (std1+std0)
    m1 = X[y == 1].mean(axis=0)
    m0 = X[y == 0].mean(axis=0)
    s1 = X[y == 1].std(axis=0) + 1e-6
    s0 = X[y == 0].std(axis=0) + 1e-6
    score = np.abs(m1 - m0) / (s1 + s0)
    idx = np.argsort(score)[::-1][:top]
    fig, axes = plt.subplots(1, top, figsize=(min(top * 2.6, 15), 3.2))
    if top == 1:
        axes = [axes]
    for ax, i in zip(axes, idx):
        ax.hist(X[y == 0, i], bins=30, alpha=0.55, label="non-fall", color="tab:green")
        ax.hist(X[y == 1, i], bins=30, alpha=0.55, label="fall", color="tab:red")
        ax.set_title(names[i], fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6)
    fig.suptitle("Feature distribution: fall vs non-fall (test users)")
    fig.tight_layout()
    return save_fig(fig, name)


def plot_model_compare(labels: list[str], accs: list[float], name: str = "model_compare.png"):
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    colors = plt.get_cmap("tab10").colors[: max(len(labels), 1)]
    bars = ax.bar(labels, accs, color=colors)
    for b, v in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.4f}",
                ha="center", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Window accuracy (test users)")
    ax.set_title("Classifier comparison")
    fig.tight_layout()
    return save_fig(fig, name)
