"""模型层：纯 PyTorch 实现的三种深度学习分类器 + 评估/阈值/持久化工具。

历史版本曾同时承担随机森林和逻辑回归两个 sklearn 模型的训练/评估，目前
已完全切到 PyTorch: 三种 DL 模型共享 ``predict_proba(X) -> (N, 2)`` 的
sklearn 风格接口，与 :mod:`.cnn` 实现完全打通。

模型清单
--------
``train_mlp``     — 三层全连接 MLP (256→128→64)，替换原 RandomForest 主力位
``train_dlsmall`` — 单层薄 MLP (64 单元)，替换原 LogisticRegression 基线位
``train_cnn``     — 1D-CNN (3 conv blocks + AdaptiveAvgPool + 2 层 FC) 见
                    :mod:`.cnn`，输入是原始 (B, 128, 52) 张量而非手工特征

所有模型自动用 ``class_weight='balanced'`` 等价的样本权重，正负样本同等参与
损失计算。
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from . import config
from .config import (
    DL_BATCH_SIZE,
    DL_DEVICE,
    DL_EPOCHS,
    DL_LR,
    DL_PATIENCE,
    DL_SEED,
    DL_WEIGHT_DECAY,
    DLS_DROPOUT,
    DLS_EPOCHS,
    DLS_HIDDEN,
    DLS_PATIENCE,
    MLP_DROPOUT,
    MLP_HIDDEN,
)
from .cnn import CNNClassifier, _split_fit_val, train_cnn  # re-export


# ----------------------------------------------------------------------------
# 网络定义 (FC-only 模型)
# ----------------------------------------------------------------------------
class MLP(nn.Module):
    """中等规模 MLP: ``MLP_HIDDEN`` 三层 + Dropout。"""

    def __init__(
        self,
        in_dim: int,
        hidden: tuple[int, ...] = MLP_HIDDEN,
        n_classes: int = 2,
        dropout: float = MLP_DROPOUT,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DLSmall(nn.Module):
    """薄 MLP: 单层隐藏 ``DLS_HIDDEN``。"""

    def __init__(
        self,
        in_dim: int,
        hidden: tuple[int, ...] = DLS_HIDDEN,
        n_classes: int = 2,
        dropout: float = DLS_DROPOUT,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ----------------------------------------------------------------------------
# sklearn 风格兼容包装器
# ----------------------------------------------------------------------------
class TorchCompat:
    """sklearn 风格包装: ``predict_proba(X) -> (N, 2)``，与 CNNClassifier 同形。

    训练时按 :class:`torch.nn.Module` 的标准流程更新参数；推理时走
    ``predict_proba`` → softmax 输出。
    """

    def __init__(self, model: nn.Module, device: str = DL_DEVICE) -> None:
        self.model = model
        self.device = device

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 256):
                batch = torch.as_tensor(X[i : i + 256], dtype=torch.float32).to(self.device)
                logits = self.model(batch)
                proba = F.softmax(logits, dim=-1).cpu().numpy()
                out.append(proba)
        return np.concatenate(out, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def state_dict_copy(self) -> dict:
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}


# ----------------------------------------------------------------------------
# 训练流程 (FC 模型通用)
# ----------------------------------------------------------------------------
def _make_class_weights(y: np.ndarray, device: str) -> torch.Tensor:
    """等价于 ``class_weight='balanced'`` 的样本权重。"""
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    n = pos + neg
    w0 = n / (2.0 * max(neg, 1.0))
    w1 = n / (2.0 * max(pos, 1.0))
    return torch.tensor([w0, w1], device=device, dtype=torch.float32)


def _fit_torch_model(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_epochs: int = DL_EPOCHS,
    batch_size: int = DL_BATCH_SIZE,
    lr: float = DL_LR,
    weight_decay: float = DL_WEIGHT_DECAY,
    patience: int = DL_PATIENCE,
    device: str = DL_DEVICE,
    seed: int = DL_SEED,
    val_X: np.ndarray | None = None,
    val_y: np.ndarray | None = None,
    groups: np.ndarray | None = None,
    verbose: bool = False,
    tag: str = "mlp",
):
    """通用 PyTorch 二分类训练。

    早停验证集优先级：外部传入 ``val_X/val_y`` > 按 ``groups`` 分组内部切
    10% > 窗级随机切 10%（有泄漏，仅兼容旧调用）。
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if val_X is None:
        fit_idx, val_idx = _split_fit_val(y, seed, groups)
        X_fit, y_fit = X[fit_idx], y[fit_idx]
        X_val, y_val = X[val_idx], y[val_idx]
    else:
        X_fit, y_fit = X, y
        X_val, y_val = val_X, val_y

    model = model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=n_epochs)
    crit = nn.CrossEntropyLoss(weight=_make_class_weights(y_fit, device))

    train_dl = DataLoader(
        TensorDataset(
            torch.as_tensor(X_fit, dtype=torch.float32),
            torch.as_tensor(y_fit, dtype=torch.long),
        ),
        batch_size=batch_size, shuffle=True,
    )
    val_dl = DataLoader(
        TensorDataset(
            torch.as_tensor(X_val, dtype=torch.float32),
            torch.as_tensor(y_val, dtype=torch.long),
        ),
        batch_size=batch_size, shuffle=False,
    )

    best_val_loss = float("inf")
    best_state = None
    bad = 0

    for ep in range(n_epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            optim.step()
            train_loss += float(loss.item()) * xb.size(0)
        sched.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += float(crit(model(xb), yb).item()) * xb.size(0)
        val_loss /= max(len(y_val), 1)

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if verbose:
            print(
                f"  [{tag}] epoch {ep+1:3d}/{n_epochs}  "
                f"train_loss={train_loss/len(y_fit):.4f}  "
                f"val_loss={val_loss:.4f}  "
                f"lr={sched.get_last_lr()[0]:.2e}"
            )

        if bad >= patience:
            if verbose:
                print(f"  [{tag}] early stop at epoch {ep+1}, best_val_loss={best_val_loss:.4f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return TorchCompat(model, device=device)


# ----------------------------------------------------------------------------
# 公共训练入口
# ----------------------------------------------------------------------------
def train_mlp(
    X: np.ndarray,
    y: np.ndarray,
    *,
    val_X: np.ndarray | None = None,
    val_y: np.ndarray | None = None,
    groups: np.ndarray | None = None,
    seed: int = DL_SEED,
    verbose: bool = False,
) -> TorchCompat:
    """训练一个 3 层 MLP，替换原 RandomForest 主力模型。

    输入 ``X`` 形状 (N, F=58) 手工特征；输出 :class:`TorchCompat` 包装。
    ``groups`` 是每条窗的样本文件 id，用于无泄漏的内部早停切分。
    """
    model = MLP(in_dim=X.shape[1])
    return _fit_torch_model(
        model, X, y,
        val_X=val_X, val_y=val_y, groups=groups,
        seed=seed, verbose=verbose, tag="mlp",
    )


def train_dlsmall(
    X: np.ndarray,
    y: np.ndarray,
    *,
    val_X: np.ndarray | None = None,
    val_y: np.ndarray | None = None,
    groups: np.ndarray | None = None,
    seed: int = DL_SEED,
    verbose: bool = False,
) -> TorchCompat:
    """训练 (128,64) 双层薄 MLP，替换原 LogisticRegression 基线模型。"""
    model = DLSmall(in_dim=X.shape[1])
    return _fit_torch_model(
        model, X, y,
        n_epochs=DLS_EPOCHS, patience=DLS_PATIENCE,
        val_X=val_X, val_y=val_y, groups=groups,
        seed=seed, verbose=verbose, tag="dlsmall",
    )


# ----------------------------------------------------------------------------
# 评估 / 阈值 (与 sklearn 版本完全一致，metrics.json 字段名保持)
# ----------------------------------------------------------------------------
def predict_proba_pos(clf, X: np.ndarray) -> np.ndarray:
    """返回正类(跌倒)概率；统一 sklearn 与 :class:`TorchCompat` 接口。"""
    proba = clf.predict_proba(X)
    return np.asarray(proba[:, 1], dtype=np.float64)


def evaluate(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5):
    """在给定阈值下计算指标字典 (含混淆矩阵)。"""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": (
            float(roc_auc_score(y_true, y_prob))
            if 0 < int(y_true.sum()) < y_true.size
            else 0.0
        ),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Youden-J 阈值选择 (与 sklearn 版本对齐)。

    两个保护:
      * sklearn 的 ``roc_curve`` 把 ``thresholds[0]`` 置为 ``inf``（对应"全判
        负类"的工作点）。模型在验证集上没有区分度时 argmax(J) 会落到这个点，
        返回 inf 意味着部署后 ``prob >= inf`` 恒假 —— 全判非跌倒且无报警。
      * 最大 J <= 0 说明区分度不优于随机，此时任何"最优"阈值都是贴边界的
        假象，退回 0.5 至少不会极端偏科（指标打印里仍能看到 AUC 不行）。
    """
    if np.unique(y_true).size < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    if len(thr) == 0:
        return 0.5
    finite = np.isfinite(thr)
    if not finite.any():
        return 0.5
    j = np.where(finite, tpr - fpr, -np.inf)
    best = int(np.argmax(j))
    if j[best] <= 0:
        return 0.5
    return float(thr[best])


def majority_vote_indices(groups: np.ndarray, labels: np.ndarray):
    """对每个样本文件内的窗标签做多数投票 -> (样本文件索引, 文件标签)。"""
    import pandas as pd

    df = pd.DataFrame({"g": groups, "y": labels})
    votes = df.groupby("g")["y"].agg(lambda s: int(float(s.mean()) >= 0.5))
    return votes.index.values, votes.values


# ----------------------------------------------------------------------------
# 持久化 (兼容 sklearn joblib 与 PyTorch state_dict)
# ----------------------------------------------------------------------------
def save_model(model, feature_names_list, path_dir: Path, tag: str):
    """保存模型 + 特征名。

    * ``TorchCompat`` (MLP / DLsmall) → joblib 整个对象 (state_dict 在内)
    * ``CNNClassifier`` → ``.pt`` 文件 (torch.save) + 没有"特征名"概念，
      写空列表占位
    * 其它 (sklearn-style 兼容接口) → joblib.dump
    """
    path_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(model, CNNClassifier):
        from .cnn import save_cnn
        save_cnn(model, path_dir, tag=tag)
        _purge_other_format(path_dir, tag, "pt")
        with (path_dir / f"{tag}_features.json").open("w", encoding="utf-8") as f:
            json.dump(list(feature_names_list or []), f, ensure_ascii=False)
        return path_dir / f"{tag}.pt"

    # MLP / DLsmall (TorchCompat) 与一切具有 .predict_proba 的模型 → joblib
    model_path = path_dir / f"{tag}.joblib"
    joblib.dump(model, model_path)
    _purge_other_format(path_dir, tag, "joblib")
    with (path_dir / f"{tag}_features.json").open("w", encoding="utf-8") as f:
        json.dump(list(feature_names_list or []), f, ensure_ascii=False)
    return model_path


def _purge_other_format(path_dir: Path, tag: str, keep_ext: str) -> None:
    """删除同一 tag 下另一种格式的历史文件。

    两种格式（CNN 的 .pt 与 MLP 家族的 .joblib）本不该共存：主模型从 CNN
    换成 MLP 时，残留的 ``primary.pt`` 会让 :func:`load_model`（它优先找 .pt）
    加载上一次的模型，而 ``primary_features.json`` 已经是新模型的特征名 ——
    结果是"加载了一个和特征维度对不上的旧模型"，且只在推理时才炸。
    """
    for ext in ("pt", "joblib"):
        if ext == keep_ext:
            continue
        stale = path_dir / f"{tag}.{ext}"
        if stale.exists():
            stale.unlink()


def load_model(path_dir: Path, tag: str):
    """加载模型 + 特征名。"""
    p = Path(path_dir)
    pt_p = p / f"{tag}.pt"
    if pt_p.exists():
        from .cnn import load_cnn
        model = load_cnn(p, tag=tag)
        with (p / f"{tag}_features.json").open("r", encoding="utf-8") as f:
            names = json.load(f)
        return model, names
    model = joblib.load(p / f"{tag}.joblib")
    with (p / f"{tag}_features.json").open("r", encoding="utf-8") as f:
        names = json.load(f)
    return model, names


def norm_config_snapshot() -> dict:
    """当前 config 里的归一化设置——训练时用它写进 ``metrics.json``。"""
    return {
        "enabled": bool(config.NORMALIZE_PER_SUBCARRIER),
        "window_s": float(config.NORM_WINDOW_S),
        "causal": bool(config.NORM_CAUSAL),
    }


def load_norm_config(output_dir=None, verbose: bool = True) -> dict:
    """取**训练时**的归一化口径，而不是当前 config。

    归一化是模型输入表示的一部分：训练用因果窗、推理用居中窗（或反过来），模型
    吃到的就是它没见过的分布。所以这个口径必须像 ``threshold`` 一样从
    ``metrics.json`` 读 —— **改了 config 不应该改变已训好模型的推理行为**。

    返回 ``{"enabled", "window_s", "causal", "source"}``。``metrics.json`` 里没有
    这个键（老版本训练的产物）时退回 config 当前值，``source`` 标成 ``"config"``
    并打警告，调用方可以据此提示用户重训。
    """
    snap = norm_config_snapshot()
    p = Path(output_dir if output_dir is not None else config.OUTPUT_DIR) / "metrics.json"
    if p.exists():
        meta = json.loads(p.read_text(encoding="utf-8"))
        saved = meta.get("normalization")
        if isinstance(saved, dict):
            return {
                "enabled": bool(saved.get("enabled", snap["enabled"])),
                "window_s": float(saved.get("window_s", snap["window_s"])),
                "causal": bool(saved.get("causal", snap["causal"])),
                "source": "metrics.json",
            }
    snap["source"] = "config"
    if verbose:
        print(
            f"[警告] {p.name} 里没有归一化口径记录，退回 config 当前值 "
            f"(enabled={snap['enabled']}, window_s={snap['window_s']}s, "
            f"causal={snap['causal']}) —— 这未必是模型训练时的口径，建议重训一次"
        )
    return snap


# 显式把 CNN 训练入口再导出 (run_train 中按需引入)
__all__ = [
    "CNNClassifier",
    "MLP",
    "DLSmall",
    "TorchCompat",
    "train_cnn",
    "train_mlp",
    "train_dlsmall",
    "predict_proba_pos",
    "evaluate",
    "youden_threshold",
    "majority_vote_indices",
    "save_model",
    "load_model",
    "norm_config_snapshot",
    "load_norm_config",
]
