"""模型层：随机森林 / 逻辑回归 训练、评估、阈值选择、持久化。

跌倒检测在 CSI 特征空间通常用集成模型(随机森林)即可取得较好结果，
逻辑回归作为可解释基线。二者都带 class_weight='balanced' 以应对
样本不平衡。
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import LR_C, LR_MAX_ITER, RF_MAX_DEPTH, RF_N_ESTIMATORS, RANDOM_STATE


def train_rf(X, y, seed=RANDOM_STATE) -> RandomForestClassifier:
    """训练随机森林分类器。"""
    clf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(X, y)
    return clf


def train_lr(X, y, seed=RANDOM_STATE) -> Pipeline:
    """训练带标准化与 L2 正则的逻辑回归（可解释基线）。"""
    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    C=LR_C,
                    max_iter=LR_MAX_ITER,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )
    clf.fit(X, y)
    return clf


def predict_proba_pos(clf, X: np.ndarray) -> np.ndarray:
    """返回正类(跌倒)概率。统一不同分类器的接口。"""
    return np.asarray(clf.predict_proba(X)[:, 1], dtype=np.float64)


def evaluate(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5):
    """在给定阈值下计算指标字典（含混淆矩阵）。"""
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
    """在训练/验证集上用 Youden-J 准则选择使 TPR-FPR 最大的决策阈值。"""
    if np.unique(y_true).size < 2:  # 仅一类时无法计算 ROC，退回默认阈值
        return 0.5
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    if len(thr) == 0:
        return 0.5
    j = tpr - fpr
    best = int(np.argmax(j))
    # thr[best] 表示"概率大于该值判正"，返回可直接用于 >= 判断的值
    return float(thr[best]) if 0 <= best < len(thr) else 0.5


def majority_vote_indices(groups: np.ndarray, labels: np.ndarray):
    """对每个样本文件内的窗标签做多数投票 -> (样本文件索引, 文件标签)。"""
    import pandas as pd

    df = pd.DataFrame({"g": groups, "y": labels})
    # round() 是银行家舍入(0.5->0)，与全项目 ">=0.5 判正" 的约定不一致，故不用
    votes = df.groupby("g")["y"].agg(lambda s: int(float(s.mean()) >= 0.5))
    return votes.index.values, votes.values


def save_model(model, feature_names_list, path_dir: Path, tag: str):
    """保存模型 + 特征名。"""
    path_dir.mkdir(parents=True, exist_ok=True)
    model_path = path_dir / f"{tag}.joblib"
    joblib.dump(model, model_path)
    with (path_dir / f"{tag}_features.json").open("w", encoding="utf-8") as f:
        json.dump(feature_names_list, f)
    return model_path


def load_model(path_dir: Path, tag: str):
    model = joblib.load(path_dir / f"{tag}.joblib")
    with (path_dir / f"{tag}_features.json").open("r", encoding="utf-8") as f:
        names = json.load(f)
    return model, names
