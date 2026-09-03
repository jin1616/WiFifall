"""数据加载：扫描 IDxx/<activity>/*.csv 并构造成样本清单(DataFrame)。

Widar3.0 风格的目录布局（此处为按"人/动作"整理的切片版本）：
    <data_root>/ID0/fall/2023-11-12_21-21-03-546_104_1.csv
    <data_root>/ID0/walk/2023-11-12_21-16-58-889_104_1.csv
    ...
每个 CSV = 一个连续动作片段，含逐包 CSI（行）。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import numpy as np

from .config import DATA_DIR, NEGATIVE_ACTIVITIES, POSITIVE_ACTIVITIES, USER_PREFIX


def _label_of(activity: str) -> int:
    """活动目录名 -> 二分类标签 (1=跌倒, 0=非跌倒)。"""
    return 1 if activity in POSITIVE_ACTIVITIES else 0


def scan_dataset(data_dir: Path | str = DATA_DIR) -> pd.DataFrame:
    """递归扫描数据集，返回每条样本文件的一行记录。

    列: user(人员ID), activity(活动名), label(0/1), path(样本绝对路径)
    """
    data_dir = Path(data_dir)
    rows = []
    if not data_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    for user_dir in sorted(data_dir.iterdir()):
        if not user_dir.is_dir() or not user_dir.name.startswith(USER_PREFIX):
            continue  # 跳过非用户目录（例如 wifalldemo/、outputs/）
        for act_dir in sorted(user_dir.iterdir()):
            if not act_dir.is_dir():
                continue
            activity = act_dir.name.lower()
            if activity not in POSITIVE_ACTIVITIES | NEGATIVE_ACTIVITIES:
                continue  # 忽略未知活动目录
            for csv_file in sorted(act_dir.glob("*.csv")):
                rows.append(
                    {
                        "user": user_dir.name,
                        "activity": activity,
                        "label": _label_of(activity),
                        "path": str(csv_file),
                    }
                )

    if not rows:
        raise FileNotFoundError(
            f"在 {data_dir} 下未找到任何符合 IDxx/<activity>/*.csv 的数据"
        )

    df = pd.DataFrame(rows, columns=["user", "activity", "label", "path"])
    return df


def balanced_file_index(df: pd.DataFrame) -> np.ndarray:
    """按"每用户每活动"采样数中位数截断，均衡不同受试者的样本量。

    背景：个别受试者可能录了远多于常人的轮次（例如 ID2 ≈ 其他人的 20 倍）。
    直接训练会让该用户的环境/体型分布主导模型，从而损害对其他人的泛化。
    策略：对每个 (user, activity)，若其文件数高于全用户中位数，则只保留
    排序靠前的 cap 个文件，使各用户对模型的贡献相当。

    返回: 需要保留的样本在原始 DataFrame 中的行号(Index)。
    """
    cap: dict[str, int] = {}
    for act in sorted(df["activity"].unique()):
        per_user = df[df["activity"] == act].groupby("user").size()
        cap[act] = int(np.median(per_user.values))

    keep: list = []
    for user in sorted(df["user"].unique()):
        for act, c in cap.items():
            sub = df[(df["user"] == user) & (df["activity"] == act)]
            keep.append(sub.index.values[:c])
    if not keep:
        return np.asarray([], dtype=int)
    return np.concatenate(keep)


def person_split(
    df: pd.DataFrame,
    test_users: tuple[str, ...] = ("ID8", "ID9"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按人员 ID 划分训练/测试，测试集用户不参与训练。

    人员级划分保证了评估结果的"跨人泛化"意义。
    """
    # 注意: 不要 reset_index —— record_ids 依赖 scan_dataset 的原始行号
    is_test = df["user"].isin(set(test_users))
    return df.loc[~is_test], df.loc[is_test]


def summary(df: pd.DataFrame) -> str:
    """生成简洁的数据集统计文本，便于日志输出。"""
    lines = [f"  样本文件总数 : {len(df)}"]
    lines.append(f"  用户数       : {df['user'].nunique()} ({', '.join(sorted(df['user'].unique()))})")
    cross = (
        df.groupby(["user", "label"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={0: "non-fall", 1: "fall"})
    )
    lines.append("  每用户类别分布:")
    for user, row in cross.iterrows():
        lines.append(f"    {user}: non-fall={int(row['non-fall']):>3}  fall={int(row['fall']):>3}")
    total_fall = int((df["label"] == 1).sum())
    lines.append(f"  正例(fall)={total_fall}  负例(non-fall)={len(df) - total_fall}")
    return "\n".join(lines)
