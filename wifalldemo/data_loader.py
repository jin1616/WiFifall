"""数据加载：扫描数据源并构造成样本清单(DataFrame)。

支持的两种数据源 (按 ``scan_dataset`` 的入参自动派发)：

1. **目录**：包含 ``ID*/<activity>/*.csv`` 的文件夹 —— 旧用法，递归扫描原始 CSI。
2. **pickle**：``run_build_sequence.py`` 生成的 ``data_sequence.pkl``，推荐用法；
   单次读盘一次到位，下游统一通过 ``<pkl_path>::<sample_idx>`` 编码的 path
   字段引用样本，由 ``preprocessing.load_sample_amplitude`` /
   ``dataset._process_one`` 自动识别并取数据。

返回的 DataFrame 始终保持相同 schema::

    user        人员名 (如 "ID0")
    activity    活动名小写 (如 "fall")
    label       0/1 (1=跌倒, 0=非跌倒)
    path        对原始 CSV 是绝对路径；对 pickle 是 ``<pkl_abs>::<idx>`` 编码
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

import pandas as pd

from .config import DATA_DIR, NEGATIVE_ACTIVITIES, OUTPUT_DIR, POSITIVE_ACTIVITIES, USER_PREFIX


# 默认 pickle 路径：与 run_build_sequence.py 的 --output 默认值保持一致
DEFAULT_PICKLE = OUTPUT_DIR / "data_sequence.pkl"

# 采样目录里和 CSI 样本并排放着的非 CSI 文件，扫描时必须跳过。
# 采集工具（csi_audio_sync）每条样本会产出四件套，其中 *_audio_env.csv 是逐行声音
# 能量包络（t_pc, csi_row, rms, dbfs），不是 52 路 CSI。不跳过的话它会被当成一条
# 正常样本吃进训练集——列数对不上，特征全是脏数，而且样本量凭空翻倍。
NON_CSI_SUFFIXES = ("_audio_env.csv",)


def _is_csi_sample(path: Path) -> bool:
    """判断一个 CSV 是不是 CSI 样本（排除音频包络等附属文件）。

    除 ``NON_CSI_SUFFIXES`` 外，``_`` 前缀的文件（如摆位探测 ``_probe.csv``）
    也不参与训练——采集包那边的体检脚本同样按这个约定跳过。
    """
    name = path.name.lower()
    if name.startswith("_"):
        return False
    return not any(name.endswith(s) for s in NON_CSI_SUFFIXES)


def _label_of(activity: str) -> int:
    """活动目录名 -> 二分类标签 (1=跌倒, 0=非跌倒)。"""
    return 1 if activity in POSITIVE_ACTIVITIES else 0


def _scan_csv_directory(data_dir: Path | str) -> pd.DataFrame:
    """递归扫描 ``IDxx/<activity>/*.csv``，返回每条样本文件的一行记录。"""
    data_dir = Path(data_dir)
    rows: list[dict] = []
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
                if not _is_csi_sample(csv_file):
                    continue  # 跳过 *_audio_env.csv 等非 CSI 附属文件
                rows.append(
                    {
                        "user": user_dir.name,
                        "activity": activity,
                        "label": _label_of(activity),
                        "path": str(csv_file.resolve()),
                    }
                )

    if not rows:
        raise FileNotFoundError(
            f"在 {data_dir} 下未找到任何符合 IDxx/<activity>/*.csv 的数据"
        )

    df = pd.DataFrame(rows, columns=["user", "activity", "label", "path"])
    return df


# 兜底：当 pickle 没存 action_name/user_name 时，按 process1 的字典序回填。
# 注意 Windows 的 Path 比较是大小写不敏感的，给出 ('fall','Jump',...)，而
# Linux/macOS 是大小写敏感 ('Jump','fall',...)；这里用 str.lower 做大小写不敏感
# 排序，保证 fallback 在两个平台上结果一致。
_ACTION_FALLBACK_BY_ID = tuple(
    sorted(("fall", "Jump", "sit", "stand", "walk"), key=str.lower)
)


def _resolve_user_name(item: dict, people_id: int) -> str:
    name = item.get("user_name")
    if isinstance(name, str) and name:
        return name
    return f"ID{people_id}"


def _resolve_action_name(item: dict, action_id: int) -> str:
    name = item.get("action_name")
    if isinstance(name, str) and name:
        return name
    if 0 <= action_id < len(_ACTION_FALLBACK_BY_ID):
        return _ACTION_FALLBACK_BY_ID[action_id]
    return f"action_{action_id}"


def scan_pickle(pkl_path: str | Path) -> pd.DataFrame:
    """从 ``data_sequence.pkl`` 加载样本清单 (推荐用法)。

    路径编码 ``<abs_pkl>::<idx>`` 由 ``preprocessing.load_sample_amplitude``
    通过 ``rpartition('::')`` 解析；这里用绝对路径避免 ``::`` 出现在
    pkl 路径中导致歧义 (双冒号不会出现在合法文件系统路径里)。
    """
    pkl_path = Path(pkl_path)
    if not pkl_path.exists():
        raise FileNotFoundError(f"pickle 数据文件不存在: {pkl_path}")
    abs_pkl = str(pkl_path.resolve())

    with pkl_path.open("rb") as f:
        seq = pickle.load(f)
    if not isinstance(seq, list) or not seq:
        raise ValueError(f"{pkl_path} 内容不是非空 list，无法解析样本")

    rows: list[dict] = []
    for idx, item in enumerate(seq):
        user = _resolve_user_name(item, int(item.get("people", 0)))
        activity = _resolve_action_name(item, int(item.get("action", 0))).lower()
        rows.append(
            {
                "user": user,
                "activity": activity,
                "label": _label_of(activity),
                "path": f"{abs_pkl}::{idx}",
            }
        )

    df = pd.DataFrame(rows, columns=["user", "activity", "label", "path"])
    if df.empty:
        raise FileNotFoundError(f"在 {pkl_path} 中未解析到任何样本")
    return df


def scan_dataset(source: str | Path | None = None) -> pd.DataFrame:
    """扫描数据集并返回样本清单 DataFrame (与旧版签名兼容)。

    入参 ``source`` 行为：
      * ``None``（默认）: 优先尝试 ``outputs/data_sequence.pkl`` (推荐新流程)；
        若不存在则退回到 ``DATA_DIR`` 下的 CSV 目录扫描。
      * 目录路径: 走原始 CSV 扫描。
      * ``.pkl`` 文件路径: 走 pickle 加载。

    下游统一以 ``path`` 列引用样本：CSV 走文件路径，pickle 走
    ``<abs_pkl>::<idx>`` 编码，二者都由 ``preprocessing.load_sample_amplitude``
    透明分发。
    """
    if source is None:
        candidate = DEFAULT_PICKLE if DEFAULT_PICKLE.exists() else DATA_DIR
    else:
        candidate = Path(source)

    if candidate.is_dir():
        return _scan_csv_directory(candidate)
    if candidate.is_file() and candidate.suffix == ".pkl":
        return scan_pickle(candidate)

    # 显式给定的目录/pickle 都不存在 —— 给出有意义的报错
    raise FileNotFoundError(
        f"数据源不存在或类型不支持: {candidate} "
        f"(期望目录或 .pkl 文件；可先运行 run_build_sequence.py 生成 pickle)"
    )


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