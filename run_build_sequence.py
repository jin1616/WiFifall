#!/usr/bin/env python
"""把 ``DATA/ID*/<action>/*.csv`` 原始 CSI 一次性打包为 data_sequence.pkl。

合并并优化了原 process1.py 与 process2.py：

* 解析子载波复数沿用 wifalldemo.preprocessing.parse_data_field
  (real = x[2i], imag = x[2i+1])，与项目现有流水线保持一致，
  顺便修正了原 process1 里 ``imag = x[2i-1]`` 在 i=0 时取到 x[-1] 的下标 bug。
* 路径、间隙阈值、输出路径均参数化，便于在不同数据集/工程间复用。
* 异常隔离：单个 CSV 解析失败仅打 ``[skip]`` 警告，不中断整体。
* 进度日志每 50 个文件输出一次，避免大批量时无反馈。

用法::

    python run_build_sequence.py                                  # 默认扫 DATA/ + 输出 outputs/data_sequence.pkl
    python run_build_sequence.py --data-dir DATA \\               # 显式指定（等价于默认）
                                 --output outputs/data_sequence.pkl
    python run_build_sequence.py --gap-us 12000                   # 调大判定 padding 的时间间隙阈值

输出::

    outputs/data_sequence.pkl  -> List[Dict]
        每条记录对应一个原始 CSV：
            time        (T,)   int64 微秒；按 packet 间 gap 插入 padding 行后的展开时间
            people      int    人员序号 (0..N-1)
            action      int    活动序号 (0..M-1)
            user_name   str    如 "ID0"，便于按人划分
            action_name str    如 "fall"，便于按活动划分
            magnitude   (T, 52) float32，padding 行为 PAD_VALUE
            phase       (T, 52) float32 (度)，padding 行为 PAD_VALUE
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from wifalldemo.config import DATA_DIR, N_SUBCARRIERS
from wifalldemo.preprocessing import parse_data_field


# 经验值：实测 median packet interval ≈ 10000us ⇒ ~100Hz；
# 与项目 SAMPLING_RATE_HZ(83Hz) 略有差异是 median < mean 的常见现象。
# 超过 1 个 loacl_gap 仍未到下一包即在中间插入 padding 行，保持时序均匀。
DEFAULT_LOCAL_GAP_US = 10_000
# padding 行用 NaN 而非 -1000 —— 关键点：|complex(NaN, NaN)| = NaN，
# 不像 -1000 取绝对值后会变成 +1000 与正常 CSI 数据混淆。NaN 同时穿过
# abs/angle，到特征工程前由 ``_load_amplitude_from_pickle`` 按 NaN 行剔除。
PAD_VALUE = float("nan")


def _csv_to_complex_rows(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """单个 CSV 的 data 列 -> (csi_complex (T, N), timestamp, local_timestamp_us)。

    复用项目 parse_data_field 完成子载波复数解析；任何一行解析失败即跳过。
    返回 None 表示该 CSV 没有可用 CSI 包。
    """
    if "data" not in df.columns or "local_timestamp" not in df.columns:
        return None
    df = df.dropna(subset=["data", "local_timestamp"]).reset_index(drop=True)
    if df.empty:
        return None

    csi_rows: list[np.ndarray] = []
    for raw in df["data"].values:
        if not isinstance(raw, str):
            continue
        c = parse_data_field(raw)  # complex64 (N,) 或 None
        if c is None:
            continue
        csi_rows.append(c.astype(np.complex64))
    if not csi_rows:
        return None
    csi = np.stack(csi_rows, axis=0)  # (T, N)

    try:
        ts = df["timestamp"].astype(str).values[: len(csi_rows)]
        lt = df["local_timestamp"].astype(np.int64).values[: len(csi_rows)]
    except (KeyError, ValueError, TypeError):
        return None
    return csi, ts, lt


def _insert_padding(
    csi: np.ndarray, local_time: np.ndarray, gap_us: int
) -> tuple[np.ndarray, np.ndarray]:
    """在 local_time 出现大间隙的位置插入 padding 帧，保持时序均匀。

    与原 process2 行为一致：padding 行的 magnitude/phase 全部置为 PAD_VALUE，
    时间戳在原区间内线性插值。
    """
    T = csi.shape[0]
    if T == 0:
        return csi, local_time.astype(np.float64)
    if T == 1:
        return csi.copy(), local_time.astype(np.float64).reshape(1)

    pad_csi = np.full(N_SUBCARRIERS, PAD_VALUE, dtype=csi.dtype)

    out_csi = [csi[0]]
    out_time = [float(local_time[0])]
    last = float(local_time[0])
    for i in range(1, T):
        cur = float(local_time[i])
        num = int(round((cur - last - gap_us) / gap_us))
        if num > 0:
            delta = (cur - last) / (num + 1)
            for _ in range(num):
                out_csi.append(pad_csi.copy())
                out_time.append(out_time[-1] + delta)
        out_csi.append(csi[i])
        out_time.append(cur)
        last = cur
    return (
        np.stack(out_csi, axis=0),
        np.asarray(out_time, dtype=np.float64),
    )


def build_sequence(
    data_dir: str | Path,
    output_path: str | Path,
    gap_us: int = DEFAULT_LOCAL_GAP_US,
    verbose: bool = True,
) -> int:
    """主流程：扫描 data_dir，生成 output_path pickle，返回写入的样本数。"""
    data_dir = Path(data_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not data_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    seq: list[dict] = []
    n_ok = 0
    n_skip = 0

    # 与项目约定一致：按目录名字典序给人员/活动分配整数 id
    user_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("ID")])
    if not user_dirs:
        raise FileNotFoundError(f"在 {data_dir} 下未找到 ID*/ 用户子目录")

    for people_id, user_dir in enumerate(user_dirs):
        for action_id, action_dir in enumerate(sorted(user_dir.iterdir())):
            if not action_dir.is_dir():
                continue
            for fp in sorted(action_dir.glob("*.csv")):
                try:
                    df = pd.read_csv(fp)
                except Exception as exc:
                    n_skip += 1
                    print(f"  [skip] 读取失败 {fp.name}: {exc}", file=sys.stderr)
                    continue
                loaded = _csv_to_complex_rows(df)
                if loaded is None:
                    n_skip += 1
                    print(f"  [skip] 无可用 CSI 包: {fp}")
                    continue
                csi, _ts, lt = loaded
                csi_padded, time_padded = _insert_padding(csi, lt, gap_us)
                magnitude = np.abs(csi_padded).astype(np.float32)
                phase = np.angle(csi_padded, deg=True).astype(np.float32)
                seq.append(
                    {
                        "time": time_padded,
                        "people": people_id,
                        "action": action_id,
                        "user_name": user_dir.name,
                        "action_name": action_dir.name,
                        "magnitude": magnitude,
                        "phase": phase,
                    }
                )
                n_ok += 1
                if verbose and n_ok % 50 == 0:
                    print(f"  已处理 {n_ok} 个文件 ...")

    if not seq:
        raise RuntimeError("未生成任何样本，请检查 data_dir 路径与文件格式")

    with output_path.open("wb") as f:
        pickle.dump(seq, f)

    if verbose:
        print(f"  完成: {n_ok} 个样本，{n_skip} 个被跳过 -> {output_path}")
    return n_ok


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="合并 process1+process2，把 ID*/<action>/*.csv 打包为 data_sequence.pkl"
    )
    ap.add_argument(
        "--data-dir",
        default=str(DATA_DIR),
        help="包含 ID*/<action>/*.csv 的根目录 (默认: config.DATA_DIR，即 DATA/)",
    )
    ap.add_argument(
        "--output",
        default=str(Path("outputs") / "data_sequence.pkl"),
        help="输出 pickle 路径 (默认: outputs/data_sequence.pkl)",
    )
    ap.add_argument(
        "--gap-us",
        type=int,
        default=DEFAULT_LOCAL_GAP_US,
        help="判定相邻 packet 之间为'间隙'的微秒阈值 (默认: 10000)",
    )
    ap.add_argument("--quiet", action="store_true", help="关闭进度打印")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    n = build_sequence(
        args.data_dir, args.output, gap_us=args.gap_us, verbose=not args.quiet
    )
    print(f"\n完成。{n} 个样本已写入 {args.output}")


if __name__ == "__main__":
    main()