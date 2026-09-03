"""特征数据集构建：遍历样本清单 -> 特征矩阵 X / 标签 y / 窗来源记录号。

所有滑窗结果会附上其来源 record_id，从而支持
  1) 按人员划分 train/test
  2) 按文件多数投票得到样本级(片段级)检测结果
并提供 npz 缓存以加速重复实验。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import WINDOW_SIZE, WINDOW_STEP
from .features import extract_window_features
from .preprocessing import load_amplitude_series


def _process_one(path: str, win: int, step: int, fs: float):
    """单个文件的滑窗特征，返回 (feat, names)。样本包不足一窗时返回 None。"""
    amp = load_amplitude_series(path)
    if amp.shape[0] < win:
        return None
    feat, names, _starts = extract_window_features(amp, fs=fs, win=win, step=step)
    return feat, names


def build_feature_dataset(
    records: pd.DataFrame,
    win: int = WINDOW_SIZE,
    step: int = WINDOW_STEP,
    fs: float = 83.0,
    cache_fp: Path | str | None = None,
    force: bool = False,
    verbose: bool = True,
):
    """把 records(至少含 path/user/label 列)转化为训练矩阵。

    返回:
        X          (N, F) 特征矩阵
        y          (N,)   0/1 标签
        record_ids (N,)   每条窗所属样本在 records 中的行号
        names      [F]
    """
    if cache_fp is not None:
        cache_fp = Path(cache_fp)
        if cache_fp.exists() and not force:
            meta = json.loads((cache_fp.with_suffix(".json")).read_text(encoding="utf-8"))
            if (
                meta.get("n_files") == len(records)
                and meta.get("win") == win
                and meta.get("step") == step
                # 频域特征依赖采样率(FFT 频率分辨率/频带边界)，fs 不一致必须重建
                and abs(float(meta.get("fs", -1.0)) - fs) < 1e-3
            ):
                if verbose:
                    print(f"[dataset] 命中缓存: {cache_fp.name}")
                data = np.load(cache_fp)
                return (
                    data["X"],
                    data["y"],
                    data["record_ids"],
                    list(meta["names"]),
                )

    t0 = time.time()
    X_list, y_list, rid_list = [], [], []
    names: list[str] | None = None
    n_ok = 0
    # record_id 用 DataFrame 的索引标签而非枚举位置号。train/test/val 掩码
    # 均以索引标签为键；"位置号 == 标签"只在 scan_dataset 恰好返回连续
    # RangeIndex 且不做过滤/排序时成立，属脆弱耦合，这里显式对齐标签。
    rec_ids = records.index.to_numpy()
    for i, rec in enumerate(records.itertuples(index=False)):
        rid = rec_ids[i]
        try:
            res = _process_one(rec.path, win, step, fs)
        except Exception as exc:  # 坏文件仅告警，不中断流程
            print(f"[dataset] 跳过损坏文件 {rec.path}: {exc}")
            continue
        if res is None:
            continue
        feat, nm = res
        if names is None:
            names = nm
        elif nm != names:
            raise RuntimeError("特征顺序不一致，请检查特征工程代码")
        X_list.append(feat)
        y_list.append(np.full(len(feat), rec.label, dtype=np.int8))
        rid_list.append(np.full(len(feat), rid, dtype=np.int64))
        n_ok += 1
        if verbose and (n_ok % 200 == 0):
            print(f"[dataset] 已处理 {n_ok} 个样本 ...")

    if not X_list:
        raise RuntimeError("没有任何样本成功提取特征，请检查数据与窗口参数")
    X = np.concatenate(X_list, axis=0).astype(np.float32)
    y = np.concatenate(y_list)
    record_ids = np.concatenate(rid_list)
    if verbose:
        print(
            f"[dataset] 完成: {len(X)} 窗 (来自 {n_ok} 样本), "
            f"耗时 {time.time() - t0:.1f}s, 特征维度 {X.shape[1]}"
        )

    if cache_fp is not None:
        cache_fp = Path(cache_fp)
        cache_fp.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_fp, X=X, y=y, record_ids=record_ids
        )
        meta = {
            "n_files": len(records),
            "win": win,
            "step": step,
            "fs": fs,
            "names": list(names),
            "X_shape": list(X.shape),
        }
        (cache_fp.with_suffix(".json")).write_text(
            json.dumps(meta, ensure_ascii=False), encoding="utf-8"
        )
    return X, y, record_ids, names
