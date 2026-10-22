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

from . import config
from .audio import (
    AUDIO_FEATURE_VERSION,
    audio_feature_names,
    extract_audio_window_features,
)
from .config import WINDOW_SIZE, WINDOW_STEP
from .features import FEATURE_VERSION, extract_window_features
from .preprocessing import load_sample_amplitude, normalize_per_subcarrier


def _norm_tag() -> str:
    """原始窗缓存的归一化指纹：开关或参考窗一变，旧缓存就失效。"""
    if not config.NORMALIZE_PER_SUBCARRIER:
        return "raw"
    return f"norm{config.NORM_WINDOW_S:g}s" + ("c" if config.NORM_CAUSAL else "")


def _records_fingerprint(records: pd.DataFrame, audio: bool = False) -> str:
    """样本清单的内容指纹：path 列表 + 源文件 mtime/size 的哈希。

    缓存 meta 只记 n_files 是弱校验——换数据源后若文件数恰好相同会静默命中
    旧缓存（历史上踩过：Widar 的特征被 realdata 复用，训出假结果且无提示）。
    把 path 列和每个源文件的 mtime/size 一起哈希，数据一变缓存自动失效，
    ``--no-cache`` 只在想强制重建时才需要。

    ``audio=True`` 时把每条样本的 ``.wav`` / ``.sync.json`` 也纳入指纹：重录声音
    但 CSI 没变（或只改了同步信息）时，音频特征必须跟着失效。
    """
    import hashlib
    import os

    h = hashlib.sha1()
    for p in records["path"].tolist():
        s = str(p)
        h.update(s.encode("utf-8", "ignore"))
        h.update(b"\0")
        fp = s.split("::", 1)[0]  # pickle 引用取 pkl 文件路径
        try:
            st = os.stat(fp)
            h.update(str(int(st.st_mtime)).encode())
            h.update(str(st.st_size).encode())
        except OSError:
            pass
        if audio and "::" not in s:
            for suffix in (".wav", ".sync.json"):
                try:
                    ast = os.stat(str(Path(s).with_suffix(suffix)))
                except OSError:
                    continue
                h.update(suffix.encode())
                h.update(str(int(ast.st_mtime)).encode())
                h.update(str(ast.st_size).encode())
    return h.hexdigest()[:16]


def features_for_sample(
    path: str,
    win: int = WINDOW_SIZE,
    step: int = WINDOW_STEP,
    fs: float = 83.0,
    audio: bool = False,
):
    """单样本 -> ``(X, names, starts)``：70 维 CSI 特征 (+ 31 维音频特征)。

    训练和推理都走这一个函数，保证切窗与特征拼接口径完全一致 —— 两处各写一遍
    迟早会漂移（历史上 CNN 通路就因为切片约定不一致出过问题）。

    ``audio=True`` 时，每条样本的 ``.wav`` 会按 ``.sync.json`` 的时间轴映射，
    对齐到**同一批窗起点**上再提特征；样本没有音频时补全零 + ``audio_present=0``。

    返回 ``None`` 表示样本包数不足一窗。
    """
    amp = load_sample_amplitude(path)
    if amp.shape[0] < win:
        return None
    feat, names, starts = extract_window_features(amp, fs=fs, win=win, step=step)
    if audio:
        a = extract_audio_window_features(path, starts, win, csi_fs=fs)
        feat = np.hstack([feat, a]).astype(np.float32)
        names = list(names) + audio_feature_names()
    return feat, names, starts


def _process_one(path: str, win: int, step: int, fs: float, audio: bool = False):
    """单个文件的滑窗特征，返回 (feat, names)。样本包不足一窗时返回 None。

    path 可以是 CSV 路径或 ``<pkl>::<idx>`` 形式的 pickle 引用；两者都通过
    ``load_sample_amplitude`` 统一加载（内部自动识别 ``::`` 并走对应分支）。
    """
    res = features_for_sample(path, win=win, step=step, fs=fs, audio=audio)
    if res is None:
        return None
    feat, names, _starts = res
    return feat, names


def _process_raw_windows(path: str, win: int, step: int, fs: float = 83.0):
    """读一个文件后切窗，返回 ``(raw_windows, starts)``；窗数不足时返回 ``None``。

    与 :func:`_process_one` 并行: 后者输出 70 维手工特征，本函数输出
    (n_windows, win, n_subcarriers) 的原始幅度张量，供 CNN 训练使用。

    **归一化发生在切窗之前、整条样本的尺度上**——参考量需要 1~2 s 的时间上下文，
    先切窗再归一化会让每个窗各自减掉自己的基线，把跌倒那 1.5 s 里的慢变也抹掉。
    """
    amp = load_sample_amplitude(path)
    if config.NORMALIZE_PER_SUBCARRIER:
        amp = normalize_per_subcarrier(
            amp, fs=fs, window_s=config.NORM_WINDOW_S, causal=config.NORM_CAUSAL
        )
    T = amp.shape[0]
    if T < win:
        return None
    starts = list(range(0, T - win + 1, step))
    raw = np.stack([amp[s : s + win] for s in starts], axis=0).astype(np.float32)
    return raw, np.asarray(starts)


def build_feature_dataset(
    records: pd.DataFrame,
    win: int = WINDOW_SIZE,
    step: int = WINDOW_STEP,
    fs: float = 83.0,
    cache_fp: Path | str | None = None,
    force: bool = False,
    verbose: bool = True,
    audio: bool = False,
):
    """把 records(至少含 path/user/label 列)转化为训练矩阵。

    参数 ``audio=True`` 时，每条样本的 70 维 CSI 特征后会再拼 31 维声学特征
    （见 :mod:`wifalldemo.audio`），特征维度变为 89；样本缺音频时那一半填零、
    ``audio_present=0``，样本本身照常保留。

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
                # 数据源内容指纹：换数据/改文件必须重建（n_files 相同也挡不住）
                and meta.get("records_hash") == _records_fingerprint(records, audio=audio)
                # 音频开关/特征版本变了，列的含义就变了，绝不能复用
                and bool(meta.get("audio", False)) == bool(audio)
                and int(meta.get("audio_version", 0)) == (AUDIO_FEATURE_VERSION if audio else 0)
                # CSI 手工特征版本同理：特征增删后旧缓存的列含义错位，必须重建
                and int(meta.get("feature_version", 0)) == FEATURE_VERSION
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
            res = _process_one(rec.path, win, step, fs, audio=audio)
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
            "audio": bool(audio),
            "audio_version": AUDIO_FEATURE_VERSION if audio else 0,
            "feature_version": FEATURE_VERSION,
            "records_hash": _records_fingerprint(records, audio=audio),
        }
        (cache_fp.with_suffix(".json")).write_text(
            json.dumps(meta, ensure_ascii=False), encoding="utf-8"
        )
    return X, y, record_ids, names


def build_raw_window_dataset(
    records: pd.DataFrame,
    win: int = WINDOW_SIZE,
    step: int = WINDOW_STEP,
    fs: float = 83.0,
    cache_fp: Path | str | None = None,
    force: bool = False,
    verbose: bool = True,
):
    """遍历 ``records`` 并输出原始滑窗张量 (CNN 用)。

    返回
    ----
    X_raw       (N, win, n_subcarriers) float32
    y           (N,) 0/1
    record_ids  (N,) 每个窗对应的 ``records`` 索引

    与 :func:`build_feature_dataset` 走同一 records 序列，窗起点一致；上层
    可分别用它俩并以 record_ids 对齐训练/测试掩码。
    """
    if cache_fp is not None:
        cache_fp = Path(cache_fp)
        if cache_fp.exists() and not force:
            meta_p = cache_fp.with_suffix(".json")
            if meta_p.exists():
                meta = json.loads(meta_p.read_text(encoding="utf-8"))
                if (
                    meta.get("n_files") == len(records)
                    and meta.get("win") == win
                    and meta.get("step") == step
                    # 归一化开关/窗长变了必须重建：换表示等于换数据
                    and meta.get("norm") == _norm_tag()
                    # 归一化参考窗按 fs 换算帧数，fs 变了同样要重建
                    and abs(float(meta.get("fs", -1.0)) - fs) < 1e-3
                    and meta.get("records_hash") == _records_fingerprint(records)
                ):
                    if verbose:
                        print(f"[dataset] 命中原始窗缓存: {cache_fp.name}")
                    data = np.load(cache_fp)
                    return data["X_raw"], data["y"], data["record_ids"]

    t0 = time.time()
    raw_list, y_list, rid_list = [], [], []
    rec_ids = records.index.to_numpy()
    n_ok = 0
    for i, rec in enumerate(records.itertuples(index=False)):
        rid = rec_ids[i]
        try:
            res = _process_raw_windows(rec.path, win, step, fs=fs)
        except Exception as exc:
            print(f"[dataset] 跳过损坏文件 {rec.path}: {exc}")
            continue
        if res is None:
            continue
        raw, _starts = res
        raw_list.append(raw)
        y_list.append(np.full(len(raw), rec.label, dtype=np.int8))
        rid_list.append(np.full(len(raw), rid, dtype=np.int64))
        n_ok += 1
        if verbose and (n_ok % 200 == 0):
            print(f"[dataset][raw] 已处理 {n_ok} 个样本 ...")

    if not raw_list:
        raise RuntimeError("没有任何样本成功切窗，请检查数据与窗口参数")

    X_raw = np.concatenate(raw_list, axis=0).astype(np.float32)
    y = np.concatenate(y_list)
    record_ids = np.concatenate(rid_list)
    if verbose:
        print(
            f"[dataset][raw] 完成: {len(X_raw)} 窗, 形状 {X_raw.shape}, "
            f"耗时 {time.time() - t0:.1f}s"
        )

    if cache_fp is not None:
        cache_fp = Path(cache_fp)
        cache_fp.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_fp, X_raw=X_raw, y=y, record_ids=record_ids)
        meta = {
            "n_files": len(records),
            "win": win,
            "step": step,
            "norm": _norm_tag(),
            "X_raw_shape": list(X_raw.shape),
        }
        cache_fp.with_suffix(".json").write_text(
            json.dumps(meta, ensure_ascii=False), encoding="utf-8"
        )
    return X_raw, y, record_ids
