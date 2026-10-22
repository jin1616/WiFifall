#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""目录扫描必须只收 CSI 样本，不能把四件套里的附属文件也当样本。

    python -m tests.test_data_loader_scan

背景（2026-09-23 真实踩到）：``csi_audio_sync`` 每条样本产出四件套
``xxx.csv`` / ``xxx_audio_env.csv`` / ``xxx.wav`` / ``xxx.sync.json``，
前两个都是 .csv 且放在同一个 ``IDxx/<action>/`` 目录里。
``scan_dataset`` 之前只按 ``*.csv`` 扫，于是 92 条真样本被扫成 184 条
——一半是只有 4 列的声音能量包络，喂进训练就是脏数据，而且样本量凭空翻倍。

这里用一个临时目录锁住"只认 CSI 样本"这件事，免得以后再退化。
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from wifalldemo.data_loader import _is_csi_sample, scan_dataset  # noqa: E402

# 项目真实的 CSI CSV 表头（列数无所谓，这里只构造能被扫到的文件）
_HEADER = "type,seq,timestamp,target,mac,rssi,local_timestamp,data\n"


def _write(path: Path, header: str = _HEADER) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header, encoding="utf-8")


def test_is_csi_sample() -> None:
    assert _is_csi_sample(Path("fall_001.csv"))
    assert not _is_csi_sample(Path("fall_001_audio_env.csv"))
    assert not _is_csi_sample(Path("FALL_001_AUDIO_ENV.CSV"))   # 大小写不敏感
    assert _is_csi_sample(Path("sit_003.csv"))
    print("  _is_csi_sample: 真样本收、音频包络拒（大小写不敏感）, OK")


def test_scan_skips_audio_env() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # ID1: 2 fall + 1 sit + 1 jump，每条都带一个 _audio_env 伴生文件
        # （jump 是 Widar 的合法负类活动，所以会被正常收进来）
        for act, n in (("fall", 2), ("sit", 1), ("jump", 1)):
            for i in range(1, n + 1):
                _write(root / "ID1" / act / f"{act}_{i:03d}.csv")
                _write(root / "ID1" / act / f"{act}_{i:03d}_audio_env.csv",
                       "t_pc,csi_row,rms,dbfs\n")
        _write(root / "_probe.csv")                       # 顶层探测文件：不在用户目录下

        df = scan_dataset(str(root))
        assert len(df) == 4, f"应只收 4 条 CSI 样本，实际 {len(df)}"
        assert not df["path"].str.contains("_audio_env").any()
        assert not df["path"].str.contains("_probe").any()
        assert sorted(set(df["activity"])) == ["fall", "jump", "sit"]
        print(f"  扫描 {len(df)} 条（2 fall + 1 sit + 1 jump），"
              f"伴生的 *_audio_env.csv 与顶层 _probe.csv 均已跳过, OK")


def test_real_dataset_has_no_audio_env() -> None:
    """真数据上回归：realdata 里 wav 数与 CSI 样本数应当相等（四件套一一对应）。"""
    real = _ROOT / "realdata"
    if not real.is_dir():
        print("  （跳过：没有 realdata 目录）")
        return
    df = scan_dataset(str(real))
    n_csi = len(df)
    n_wav = len(list(real.rglob("*.wav")))
    assert not df["path"].str.contains("_audio_env").any(), "扫描混入了音频包络"
    if n_wav:
        assert n_csi == n_wav, f"CSI 样本 {n_csi} 与 wav {n_wav} 数量不等，四件套不配套"
    print(f"  realdata: {n_csi} 条 CSI 样本 / {n_wav} 个 wav，一一对应, OK")


if __name__ == "__main__":
    test_is_csi_sample()
    test_scan_skips_audio_env()
    test_real_dataset_has_no_audio_env()
    print("\n[PASS] 目录扫描只收 CSI 样本 3/3")
