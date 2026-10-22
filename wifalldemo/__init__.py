"""Wi-Fi 跌倒检测 Demo 工具包。

本工程面向 Widar3.0 兼容的 Wi-Fi CSI 数据集（每条样本是一个
由 N 个 CSI 包组成的 CSV 流，data 列以 52 子载波 (实部, 虚部) 交织
的 104 个浮点形式呈现），完成从数据加载、特征工程、二分类建模到
可视化演示的完整链路。

模型层: 当前版本已切到 PyTorch 深度学习 (:mod:`.models` 与 :mod:`.cnn`)。

.. note::
   这里**刻意不做** ``from . import cnn``。PyTorch 子模块按需导入
   （``from wifalldemo.cnn import ...``），否则只要碰一下
   ``wifalldemo.preprocessing`` 就会把 torch 拉进来 —— 而纯采集场景
   （``tools/csi_serial_to_csv.py`` / ``csi_collect_labeled.py``）根本不需要
   torch，现场笔记本装个 numpy + pyserial 就能开工。
"""
from __future__ import annotations

__version__ = "0.2.0"
