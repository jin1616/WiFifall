"""通用测试入口：当 ``tests/`` 目录被 ``python -m tests`` 形式调用时，确保
``wifalldemo`` 包可被找到 (项目根在 sys.path)。
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
