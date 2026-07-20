"""標高グリッドの O(1) サンプリング(3D Phase 0・第38バッチ W2)。

抽出済みの地形格子(terrain.npz + terrain.json)を読み、局所座標 (x, y) の
地表高 z(基準面からの相対 m)を双一次補間で返す。表示・観測専用
(移動判定・認知・プロンプトには使わない)。乱数なし・純関数=決定論・k 非依存。
格子仕様と補間は scripts/export_3d.py の読取と同値(H[iy][ix]・row-major・
端セルへのクランプ=外挿しない・小数1桁丸め)= sim 由来 z とエクスポート z が一致する。
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


class ElevationGrid:
    """2次元標高格子の双一次補間サンプラ(格子外は端セルにクランプ)。"""

    def __init__(self, heights, x0: float, y0: float, cell_m: float) -> None:
        H = np.ascontiguousarray(np.asarray(heights, dtype=np.float64))
        if H.ndim != 2:
            raise ValueError("heights は 2次元 (ny, nx) 配列であること")
        self._H = H
        self.ny, self.nx = H.shape
        self.x0 = float(x0)
        self.y0 = float(y0)
        self.cell_m = float(cell_m)

    @classmethod
    def load(cls, dir_path: str | Path) -> "ElevationGrid | None":
        """terrain.npz + terrain.json のあるフォルダから読む。無ければ None(縮退)。"""
        npz_p = Path(dir_path) / "terrain.npz"
        json_p = Path(dir_path) / "terrain.json"
        if not (npz_p.exists() and json_p.exists()):
            return None
        meta = json.loads(json_p.read_text(encoding="utf-8"))
        data = np.load(npz_p)
        if "heights" in data:
            H = data["heights"]
        elif "z" in data:
            H = data["z"]
        else:
            H = data[data.files[0]]
        return cls(H, meta["x0"], meta["y0"], meta["cell_m"])

    def height_at(self, x: float, y: float) -> float:
        """(x, y) の地表高 [m](小数1桁)。平面格子では格子内で厳密に平面値。"""
        fx = (x - self.x0) / self.cell_m
        fy = (y - self.y0) / self.cell_m
        ix = min(max(math.floor(fx), 0), max(self.nx - 2, 0))
        iy = min(max(math.floor(fy), 0), max(self.ny - 2, 0))
        tx = min(max(fx - ix, 0.0), 1.0)
        ty = min(max(fy - iy, 0.0), 1.0)
        ix1 = min(ix + 1, self.nx - 1)
        iy1 = min(iy + 1, self.ny - 1)
        H = self._H
        h00 = float(H[iy][ix])
        h10 = float(H[iy][ix1])
        h01 = float(H[iy1][ix])
        h11 = float(H[iy1][ix1])
        h = (h00 * (1 - tx) * (1 - ty) + h10 * tx * (1 - ty)
             + h01 * (1 - tx) * ty + h11 * tx * ty)
        return round(h, 1)
