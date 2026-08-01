"""決定論的 .npz 書き出しユーティリティ(stdlib + numpy のみ)。

`np.savez` / `np.savez_compressed` は zip エントリに **実行時のローカル時刻** を焼き込むため、
同一入力でも 2 回実行するとバイト列が一致しない。来歴(provenance)を成果物のハッシュで
主張したい本リポの流儀(第78バッチ metrics_spec_hash 等)と噛み合わないので、
ZipInfo を固定した最小の書き出し器を用意する。

固定するもの:
  - date_time = (1980, 1, 1, 0, 0, 0)  … zip が表現できる最小時刻
  - create_system = 0                  … 既定は win32/posix で分岐するため明示固定
  - external_attr = 0o600 << 16        … 権限ビットの環境差を消す
  - compress_type / compresslevel      … 明示指定(zlib の既定変化に依存しない)
  - キー順 = 呼び出し側の引数順(dict の挿入順)

読み出しは `np.load(path)` がそのまま使える(中身は通常の .npy を並べた zip)。
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np

FIXED_DATE = (1980, 1, 1, 0, 0, 0)
COMPRESSLEVEL = 6


def save_npz(path, arrays, compress=True):
    """arrays(dict[str, array-like])を決定論的な .npz として path に書く。

    同一の arrays を同じ numpy/zlib で 2 回書けばバイト列は完全一致する。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    comp = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    kw = {"compresslevel": COMPRESSLEVEL} if compress else {}
    with zipfile.ZipFile(path, "w", compression=comp, **kw) as zf:
        for name, arr in arrays.items():
            a = np.asanyarray(arr)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=FIXED_DATE)
            info.compress_type = comp
            info.create_system = 0
            info.external_attr = 0o600 << 16
            with zf.open(info, "w", force_zip64=True) as fh:
                np.lib.format.write_array(fh, a, allow_pickle=False)
    return path


def dump_json(path, obj, indent=None):
    """JSON も改行/エンコーディングを固定して決定論的に書く(LF 固定・UTF-8)。"""
    import json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    txt = json.dumps(obj, ensure_ascii=False, indent=indent,
                     separators=(",", ":") if indent is None else (",", ": "),
                     sort_keys=False)
    path.write_bytes((txt + "\n").encode("utf-8"))
    return path
