"""中央集権シード(D13)。

原則: 乱数は必ず `RngHub.stream(...)` から取り、キーに (用途, agent_id, step) を含める。
→ どのエージェントの乱数も他者の実行順に依存しない = 決定論リプレイの土台。
"""
from __future__ import annotations

import hashlib

import numpy as np


def _stable_hash(value: str) -> int:
    """Python の hash() はプロセス毎に変わるため sha256 で安定化。"""
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


class RngHub:
    def __init__(self, master_seed: int):
        self.master_seed = int(master_seed)

    def stream(self, *key: int | str) -> np.random.Generator:
        """(master_seed, *key) から独立した乱数ストリームを派生する。"""
        ints = [self.master_seed]
        for part in key:
            ints.append(part if isinstance(part, int) else _stable_hash(part))
        return np.random.Generator(np.random.PCG64(np.random.SeedSequence(ints)))

    def key_name(self, *key: int | str) -> str:
        return "/".join(str(part) for part in key)
