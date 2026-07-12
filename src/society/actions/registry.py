"""動詞レジストリ(Bundle C の第1弾: move_to / speak / coin_label / stay)。

新しい動詞の追加 = ここに関数を1個登録(design §9 のモジュール式拡張)。
実行は engine/scheduler が行う(このモジュールは「何ができるか」の定義のみ)。
"""
from __future__ import annotations

from typing import Callable

VERBS: dict[str, str] = {}


def register_verb(name: str, description: str) -> None:
    VERBS[name] = description


register_verb("move_to",    "目的地を決めて経路移動を開始する")
register_verb("continue",   "現在の経路を進み続ける")
register_verb("stay",       "その場に留まる")
register_verb("speak",      "近くの人に話す(非ブロードキャスト)")
register_verb("coin_label", "状況に新しい名前(語)を付ける")
register_verb("reflect",    "内省する(夜・ソロ)")
