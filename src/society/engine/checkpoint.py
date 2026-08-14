"""D16: チェックポイント(完全状態の保存)と途中再開。

設計原則
--------
- RngHub はステートレス(stream は (用途, id, step) から都度導出)。master seed だけで
  再現できるので**乱数状態は保存しない**。
- city(地図)/ router / clock / logger / llm / transit は config から再構築できる or
  load 後に再配線するので**保存しない**。scenario の封鎖(closed フラグ)だけは
  load 時に city.graph へ再適用し、router.invalidate() で経路キャッシュを捨てる。
- 「何を集め、どう戻すか」は本モジュールに一元管理する(個別クラスに __getstate__ を
  生やさない)。共有参照(labels.items ⇄ sim.items の ItemStore、Item 実体、
  labels.text_to_item と items.items が同じ Item を指すこと)は **1 回の
  pickle.dumps に同梱**して保つ(pickle は 1 回の dumps 内では共有参照を保存する)。
- 決定論: set は原則 membership / sorted 反復のみ(全ファイル監査済み)。順序安定の
  ため scenario.closed は sorted list で直列化して復元する。pickle はプロセス内でも
  set の反復順を保存しない(dict は保存する)ため、反復順が観測に効く箇所は sorted で
  なければならない — 監査の結果、そうなっている(唯一の生反復 adopted は全経路で
  sorted、例外は engine の post/dm フォールバック 1 行のみで mock では非到達)。
"""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
import os
import pickle
from pathlib import Path

from .. import ablate as _ablate_ckpt
from .. import facility_devices as _facility_mod
from .. import medical as _medical_mod
from .. import population as _population
from .. import worldview as _wv_ckpt

FORMAT = 1

# resume で正当に変わりうるキー(整合性ハッシュから除外する)。
# n_steps は「さらに先まで回す」ため必ず変わる。name/out_dir/checkpoint_every も run 制御。
_VOLATILE_KEYS = [
    ("run", "n_steps"), ("run", "name"), ("run", "out_dir"),
    ("observer", "checkpoint_every"),
]


def config_hash_from_container(data: dict) -> str:
    """resolved config(OmegaConf.to_container 済み dict)の内容ハッシュ。

    resolve 済みの dict を既に持っている呼び出し側(observer/manifest.py)が
    to_container を二度走らせずに済むよう、定義をここ 1 箇所に保ったまま口を分けてある。
    """
    data = copy.deepcopy(data)
    for sect, key in _VOLATILE_KEYS:
        if isinstance(data.get(sect), dict):
            data[sect].pop(key, None)
    blob = json.dumps(data, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def config_hash(cfg) -> str:
    """決定論に効く config の内容ハッシュ(resume 制御キーは除外)。"""
    from omegaconf import OmegaConf
    return config_hash_from_container(OmegaConf.to_container(cfg, resolve=True))


def save(sim, step: int, path: str | Path) -> Path:
    """シミュレーションの完全状態を pickle+gzip で保存する(原子的 rename)。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "format": FORMAT,
        "step": int(step),
        "config_hash": config_hash(sim.cfg),
        # --- 完全状態(自己完結・transient 参照を含まないオブジェクト群) ---
        "agents": sim.agents,             # dataclass/MemoryStore/Counter/set/opinion/money…
        "labels": sim.labels,             # LabelSystem(内部に ItemStore=sim.items を同梱)
        "net": sim.net,                   # posts/news/follows/contacts/likes/priority/既読
        "tools": sim.tools,               # events/groups/proposals/ventures/flyers/member_of
        "rulebook": sim.rulebook,         # 制度DSL: active ルール/日次進行/ブースト(自己完結)
        "world_events": sim.world_events, # 残り(shock_event が実行時に append しうる)
        # --- scheduler が sim に持つランタイムカウンタ ---
        "runtime": {
            "drive_stats": sim.drive_stats,
            "econ_day": sim._econ_day,
            "acct_day": sim._acct_day,          # 口座 E5: 暦日境界の進行(給料日/家賃)
            # 会社観測データ層 B4: per-day アキュムレータ(mid-day checkpoint でも resume==straight を保つ)。
            # OFF ランでは空 dict / -1=挙動不変(load は .get で旧 checkpoint 互換)。workers 集合は
            # emit 時に必ず sorted 反復=pickle の集合反復順非保存の影響を受けない(determinism 監査済み)。
            "org_day": getattr(sim, "_org_day", {}),
            "org_ledger_day": getattr(sim, "_org_ledger_day", -1),
            # 第97バッチ IF-E2 の前提工事(**既存欠陥**): オフィス産出の日境界進行。
            # 未保存だったため work.service ON かつ office.by_org OFF のランで
            # mid-day resume が同じ日の org_output を**二重記録**していた
            # (実測: 288step / 150step 分割で 33 → 44 件)。第80 W2(weather の二重記録)と
            # まったく同じ型のギャップ。OFF ランでは -1=挙動不変(load は .get で旧 ckpt 互換)。
            "work_day": getattr(sim, "_work_day", -1),
            # 内部可動性 第60バッチ b: 転居/同棲の日境界進行(mid-day checkpoint でも resume==straight
            # を保つ)。新規の per-agent 状態(通勤既知値/同棲経過日/同棲フラグ)は agents pickle に自然
            # 同梱=ここは日カウンタのみ中央管理する。OFF ランでは -1=挙動不変(load は .get で旧 ckpt 互換)。
            "housing_day": getattr(sim, "_housing_day", -1),
            # キャリア転換(Wave G5)の日境界進行。第60バッチ b の career 由来転居が resume で転職を
            # 二重発火しないよう中央管理する(従来は未保存=career ON の mid-day resume が未検証だった)。
            # OFF ランでは -1=挙動不変(load は .get で旧 checkpoint 互換)。
            "career_day": getattr(sim, "_career_day", -1),
            # 社会関係(Wave G2)の日境界進行 第61検収補修: 未保存だと relations ON の mid-day resume で
            # _phase_relations_day が同じ日を再処理(closeness減衰/評判風化の二重発火)し resume!=straight。
            # 第61バッチの gossip resume テストが顕在化させた既存ギャップ。OFF ランでは -1=挙動不変。
            "rel_day": getattr(sim, "_rel_day", -1),
            # 資産レンズ 第59(検収補修): 前日 wealth スナップ+τ(observer/assets の state)。
            # processed はプロセス内 logger カウンタ由来なので保存しない(load 側で 0 に戻す)。
            # OFF ランでは _assets_state 自体が無い → None=挙動不変(load は .get で旧 ckpt 互換)。
            "assets_state": (
                {"day": sim._assets_state["day"], "prev": sim._assets_state["prev"],
                 "tau": sim._assets_state["tau"]}
                if getattr(sim, "_assets_state", None) else None),
            # 負の評判 第61バッチ c: 種/伝播/忘却の日境界進行(gossip_day)+ 当日カウンタ(gossip_state)+
            # 未ロールの種候補(gossip_pending)。agent 側の _gossip_known/_gossip_heard は agents pickle に
            # 自然同梱。watermark はプロセス内 logger カウンタ由来なので保存しない(load で 0 に戻す=assets
            # と同流儀)。OFF ランでは -1/None=挙動不変(load は .get で旧 checkpoint 互換)。
            "gossip_day": getattr(sim, "_gossip_day", -1),
            "gossip_state": getattr(sim, "_gossip_state", None),
            "gossip_pending": getattr(sim, "_gossip_pending", None),
            # 共同行動 第62バッチ: 日境界進行(joint_day)+当日の成立グループ(logged フラグ込み)+
            # 累積件数(joint_total=L2 n_joint_activity)。従来未保存で mid-day resume が同じ日を
            # 再編成し joint_activity を二重記録し得た(既存ギャップ)。承諾内生化(joint_invite が
            # 日境界で出る)で resume==straight を成立させるため中央管理へ。agent 側 joint_today は
            # agents pickle に自然同梱。endo_state は当日の誘いタリー(承諾/内生判定/履行の set は
            # membership/len のみ使用=集合反復順非保存の影響なし)。OFF ランでは -1/[]/0/None=挙動不変
            # (load は .get で旧 checkpoint 互換)。
            "joint_day": getattr(sim, "_joint_day", -1),
            "joint_groups": getattr(sim, "_joint_groups", []),
            "joint_total": getattr(sim, "_joint_total", 0),
            "endo_state": getattr(sim, "_endo_state", None),
            # 第64バッチ: 誘い経路の当日タリー(invite ON 時のみ存在。int のみ=set なし=
            # 決定論監査は自明)。OFF ランでは None=挙動不変(load は .get で旧 checkpoint 互換)。
            "invite_state": getattr(sim, "_invite_state", None),
            # 第65バッチ: 会話由来 magnitude の当日タリー(quality ON 時のみ存在。int/float のみ)。
            # mid-day checkpoint でも当日平均(L2 quality_magnitude_mean)が resume==straight に
            # なるよう中央管理する。OFF ランでは None=挙動不変(load は .get で旧 ckpt 互換)。
            "quality_state": getattr(sim, "_quality_state", None),
            # 第70バッチ IDEA①(エコー計測): rolling 窓のタリー(deque + カウンタ)。L2 の 5 列は
            # **常設**なので、これを保存しないと mid-day resume の L2 が straight と食い違う
            # (第62の joint 状態と同じ既存ギャップの型)。processed カウンタは**プロセス内 logger
            # 由来**なので保存しない(load 側で 0 に戻す=assets/gossip と同流儀)。集合は使わず
            # deque/dict/Counter のみ=pickle の集合反復順非保存の影響を受けない。
            # observer.echo.enabled=false のランでは state 自体が生えない → None=挙動不変。
            "echo_state": getattr(sim, "_echo_state", None),
            # 第91バッチ(退行シグナル監視): rolling 窓の step バケツ deque + 窓合計。
            # 第70 echo とまったく同じ型のギャップで、保存しないと mid-day resume の
            # 監視 14 列が straight と食い違う(窓が空から再充填されるため)。
            # processed カウンタは**プロセス内 logger 由来**なので保存しない(load 側で 0)。
            # 集合は 1 つも使わず deque/dict/Counter のみ = pickle の集合反復順非保存の
            # 影響を受けない。既定 OFF(observer.regression.enabled=false)では state 自体が
            # 生えない → None=挙動不変(load は .get で旧 checkpoint 互換)。
            "regression_state": getattr(sim, "_regression_state", None),
            # 第74バッチ IDEA③(規範化ステージ): 語ごとの 4 段階台帳(素の dict/list のみ)。
            # 既定 OFF(labeling.norm_stage.enabled=false)では state 自体が生えない → None=
            # 挙動不変。ON のときは L2 の 2 列が累積量なので、保存しないと mid-day resume の
            # L2 が straight と食い違う(第62 joint / 第70 echo と同じ型のギャップ)。
            # processed カウンタは**プロセス内 logger 由来**なので保存しない(load 側で 0)。
            "norm_state": getattr(sim, "_norm_state", None),
            # 第75バッチ IDEA⑤(ダンバー認知枠): 活性関係数の日次スナップ+休眠/再会の累積カウンタ
            # (int/float のみ)。**休眠状態そのもの**(rel の dormant/dormant_closeness/
            # dormant_step)と直近接触 step(rel の last_step)は agents pickle に自然同梱される
            # ので、ここで中央管理するのは L2 3 列の材料だけ。保存しないと mid-day resume の
            # dormant_total / rekindle_total(累積量)が straight と食い違う(第62 joint /
            # 第70 echo / 第74 norm_state と同じ型のギャップ)。既定 OFF では state 自体が
            # 生えない → None=挙動不変(load は .get で旧 checkpoint 互換)。
            "dunbar_state": getattr(sim, "_dunbar_state", None),
            # 第86バッチ day_plan v1: 計画/ブロック/修復/後退/削除/ずらし/再計画の累積タリー
            # (int/float と素の dict のみ=集合なし=決定論監査は自明)。**計画そのもの**
            # (agent._dayplan / _dayplan_prev)は agents pickle に自然同梱されるので、ここで
            # 中央管理するのは L2 7 列と summary.day_plan の材料だけ。保存しないと mid-day
            # resume で累積列が straight と食い違う(第62 joint / 第70 echo / 第75 dunbar と
            # 同じ型のギャップ)。既定 OFF では state 自体が生えない → None=挙動不変
            # (load は .get で旧 checkpoint 互換)。
            "dayplan_state": getattr(sim, "_dayplan_state", None),
            # 第87バッチ engaged モード: エピソード数/ターン/滞在分/脱出理由の累積タリー
            # (int と素の dict のみ)。**エピソードそのもの**(agent._engaged)と不応期
            # (agent._engaged_refr)は agents pickle に自然同梱されるので、ここで中央管理
            # するのは L2 5 列と summary.engaged の材料だけ(第75 dunbar / 第86 day_plan と
            # 同じ型)。保存しないと mid-day resume で累積列が straight と食い違う。
            # 既定 OFF では state 自体が生えない → None=挙動不変(load は .get で旧 ckpt 互換)。
            "engaged_state": getattr(sim, "_engaged_state", None),
            # 第94バッチ IF-B(拒否通知): 通知件数・前倒し件数・縮退件数・失敗理由の
            # 設定/降格/注入の累積タリー(int と素の dict のみ)。**失敗理由そのもの**
            # (agent._plan_exception_why)と例外の印(agent._plan_exception)は
            # agents pickle に自然同梱されるので、ここで中央管理するのは
            # summary.rejection_notify の材料だけ(第86 day_plan / 第87 engaged と同じ型)。
            # 保存しないと mid-day resume で summary の累積値が straight と食い違う。
            # 既定 silent では state 自体が生えない → None=挙動不変(load は .get で
            # 旧 checkpoint 互換)。
            "reject_state": getattr(sim, "_reject_state", None),
            # 第95バッチ IF-C(噂 = 情報オブジェクトの一般化): 噂 Item の台帳
            # (item_id → 源イベント種 / ノード / 誕生 step / 発端者 / 到達人数)と
            # 累積タリー・忘却掃引の日境界。**誰が何を知っているか**(agent._rumors)は
            # agents pickle に自然同梱され、**Item 実体**(text / transmissions)は
            # 既に "labels"(= sim.items と同一の ItemStore)へ同梱されているので、
            # ここで中央管理するのは summary.rumors の材料と日カウンタだけ
            # (第75 dunbar / 第94 IF-B と同じ型)。保存しないと mid-day resume で
            # reach / born / stiflers が straight と食い違い、忘却掃引が同じ日を二重に走る。
            # watermark(_rumor_watermark)はプロセス内 logger カウンタ由来なので**保存しない**
            # (load で 0 に戻す = 第59 assets / 第61 gossip と同流儀)。
            # 既定 OFF では state 自体が生えない → None=挙動不変(load は .get で旧 ckpt 互換)。
            "rumor_state": getattr(sim, "_rumor_state", None),
            # 第96バッチ IF-D(痕跡 = 場所のイベント履歴): 痕跡ストア本体
            # (node → 種 → 強度 / 最終集約 step / 当事者名簿)と、蒸発の日境界・
            # 前回蒸発 step・累積タリー。★噂(IF-C)や dunbar と違い、**状態の本体が
            # sim 側にある**(場所に紐づくので agents pickle には載らない)= ここで
            # 保存しないと resume 直後に街の痕跡が全部消え、L1 の trace_mark も
            # プロンプトの 1 行も straight と食い違う。半減期減衰は「前回蒸発からの
            # 経過 step」で計算するので、evap_step を保存すれば日境界を跨いだ resume でも
            # straight と同じ強度に落ち着く。watermark(_trace_watermark)はプロセス内
            # logger カウンタ由来なので**保存しない**(load で 0 に戻る = rumors と同流儀)。
            # 既定 OFF では state 自体が生えない → None=挙動不変(load は .get で旧 ckpt 互換)。
            "trace_state": getattr(sim, "_trace_state", None),
            # H5(事件レイヤーの環境側 3 族): 進行中の火災(鎮火予定 step・重度・出場した
            # 隊員)・群集の閾値跨ぎの武装フラグ・日次カウンタ・累積タリー。**状態の本体が
            # sim 側にある**(場所と世界の出来事に紐づくので agents pickle には載らない)=
            # ここで保存しないと resume 直後に燃えている火が消え、群集の「1 エピソード 1 件」
            # が二重に出る。watermark は持たない機構なので保存する状態はこれで全部。
            # 既定 OFF では state 自体が生えない → None=挙動不変(load は .get で旧 ckpt 互換)。
            "incenv_state": getattr(sim, "_incenv_state", None),
            # H5(設備 = 摩耗する装置): **DEVS 状態そのもの**(台ごとの wear / uses /
            # 故障中か / 復旧予定の絶対時刻 / 次の保守の絶対時刻)と累積タリー。装置は
            # sim.facilities(名簿オブジェクト)に居るので pickle には載せず、素の dict へ
            # 落として保存する(load 側が復元値から名簿を組み直す)。保存しないと resume で
            # 全台の摩耗が 0 に戻り、故障ハザードが straight と食い違う。
            "facility_state": _facility_mod.state_of(sim),
            # H3(遺失物ループ): **進行中の遺失物そのもの**(どの品目がどこに落ちていて、
            # 誰が落とし主で、中にいくら入っていて、いま拾われているのか届いているのか)と
            # 累積タリー・通し番号・現金の在庫。★痕跡(IF-D)と同じく**状態の本体が sim 側に
            # ある**(物は場所に紐づくので agents pickle には載らない)= 保存しないと resume
            # 直後に街の落とし物が全部消え、しかも財布の中の現金が**世界から蒸発する**
            # (所持金から分離済みなので、在庫を失うと総マネー保存が破れる)。時効・寿命は
            # 「落ちた step からの経過」で計算するので、seq と items を戻せば日境界を跨いだ
            # resume でも straight と同じ決着になる。
            # 既定 OFF では state 自体が生えない → None=挙動不変(load は .get で旧 ckpt 互換)。
            "lost_state": getattr(sim, "_lost_state", None),
            # H4(対人事件の収束化): 累積タリー(候補・共在ペア・抽選・事件・通報・応答)と
            # パターン検収の材料(被害者別・ノード別の件数)。★事件そのものは L1 に確定して
            # いるが、summary.incidents_interpersonal の分母(共在機会比)と反復被害/場所集中の
            # 集計は sim 側にしか無いので、保存しないと mid-day resume で straight と食い違う
            # (第96 IF-D の trace_state と同じ理由)。★酩酊の印(agent._inc_intox_until)は
            # agents pickle に自然同梱される。既定 OFF では state 自体が生えない → None=挙動不変。
            "incident_state": getattr(sim, "_inc_state", None),
            # 所有権レイヤー O1+O3(登記簿そのもの): 資産レコード・権利行・双方向索引・初期
            # ストック・移転タリー・相続済みの故人 id・国庫へ出た現金。★**状態の本体が sim 側に
            # ある**(所有は場所でも個体でもなく世界の登記簿なので agents pickle には載らない)=
            # 保存しないと resume 直後に街の所有関係が全部消え、しかも初期配賦が**もう一度**
            # 走って新 stream "asset_alloc" を二重に引く(= 資産保存則も決定論も同時に破れる)。
            # 第96 traces / H3 遺失物 / 第97 IF-E2 と同じ型の中央管理。watermark(_asset_watermark)
            # はプロセス内 logger カウンタ由来なので**保存しない**(load で 0 に戻る = rumors 流儀。
            # 相続の二重適用は heired 集合が state 側で防ぐので、戻っても無害)。
            # 既定 OFF では state 自体が生えない → None=挙動不変(load は .get で旧 ckpt 互換)。
            # ★キー名/属性名を "asset_ledger" / "_ledger_state" にしてあるのは、第59 の資産
            #   **レンズ**(observer/assets.py)が既に "assets_state" / sim._assets_state を
            #   使っているため(所有の台帳と資産分布の観測レンズは別物)。
            "asset_ledger": getattr(sim, "_ledger_state", None),
            # 存在の内生化 POP(転出・転入=定着昇格・出生の台帳)。★**状態の本体が sim 側に
            # ある**(「誰がもう名簿に居ないか」「誰が住民に昇格したか」は個体ではなく世界の
            # 台帳)= 保存しないと resume 直後に転出者が街へ戻り、定着者が来街者へ落ち、
            # 新生児 id が振り直される(= 人口会計も決定論も同時に破れる)。日境界の進行
            # (state["day"])も同じ dict に入っているので mid-day resume の二重発火も防げる。
            # 第109 asset_ledger / 第97 IF-E2 と同じ型の中央管理。既定 OFF では state 自体が
            # 生えない → None=挙動不変(load は .get で旧 checkpoint 互換)。
            "pop_state": _population.checkpoint_state(sim),
            # 第88バッチ 心モデル固定: L1 `mind_assign` を出し終えた agent_id。
            # **割当そのものは保存しない** — 誕生時固定は (master_seed, agent_id) の純関数
            # (専用 stream mind_model / mind_tier)なので resume でも pool 再入場でも同じ
            # モデルに戻り、agent.mind は agents pickle に自然同梱される。ここで中央管理
            # するのは「もう記録したか」だけで、これが無いと resume 直後に全員ぶんの
            # mind_assign を二重記録して resume≠straight になる(第80 W2 と同じ型)。
            # 既定 OFF では空集合 = 何も起きない。sorted list で保存する(集合の反復順は
            # pickle 間で保証されないため=決定論監査の作法)。
            "mind_logged": sorted(getattr(sim, "_mind_logged", None) or ()),
            # 第89バッチ(プラセボ L1): context_shuffle のドナー輪(節種ごと FIFO・上限固定)と
            # 観測カウンタ。輪は「同一ラン内の直近の同種節」なので**跨ぐ状態**であり、保存しないと
            # resume 直後の輪が空=最初の数プロンプトが sever へ後退して resume≠straight になる。
            # ペルソナ対合表は構築時の名簿からの決定論導出なので保存しない(load 側で組み直す)。
            # プラセボ OFF のランでは None=挙動不変(load は .get で旧 checkpoint 互換)。
            "placebo_state": _ablate_ckpt.placebo_state(sim),
            # 第73バッチ(真偽台帳 Part B): sim 側の台帳=fact レコード(_tl_facts)・重複判定キー
            # (_tl_keys)・累積カウンタ(_tl_stats)。**保存しないと resume 後に fact_id が振り直され、
            # 信念(agents pickle に自然同梱される _fact_beliefs)の参照先が全部迷子になる**
            # (第62 joint / 第70 echo と同型の「日次/累積状態の未保存」ギャップ)。watermark は
            # プロセス内 logger カウンタ由来なので保存しない(load 側で 0 に戻す=assets/gossip 流儀)。
            # beliefs OFF のランでは属性自体が生えない → None=挙動不変(load は .get で旧 ckpt 互換)。
            "tl_facts": getattr(sim, "_tl_facts", None),
            "tl_keys": getattr(sim, "_tl_keys", None),
            "tl_stats": getattr(sim, "_tl_stats", None),
            # 第80バッチ W2(検収で顕在化した**既存ギャップ**): 日付・天気の日境界進行
            # (_cal_day)と当日確定値(date_line / weather / weather_line)。従来未保存
            # だったため、weather.enabled=true の mid-day resume は再開直後の 1 step で
            # _phase_calendar_weather が同じ日を再処理し、**weather イベントを二重記録**
            # (+ rain_grievance>0 なら不快感を二重加算)していた=resume≠straight。
            # 第62 joint / 第70 echo / 第75 dunbar と同じ型のギャップ。天気の値そのものは
            # 日インデックスの決定論関数なので保存しない(復元は当日値のキャッシュのみ)。
            # calendar/weather OFF のランでは -1/None = 挙動不変(load は .get で旧 ckpt 互換)。
            "cal_day": getattr(sim, "_cal_day", -1),
            "today_date_line": getattr(sim, "today_date_line", None),
            "today_weather": getattr(sim, "today_weather", None),
            "today_weather_line": getattr(sim, "today_weather_line", None),
            # 第81バッチ: 認知イベントキューの状態(pending = {agent_id: {at, reason, tok}})。
            # ★これを保存しないと resume 直後に「全員が今すぐ期限」の初期化に戻り、発火の
            # スケジュール列が straight と食い違う(= 新不変量「スケジュール列が同一なら
            # 世界が同一」が resume で破れる)。第62 joint / 第70 echo / 第75 dunbar / 第80 W2 と
            # 同じ型の中央管理。ヒープ本体は保存しない — 順序は pending の内容だけで決まるので
            # load 側で決定論的に再構築できる。期待値 ô(_fire_pred)・凍結観測(_fire_obs)・
            # 前 tick のゲージ(_fire_prev_drive)は agents pickle に自然同梱される。
            # cognition.fire OFF のランでは None = 挙動不変(load は .get で旧 ckpt 互換)。
            "cogq": (sim.cogq.state() if getattr(sim, "cogq", None) is not None
                     else None),
            # 第84バッチ: 環境フィードバックの状態(遅延[分]・入場規制の解除/クールダウン step・
            # 待ち行列で除外中の POI ノード → 解除 step・累積カウンタ)。**これを保存しないと
            # resume 直後に遅延 0・規制なしへ戻り、環境の履歴(回復運転で減衰していく途中の値)が
            # 消える**= resume≠straight。第62 joint / 第70 echo / 第75 dunbar / 第80 W2 と同じ型の
            # 中央管理。個体側の待ち(_env_exit_wait / _env_exit_held / _env_ret_*)は agents pickle
            # に自然同梱される。POI ノード除外は dict(集合ではない)ので反復順は決定論。
            # env.feedback OFF のランでは属性自体が生えない → None=挙動不変(load は .get で
            # 旧 checkpoint 互換)。
            "envfb_state": getattr(sim, "_envfb", None),
            # 竹-4(P3 境界縫合): ゲート通過の累計・境界連続性ヒストグラム・重なり最小値。
            # **ゾーンの所有そのもの**(agent._phys_zone / _phys_pos / _phys_vel / _phys_path …)は
            # agents pickle に自然同梱されるので、ここで中央管理するのは L2 の累積列と
            # 検収指標の材料だけ(第62 joint / 第70 echo / 第75 dunbar と同じ型)。
            # 保存しないと mid-day resume で zone_gate_*_total(累積量)が straight と食い違う。
            # physics.zones OFF のランでは属性自体が生えない → None=挙動不変
            # (load は .get で旧 checkpoint 互換)。
            "phys_state": getattr(sim, "_phys_state", None),
            # 第82バッチ: θ 恒常性の日境界の進行(**日オーダーの時定数**なので、これを
            # 保存しないと resume 直後に「初日扱い」へ戻って 1 日ぶんの恒常性が飛ぶ)。
            # g / ē / credit / 監視仕様(_fire_watch)は agents pickle に自然同梱される。
            "g_day": int(getattr(sim, "_g_day", -1)),
            # 第97バッチ IF-E2 案B(org の会計主体化 + rest-of-world): org のスカラー預金・
            # RoW のチャネル別累積・行政の預り金(escrow)・受け手解決の段別件数。
            # ★状態の本体が sim 側にある(場所でも個体でもなく世界の会計)ので、保存しないと
            #   resume 直後に全 org の預金が消え、期首配賦が二重に走り、不変量
            #   「Σ(全主体残高)+RoW 累積=一定」が resume で必ず破れる(第96 traces と同じ型)。
            # 既定 OFF では state 自体が生えない → None=挙動不変(load は .get で旧 ckpt 互換)。
            "sfc_state": getattr(sim, "_sfc_state", None),
            # H2 医療(搬送・入院・医療費の累計タリー)。**個体の在院状態は agent pickle に
            # 自然同梱される**ので、ここで中央管理するのは世界側のカウンタだけ
            # (第96 traces / H3 遺失物 / H4 対人事件と同じ型)。OFF では属性自体が生えない
            # → None = 挙動不変(load は .get で旧 checkpoint 互換)。
            "medical_state": getattr(sim, "_med_state", None),
            # ★同バッチで塞ぐ**既存欠陥**: 非エージェントの残高 3 つ(行政 / 銀行 / VC ファンド)は
            #   どれも checkpoint に入っておらず、resume で初期値へ戻っていた(研究文書 §3-0 の
            #   「重要な構造的事実」)。いずれも sim 参照を持たない plain データなので
            #   そのまま同梱できる(government.py の docstring が「将来 checkpoint 同梱も可能」と
            #   予告していた形)。org 預金だけ保存して行政が戻ると非対称なバグになるため、
            #   IF-E2 の前提工事として同時に塞ぐ。未構築のランでは None=従来どおり。
            "government": getattr(sim, "government", None),
            "bank": getattr(sim, "bank", None),
            "vc_fund": getattr(sim, "vc_fund", None),
            # ------------------------------------------------------------------
            # 第98バッチ 小粒A(resume 整合の**全数**監査): 日/期ガードの未保存を一括で塞ぐ。
            # これまで各バッチが「自分の機構の日カウンタ」を1件ずつ足してきた結果、同型の
            # 未保存が src 全体に散らばって残っていた(AST で sim.<attr> 代入を全数走査して確定)。
            # どれも第80 W2(weather の二重記録)と同じ型 —「mid-day resume が同じ暦日を
            # もう一度処理する」= 二重発火 / 二重記録で resume≠straight になる。
            # OFF ランでは -1/None/0 = 挙動不変(load は .get で旧 checkpoint 互換)。
            "bank_day": getattr(sim, "_bank_day", -1),        # E-W1 融資の日次返済(二重返済)
            "reflect_day": getattr(sim, "_reflect_day", -1),  # 第12 無意識層の日次更新(EMA 二重適用)
            "freedom_day": getattr(sim, "_freedom_day", -1),  # 第17 価値充足の中立回帰(二重減衰)
            "sched_day": getattr(sim, "_sched_day", -1),      # 予定の日次 GC
            "partner_day": getattr(sim, "_partner_day", -1),  # H2 パートナー形成(同日二重成立)
            "health_day": getattr(sim, "_health_day", -1),    # G6 発症/受診の日次抽選(二重抽選)
            # 偶発イベント: stream キーが (agent.id, **day**) なので同じ日を引き直すと
            # **同じ当たりがもう一度効く**(windfall の二重入金)= 特に害が大きい。
            "chance_day": getattr(sim, "_chance_day", -1),
            "goods_review_day": getattr(sim, "_goods_review_day", -1),      # 物流の補充レビュー(二重補充)
            "council_budget_day": getattr(sim, "_council_budget_day", -1),  # 議会の区予算承認(二重記録)
            "vc_review_day": getattr(sim, "_vc_review_day", -1),            # E-W2 VC 定期審査(二重出資)
            "crowd_surge_day": getattr(sim, "_crowd_surge_day", -1),        # 群集の 1日1回記録
            # 年中行事: 日カウンタ**だけ**戻すと today_event_line が None のまま当日が終わり
            # プロンプト文脈が消えるので、第80 W2(cal_day + today_weather 3 点)とまったく
            # 同じ様式で「当日の確定値」も一緒に運ぶ。行事の発生自体は暦の純関数。
            "annual_day": getattr(sim, "_annual_day", -1),
            "today_event_line": getattr(sim, "today_event_line", None),
            "today_crowd_event": getattr(sim, "today_crowd_event", None),
            # 合成地位(T6): 日カウンタ + **前日 rank**(L2 status_rank_mobility の材料)。
            # status 値そのものは agents pickle に自然同梱。prev_rank を落とすと翌日の移動性が
            # 0.0 に潰れ、mobility 自体も再開直後から当日の残り step で 0.0 に化ける。
            "status_day": getattr(sim, "_status_day", -1),
            "status_prev_rank": getattr(sim, "_status_prev_rank", None),
            "status_rank_mobility": getattr(sim, "_status_rank_mobility", None),
            # 世界観(C2/C6): 日カウンタ + 規範窓(直近 norm_window_days 日の開拓行動数 deque)。
            # 日カウンタが無いと resume 直後の日境界が **first 扱い**になり、その日の worldview
            # スナップショット(世界1件 + 全員1件)がまるごと落ちる。窓は日を跨ぐ状態なので運ぶ。
            # 走査位置(_wv_pos)は**プロセス内 logger カウンタ由来**なので保存しない(load で 0
            # = assets/gossip と同流儀)。
            "wv_day": getattr(sim, "_wv_day", -1),
            "wv_pioneer": list(getattr(sim, "_wv_pioneer", None) or ()),
            # 都市・環境ショック(H4): 日カウンタ + **継続中の災害/インフラ障害そのもの**
            # (種・解除日)+ 当日のプロンプト1行 + 電車の運休フラグ。★状態の本体が sim 側に
            # ある(第96 traces と同じ型)ので、保存しないと resume 直後に災害が消え、電車が
            # 勝手に復旧し、同じ日にもう一度 onset 抽選が走る。個体側の在宅フラグ
            # (_disaster_homebound)は agents pickle に自然同梱。OFF では属性自体が生えない
            # → None=挙動不変(load は .get で旧 checkpoint 互換)。
            "disaster_state": ({
                "day": int(getattr(sim, "_disaster_day", -1)),
                "active": bool(getattr(sim, "_disaster_active", False)),
                "kind": getattr(sim, "_disaster_kind", None),
                "until_day": int(getattr(sim, "_disaster_until_day", -1)),
                "infra_active": bool(getattr(sim, "_infra_active", False)),
                "infra_kind": getattr(sim, "_infra_kind", None),
                "infra_until_day": int(getattr(sim, "_infra_until_day", -1)),
                "line": getattr(sim, "today_disaster_line", None),
                "suspended": bool(getattr(getattr(sim, "transit", None),
                                          "suspended", False)),
            } if hasattr(sim, "_disaster_day") else None),
            # 議会(名簿制 / SNTV)と立候補受付。**非エージェントの plain データ**で、第97 が
            # 塞いだ government / bank / vc_fund とまったく同じ型の**既存欠陥** — 保存しないと
            # resume 直後に議会が消え、_assembly_realism が同じ日に改選をやり直して
            # council_elected / election_result を二重記録し、任期(term)も打ち直される。
            "council": getattr(sim, "council", None),
            "council_campaign": getattr(sim, "council_campaign", None),
            # 卸(B2B)台帳・宅配の到着待ち・累積カウンタ 3 種(素の dict/list/int)。
            # b2b の卸在庫は**物**なので落とすと resume で在庫が湧き欠品が消える。delivery の
            # 到着待ちは「金は払ったが品はまだ」の中間状態=落とすと注文が宙に消える。
            # 累積 3 種は L2 の常設列(第62 joint_total と同じ型)。
            "b2b": getattr(sim, "_b2b", None),
            "b2b_total": int(getattr(sim, "_b2b_total", 0)),
            "delivery_pending": getattr(sim, "_delivery_pending", None),
            "delivery_total": int(getattr(sim, "_delivery_total", 0)),
            "service_total": int(getattr(sim, "_service_total", 0)),
            # 屋外広告(OOH)のキャンペーン期。改定は期境界で 1 枠 1 draw なので、保存しないと
            # resume 直後に同じ期のキャンペーンを引き直す(ad_campaign の二重記録 + 別 step キー
            # からの draw)。枠(_ads_slots)は city からの決定論導出なので保存しない
            # (street.py 側が resume で期を上書きしないよう getattr 既定に直してある)。
            "ads_period": getattr(sim, "_ads_period", None),
            "ads_campaigns": getattr(sim, "_ads_campaigns", None),
            # 内面本格版(H6)の「起動後 1 回」フラグ。第88 mind_logged と同じ型 — 保存しないと
            # resume 直後に precompute が再走し、全員ぶんの long_goal を二重記録する
            # (目標/趣味の値は決定論なので同じ。記録だけが二重になる)。
            "inner_life_init": bool(getattr(sim, "_inner_life_init", False)),
            # ------------------------------------------------------------------
            # 第98バッチ W4-A(resume 整合の完遂): 観測レンズの**暦日タリー**4 本。
            # どれも echo(第70)/ norm(第74)/ regression(第91)とまったく同じ型 ——
            # 「logger を増分走査して当日ぶんを積む」観測で、state を運ばないと mid-day
            # resume の当該 L2 列が straight と食い違う(当日タリーが途中から数え直しに
            # なる。silence.py の docstring が「resume の限界(正直註記)」として明記して
            # いたもの)。watermark(_lens_processed / _silence_processed / _struct_processed
            # / _dev_processed)は**プロセス内 logger カウンタ由来**なので保存しない
            # (load で 0 に戻す = assets / gossip / echo と同流儀)。
            # ★凍結 SPEC_FILES(lens/silence/structure/deviation)は 1 バイトも触らない —
            #   状態は全て sim 側属性なので、保存も復元も本 module だけで完結する。
            # 決定論: _silence_state["speakers"] は集合だが membership と len でしか使われず
            # (silence.py 明記)、他は dict/int のみ = pickle の集合反復順非保存の影響なし。
            # 既定 OFF では state 自体が生えない → None=挙動不変(load は .get で旧 ckpt 互換)。
            "lens_state": getattr(sim, "_lens_state", None),       # 価値4軸 / 3M欲望の当日構成
            "silence_state": getattr(sim, "_silence_state", None),  # 当日の発話者集合・熟慮/沈黙件数
            "struct_state": getattr(sim, "_struct_state", None),   # 当日の edge 形成/断絶/風化 + 母数
            "dev_state": getattr(sim, "_dev_state", None),         # 当日の住民別逸脱 + 職業 map
            # ★正直な限界(保存では**解決しない**もの): silence の `undefined_action_total` /
            #   `undefined_action_rate` は sim ではなく **logger のプロセス内カウンタ**
            #   (logger.n_undefined_action_events ÷ LLM 総呼数)を読む設計で、llm_health 3 列
            #   (llm_fallback_rate 等)と同じ「プロセス開始からの累積」族。族ごと watermark を
            #   再設計しない限り一貫させられないため**あえて保存しない** → freedom.
            #   undefined_register=true のランでは resume 後のこの 2 列は straight と食い違う
            #   (0 から数え直す)。ON にするなら日境界での checkpoint を推奨。
            # 世界観 C2/C6 の「直前の日境界 〜 checkpoint」区間(worldview の走査は日境界に
            # しか走らないので、日カウンタ wv_day を戻すだけでは**この区間が丸ごと落ちる**)。
            # 保存するのはイベント本体ではなく**有界な射影**(応答 3 つ組列 + 開拓行動数)。
            # 生成は純粋な読み取り = straight ラン(分割 straight を含む)は完全に不変。
            # 既定 OFF では None=挙動不変(load は .get で旧 checkpoint 互換)。
            "wv_carry": _wv_ckpt.checkpoint_state(sim),
            # 火種介入 spark の t=0 名簿イベントを**もう記録したか**(第88 mind_logged /
            # scheduler の _workbind_reported と同型の「起動時 1 回」フラグ)。spark.apply は
            # Simulation.__init__ から呼ばれるので resume した 2 つ目のプロセスでも走り、
            # 同じ spark_roster を二重に記録していた(load 側でこのフラグを見て抑止する)。
            # spark OFF では False=挙動不変(load は .get で旧 checkpoint 互換)。
            "spark_reported": bool(getattr(sim, "_spark_reported", False)),
            # ------------------------------------------------------------------
            # レーン D1(第109): resume 二重発火の**総括的**な根治。
            # 第98 W4-A は spark_roster を**1 種だけ**名指しで塞いだが、
            # Simulation.__init__ が L1 へ出す「起動時 1 回」の発火体は他にもあり
            # (friend_graph_built / party の joint_activity{type:party} / icebreak …)、
            # relations+friend_graph+joint+party+pool を同時 ON にした第109 縦煙で
            # 初めて表に出た(+106 行 = +0.89%。joint_activity が +97)。
            # ★種の名指しを止め、「__init__ が出し終えた時点のバッファ長」を運ぶ。
            #   load 側はその本数を先頭から捨てる(= 前のプロセスが part へ確定済み)。
            #   将来 __init__ に発火体が増えても自動的に守られる。
            # 旧 checkpoint(キー無し)= 0 = 従来挙動(spark の名指しフィルタだけが効く)。
            "init_events": int(getattr(sim, "_init_event_mark", 0)),
            # 装置層: ``signal_summary`` の「1 交差点 1 日 1 件」ガード。第80 W2(weather の
            # 二重記録)と同型の未保存で、resume 直後にその日の要約をもう一度出していた
            # (第109 縦煙の実測 +1 件)。dict のみ = 集合反復順非保存の影響なし。
            # 既定 OFF(装置層 OFF)では属性自体が生えない → None=挙動不変。
            "dev_signal_day": getattr(sim, "_dev_signal_day", None),
            # 情報環境(バイラル / 誤情報): 走査済み投稿の**絶対 id** watermark と
            # 加重済みの元投稿 id。★他の watermark と違いこれは logger ではなく
            # **sim.net の投稿 id 空間**由来なので「load で 0 に戻す」流儀が通用しない
            # (net は pickle で丸ごと戻るのに watermark だけ 0 に戻ると、既に処理済みの
            # 投稿を全部走査し直す)。実測: viral_cascade を再発火して reshares を
            # **二重加算**し、misinfo を別 step キーで引き直して炎上を捏造していた
            # (第109 縦煙の viral_cascade +3 / misinfo +2 / state_update +1)。
            # 集合は sorted list で保存する(反復順は pickle 間で保証されない = 決定論の作法)。
            "infoenv_watermark": int(getattr(sim, "_infoenv_watermark", 0)),
            "viral_done": sorted(getattr(sim, "_viral_done", None) or ()),
            # 世帯の夕食共食(S-R1): 記録済みの (household_id, day)。第109 縦煙の 48 step
            # では夕食帯(18:00-21:00)に届かず露見しなかったが、**同じ型**の未保存である
            # (帯の途中で resume すると同じ世帯の joint_activity{family_dinner} が
            # もう 1 件出る)。household.family_dinner OFF では空集合 = 挙動不変。
            "dinner_logged": sorted(getattr(sim, "_dinner_logged", None) or ()),
            # 第71バッチ: LLM 入出力ジャーナルの確定点(ファイル名 → {records, bytes})。
            # mark() が flush してから採るので、この時点のファイル末尾は必ず gzip メンバ境界
            # = 安全な切り詰め点。resume(load)がここまで巻き戻すことで、「checkpoint 後に
            # 走ってクラッシュした分」を再走したときの二重記録と seq の巻き戻りを両方防ぐ。
            # journal OFF のランでは {} = 挙動不変(load は .get で旧 checkpoint 互換)。
            "llm_journal": (sim._journal_marks()
                            if hasattr(sim, "_journal_marks") else {}),
        },
        # --- scenario は config から再構築される。封鎖の進行だけを直列化(順序安定) ---
        "scenario": {
            "active": bool(sim.scenario.active),
            "closed": sorted(sim.scenario.closed),   # set[tuple] → sorted list
        },
        # --- traffic は city/router/hub 付きで再構築される。可変ランタイムのみ ---
        "traffic": {
            "cars": sim.traffic.cars,
            "n_active": sim.traffic.n_active,
            "last_n": sim.traffic.last_n,
            "total_spawned": sim.traffic.total_spawned,
        },
    }
    raw = pickle.dumps(blob, protocol=pickle.HIGHEST_PROTOCOL)
    tmp = path.with_name(path.name + ".tmp")
    with gzip.open(tmp, "wb") as f:
        f.write(raw)
    os.replace(tmp, path)
    return path


def load(sim, path: str | Path) -> int:
    """checkpoint を sim へ上書き復元し、次に実行すべき step を返す。"""
    with gzip.open(Path(path), "rb") as f:
        blob = pickle.loads(f.read())
    if blob.get("format") != FORMAT:
        raise ValueError(f"未知の checkpoint format: {blob.get('format')}")
    expect = config_hash(sim.cfg)
    if blob.get("config_hash") != expect:
        raise ValueError(
            "checkpoint の config が現在の config と不整合(決定論が壊れる)。"
            " seed/n_agents/因子など resume 対象外のキーが変わっている可能性。"
            " ★run.dt_min≠10 のランで出た場合はまず Δt の二重変換を疑うこと:"
            " 保存済み config.yaml は apply_dt 済みなので、読み直すときは"
            " load_config(path=…, apply_dt=False) でなければ全定数が二重に変換される"
            "(scripts/run.py --resume は対応済み)。")

    # 状態オブジェクト(共有参照は 1 pickle 内で保存済み)
    sim.agents = blob["agents"]
    sim.agent_by_id = {a.id: a for a in sim.agents}
    sim.labels = blob["labels"]
    sim.items = sim.labels.items            # ItemStore を labels と同一実体に固定
    sim.net = blob["net"]
    sim.tools = blob["tools"]
    sim.rulebook = blob["rulebook"]         # 制度DSL: active ルールを跨いで復元
    sim.world_events = blob["world_events"]

    rt = blob["runtime"]
    sim.drive_stats = rt["drive_stats"]
    sim._econ_day = rt["econ_day"]
    sim._acct_day = rt.get("acct_day", -1)      # 旧 checkpoint 互換(無ければ -1)
    sim._org_day = rt.get("org_day", {})        # B4: 会社観測 per-day アキュムレータ(旧 checkpoint 互換)
    sim._org_ledger_day = rt.get("org_ledger_day", -1)
    sim._work_day = rt.get("work_day", -1)      # 第97: オフィス産出の日境界(旧 ckpt 互換)
    sim._housing_day = rt.get("housing_day", -1)  # 第60バッチ b: 転居/同棲の日境界進行(旧 checkpoint 互換)
    sim._career_day = rt.get("career_day", -1)    # 第60バッチ b: career 日境界進行(旧 checkpoint 互換)
    sim._rel_day = rt.get("rel_day", -1)          # 第61検収補修: 社会関係 日境界進行(旧 checkpoint 互換)
    ast = rt.get("assets_state")                # 第59: 資産レンズ τ の前日状態(旧 checkpoint 互換=無ければ素通り)
    if ast:
        sim._assets_state = {"day": ast["day"], "prev": ast["prev"],
                             "tau": ast["tau"], "processed": 0}
    sim._gossip_day = rt.get("gossip_day", -1)  # 第61 c: 負の評判の日境界進行(旧 checkpoint 互換)
    gst = rt.get("gossip_state")
    if gst is not None:
        sim._gossip_state = gst
    gpd = rt.get("gossip_pending")
    if gpd is not None:
        sim._gossip_pending = gpd
    sim._gossip_watermark = 0                   # watermark は load で 0(fresh logger=total 0 から再走査。assets と同流儀)
    # 共同行動 第62バッチ: 日境界進行+当日グループ+累積件数+承諾内生化タリー(旧 checkpoint 互換=
    # 無ければ従来どおり -1/[]/0 で再編成へ後退)。
    sim._joint_day = rt.get("joint_day", -1)
    jg = rt.get("joint_groups")
    if jg is not None:
        sim._joint_groups = jg
    sim._joint_total = rt.get("joint_total", 0)
    est = rt.get("endo_state")
    if est is not None:
        sim._endo_state = est
    ivs = rt.get("invite_state")                # 第64: 誘い経路タリー(旧 checkpoint 互換=無ければ素通り)
    if ivs is not None:
        sim._invite_state = ivs
    qst = rt.get("quality_state")               # 第65: 会話の質タリー(旧 checkpoint 互換=無ければ素通り)
    if qst is not None:
        sim._quality_state = qst
    est_echo = rt.get("echo_state")             # 第70 IDEA①: エコー窓(旧 checkpoint 互換=無ければ素通り)
    if est_echo is not None:
        sim._echo_state = est_echo
    sim._echo_processed = 0                     # fresh logger=total 0 から再走査(assets と同流儀)
    sim._echo_cache = None
    est_reg = rt.get("regression_state")         # 第91 退行シグナル窓(旧 ckpt 互換=無ければ素通り)
    if est_reg is not None:
        sim._regression_state = est_reg
    sim._regression_processed = 0                # fresh logger=total 0 から再走査(echo と同流儀)
    sim._regression_cache = None
    nst = rt.get("norm_state")                  # 第74 IDEA③: 規範化ステージ台帳(旧 ckpt 互換=無ければ素通り)
    if nst is not None:
        sim._norm_state = nst
    sim._norm_processed = 0                     # fresh logger=total 0 から再走査(echo と同流儀)
    sim._norm_cache = None
    dst = rt.get("dunbar_state")                # 第75 IDEA⑤: 認知枠タリー(旧 ckpt 互換=無ければ素通り)
    if dst is not None:
        sim._dunbar_state = dst
    dps = rt.get("dayplan_state")               # 第86 day_plan v1(旧 ckpt 互換=無ければ素通り)
    if dps is not None:
        sim._dayplan_state = dps
    egs = rt.get("engaged_state")               # 第87 engaged(旧 ckpt 互換=無ければ素通り)
    if egs is not None:
        sim._engaged_state = egs
    rjs = rt.get("reject_state")                # 第94 IF-B(旧 ckpt 互換=無ければ素通り)
    if rjs is not None:
        sim._reject_state = rjs
    rms = rt.get("rumor_state")                 # 第95 IF-C(旧 ckpt 互換=無ければ素通り)
    if rms is not None:
        sim._rumor_state = rms
    trs = rt.get("trace_state")                 # 第96 IF-D(旧 ckpt 互換=無ければ素通り)
    if trs is not None:
        sim._trace_state = trs
    ies = rt.get("incenv_state")                # H5 環境事件(旧 ckpt 互換=無ければ素通り)
    if ies is not None:
        sim._incenv_state = ies
    # H5 設備(DEVS 状態)。復元値から**名簿を組み直す**ので、ここで名簿を捨てる
    # (旧 ckpt 互換=無ければ no-op = OFF ランは無風)。
    _facility_mod.restore_state(sim, rt.get("facility_state"))
    lps = rt.get("lost_state")                  # H3 遺失物(旧 ckpt 互換=無ければ素通り)
    if lps is not None:
        # items のキーは JSON を経由しない pickle なので int のまま戻る(sorted の決定論を保つ)
        sim._lost_state = lps
    ics = rt.get("incident_state")              # H4 対人事件(旧 ckpt 互換=無ければ素通り)
    if ics is not None:
        sim._inc_state = ics
    alg = rt.get("asset_ledger")                # 所有権 O1+O3(旧 ckpt 互換=無ければ素通り)
    if alg is not None:
        # init=True が戻るので arm は二度と走らない(= "asset_alloc" の二重消費が起きない)
        sim._ledger_state = alg
    sfc = rt.get("sfc_state")                   # 第97 IF-E2(旧 ckpt 互換=無ければ素通り)
    if sfc is not None:
        sim._sfc_state = sfc
    # H2 医療(旧 ckpt 互換=無ければ no-op = OFF ランは無風)。
    _medical_mod.restore_state(sim, rt.get("medical_state"))
    # 存在の内生化 POP(旧 ckpt 互換=無ければ no-op = OFF ランは属性も生えない)。
    _population.restore_state(sim, rt.get("pop_state"))
    # 非エージェント残高 3 つ(既存欠陥の修復)。旧 checkpoint / 未構築ランでは None=従来どおり
    # 遅延構築に任せる(= resume で初期値に戻る旧挙動)。
    for _attr in ("government", "bank", "vc_fund"):
        _obj = rt.get(_attr)
        if _obj is not None:
            setattr(sim, _attr, _obj)
    # 第89 プラセボ L1: 輪を戻し、pickle 復元された個体を構築時の Placebo へ繋ぎ直す
    # (OFF は no-op。旧 ckpt 互換=無ければ輪は空のまま=OFF ランと同じ)。
    _ablate_ckpt.placebo_restore(sim, rt.get("placebo_state"))
    mlg = rt.get("mind_logged")                 # 第88 心モデル固定(旧 ckpt 互換=無ければ素通り)
    if mlg is not None:
        sim._mind_logged = {int(x) for x in mlg}
    # 割当表(_mind_binding)は保存しない=誕生時固定が純関数なので agents の属性から復元される。
    from .. import mind as _mind_mod
    if _mind_mod.enabled(sim):
        sim._mind_binding = {int(a.id): m["model"] for a in sim.agents
                             if (m := getattr(a, "mind", None))}
    # 第73 Part B: 真偽台帳(旧 checkpoint 互換=無ければ素通り=beliefs OFF の挙動不変)。
    # canary は module 側のプロセス内レジストリなので、復元した fact 分を再登録して武装を保つ。
    tlf = rt.get("tl_facts")
    if tlf is not None:
        from ..truth_ledger import register_canary as _tl_canary
        sim._tl_facts = tlf
        for _fid in sorted(tlf):
            _tl_canary(_fid)
    tlk = rt.get("tl_keys")
    if tlk is not None:
        sim._tl_keys = tlk
    tls = rt.get("tl_stats")
    if tls is not None:
        sim._tl_stats = tls
    sim._tl_watermark = 0                       # fresh logger=total 0 から再走査(gossip と同流儀)
    # 第80バッチ W2: 日付・天気の日境界進行と当日確定値(旧 checkpoint 互換=無ければ従来どおり
    # -1/None=再開直後に当日を作り直す)。生成モードの日別系列は保存しない — 再構築が
    # (master_seed, params, 起点の日) の純関数で、生成は逐次=prefix 安定だから同一系列になる。
    sim._cal_day = rt.get("cal_day", -1)
    if "today_weather" in rt:
        sim.today_date_line = rt.get("today_date_line")
        sim.today_weather = rt.get("today_weather")
        sim.today_weather_line = rt.get("today_weather_line")
    # 第81: 認知イベントキュー(旧 checkpoint 互換=無ければ初期化のまま=OFF ランは無風)。
    if getattr(sim, "cogq", None) is not None:
        from ..cognition.fire import CogQueue as _CogQueue
        sim.cogq = _CogQueue.restore(rt.get("cogq"))
    # 第84: 環境フィードバックの状態(旧 checkpoint 互換=無ければ素通り=OFF ランは無風)。
    efb = rt.get("envfb_state")
    if efb is not None:
        sim._envfb = efb
    # 竹-4: 物理ゾーンのタリー(旧 checkpoint 互換=無ければ素通り=OFF ランは無風)。
    phs = rt.get("phys_state")
    if phs is not None:
        sim._phys_state = phs
    # 第82: θ 恒常性の日境界(旧 checkpoint 互換 = 無ければ -1 = 従来どおり初日扱い)。
    sim._g_day = int(rt.get("g_day", -1))
    # ---- 第98バッチ 小粒A: 日/期ガードの全数復元 --------------------------------
    # 旧 checkpoint 互換 = キーが無ければ -1/None/0 = **従来の挙動そのまま**(getattr 既定と同値)。
    sim._bank_day = rt.get("bank_day", -1)
    sim._reflect_day = rt.get("reflect_day", -1)
    sim._freedom_day = rt.get("freedom_day", -1)
    sim._sched_day = rt.get("sched_day", -1)
    sim._partner_day = rt.get("partner_day", -1)
    sim._health_day = rt.get("health_day", -1)
    sim._chance_day = rt.get("chance_day", -1)
    sim._goods_review_day = rt.get("goods_review_day", -1)
    sim._council_budget_day = rt.get("council_budget_day", -1)
    sim._vc_review_day = rt.get("vc_review_day", -1)
    sim._crowd_surge_day = rt.get("crowd_surge_day", -1)
    sim._annual_day = rt.get("annual_day", -1)
    if "today_crowd_event" in rt:               # 第80 W2 と同じ様式(当日の確定値も一緒に戻す)
        sim.today_event_line = rt.get("today_event_line")
        sim.today_crowd_event = rt.get("today_crowd_event")
    sim._status_day = rt.get("status_day", -1)
    spr = rt.get("status_prev_rank")            # 前日 rank(無ければ素通り=従来どおり mobility 0)
    if spr is not None:
        sim._status_prev_rank = spr
    srm = rt.get("status_rank_mobility")
    if srm is not None:
        sim._status_rank_mobility = float(srm)
    sim._wv_day = rt.get("wv_day", -1)
    wvp = rt.get("wv_pioneer")                  # 規範窓(deque。maxlen は conf から組み直す)
    if wvp:
        from collections import deque as _deque
        _wvcfg = getattr(sim, "worldviewcfg", None) or {}
        sim._wv_pioneer = _deque(
            wvp, maxlen=max(1, int(_wvcfg.get("norm_window_days", len(wvp)) or len(wvp))))
        sim._wv_pos = 0                         # 走査位置は logger 由来=0 から(assets/gossip 同流儀)
    dsr = rt.get("disaster_state")              # H4 ショック(旧 ckpt 互換=無ければ素通り=OFF は無風)
    if dsr is not None:
        sim._disaster_day = int(dsr["day"])
        sim._disaster_active = bool(dsr["active"])
        sim._disaster_kind = dsr["kind"]
        sim._disaster_until_day = int(dsr["until_day"])
        sim._infra_active = bool(dsr["infra_active"])
        sim._infra_kind = dsr["infra_kind"]
        sim._infra_until_day = int(dsr["infra_until_day"])
        sim.today_disaster_line = dsr["line"]
        if getattr(sim, "transit", None) is not None:
            sim.transit.suspended = bool(dsr["suspended"])   # 運休は日次に打ち直される派生値
    for _attr in ("council", "council_campaign"):   # 議会(government/bank/vc_fund と同型)
        _obj = rt.get(_attr)
        if _obj is not None:
            setattr(sim, _attr, _obj)
    _b2b = rt.get("b2b")
    if _b2b:                                    # 未構築の空 dict は素通り(simulation の初期値のまま)
        sim._b2b = _b2b
    sim._b2b_total = int(rt.get("b2b_total", 0))
    _dpn = rt.get("delivery_pending")
    if _dpn is not None:
        sim._delivery_pending = _dpn
    sim._delivery_total = int(rt.get("delivery_total", 0))
    sim._service_total = int(rt.get("service_total", 0))
    _adp = rt.get("ads_period")                 # OOH キャンペーン期(枠は city から組み直す)
    if _adp is not None:
        sim._ads_period = int(_adp)
        sim._ads_campaigns = rt.get("ads_campaigns") or {}
    if rt.get("inner_life_init"):               # H6 起動後 1 回(long_goal の二重記録を防ぐ)
        sim._inner_life_init = True
    # ---- 第98バッチ W4-A: 観測レンズの暦日タリー(旧 ckpt 互換=無ければ素通り=OFF は無風)----
    # watermark は 4 本とも 0 に戻す(fresh logger=total 0 から再走査。echo/gossip と同流儀)。
    lns = rt.get("lens_state")                  # 価値4軸 / 3M欲望の当日構成
    if lns is not None:
        sim._lens_state = lns
    sim._lens_processed = 0
    sls = rt.get("silence_state")               # 当日の発話者集合・熟慮/沈黙件数
    if sls is not None:
        sim._silence_state = sls
    sim._silence_processed = 0
    sts = rt.get("struct_state")                # 当日の edge 形成/断絶/風化 + 母数
    if sts is not None:
        sim._struct_state = sts
    sim._struct_processed = 0
    dvs = rt.get("dev_state")                   # 当日の住民別逸脱 + 職業 map
    if dvs is not None:
        sim._dev_state = dvs
    sim._dev_processed = 0
    # 世界観 C2/C6 の未走査区間(有界射影)。OFF / 旧 ckpt では no-op=従来挙動。
    _wv_ckpt.restore_checkpoint(sim, rt.get("wv_carry"))
    # ---- レーン D1(第109): 「起動時セグメント」を丸ごと捨てる ------------------------
    # ここに来た時点で sim.logger.events は **__init__ が出したイベントだけ**である
    # (run() は load の後で初めて step を回す)。それらは前のプロセスが同じ config で
    # 同じ順に出し、checkpoint と対で flush 済み = part に確定している。よって
    # 「前のプロセスが出した本数」だけ先頭から捨てるのが正しい(捨てる本数を保存値から
    # 採るので、万一 __init__ の出力が食い違えば残差として現れる = トリップワイヤーも兼ねる)。
    # ★これで friend_graph_built / party の joint_activity / spark_roster / icebreak …
    #   を**種ごとに名指しせず**に塞げる(第109 縦煙 +106 行の主因)。
    # ★副次効果: logger 増分走査の watermark(_dev_watermark / _facility_watermark /
    #   _rumor_watermark / _trace_watermark / lens 4 本 / worldview)は load で 0 に
    #   戻る設計なので、起動時セグメントが残っていると resume 直後にそれを**もう一度**
    #   走査していた。捨てることでこの経路も同時に閉じる。
    # 旧 checkpoint(キー無し)= 0 = 従来どおり(下の spark 名指しフィルタだけが効く)。
    _n_init = int(rt.get("init_events", 0) or 0)
    if _n_init > 0 and getattr(sim, "logger", None) is not None:
        del sim.logger.events[:min(_n_init, len(sim.logger.events))]
    # ---- 第98バッチ W4-A: 火種介入 spark_roster の resume 二重記録を塞ぐ --------------
    # spark.apply は **Simulation.__init__ から**呼ばれ、t=0 の名簿イベント spark_roster を
    # 1 件記録する。resume した 2 つ目のプロセスでも __init__ は丸ごと走るので、同じイベントが
    # もう一度バッファへ積まれ、結合後の L1 に 2 件並んでいた(実測)。scheduler 側の
    # `workplace_bound` が `step != 0` ガード + `_workbind_reported` で防いでいるのと**同型**の
    # ギャップだが、apply は step を見る術が無い(構築時=まだ step の概念が無い)ため、
    # 「前のプロセスが既に記録済み」という事実を checkpoint に載せて**ここで**ガードする。
    # 旧 checkpoint(キー無し)= False = 従来挙動。fresh ランは load を通らない=1 バイト不変。
    # log() は spark_roster でカウンタを 1 つも進めない(fallback / undefined_action のみ加算)
    # ので、バッファから外すこと自体に副作用は無い。
    if rt.get("spark_reported"):
        sim._spark_reported = True
        if getattr(sim, "logger", None) is not None:
            sim.logger.events = [e for e in sim.logger.events
                                 if e.kind != "spark_roster"]
    # ---- レーン D1(第109): 起動時 1 回ではない「1 日 1 件 / 走査済み」ガードの復元 ----
    # どれも第80 W2 と同型の未保存(= resume 直後に同じ判定をやり直す)。
    # 旧 checkpoint / OFF ランでは None・0 = 従来挙動(getattr 既定と同値)。
    _dsd = rt.get("dev_signal_day")             # signal_summary の 1 交差点 1 日 1 件
    if _dsd is not None:
        sim._dev_signal_day = {str(k): int(v) for k, v in _dsd.items()}
    if "infoenv_watermark" in rt:               # バイラル / 誤情報の走査済み投稿(絶対 id)
        sim._infoenv_watermark = int(rt["infoenv_watermark"])
    _vdn = rt.get("viral_done")
    if _vdn:
        sim._viral_done = {int(x) for x in _vdn}
    _dnl = rt.get("dinner_logged")              # 世帯の夕食共食 (household_id, day)
    if _dnl:
        sim._dinner_logged = {tuple(x) for x in _dnl}
    if hasattr(sim, "_journal_rewind"):         # 第71: LLM ジャーナルを確定点まで巻き戻す
        sim._journal_rewind(rt.get("llm_journal"))

    # scenario: __init__ で config から再構築済み。封鎖の進行を復元し city へ再適用。
    sc = blob["scenario"]
    sim.scenario.active = bool(sc["active"])
    sim.scenario.closed = {tuple(e) for e in sc["closed"]}
    for u, v in sim.scenario.closed:
        if sim.city.graph.has_edge(u, v):
            sim.city.graph.edges[u, v]["closed"] = True
    sim.router.invalidate()                 # 封鎖(または非封鎖)を経路計算へ反映

    # traffic: __init__ で city/router/hub 付きで再構築済み。可変ランタイムだけ戻す。
    tr = blob["traffic"]
    sim.traffic.cars = tr["cars"]
    sim.traffic.n_active = tr["n_active"]
    sim.traffic.last_n = tr["last_n"]
    sim.traffic.total_spawned = tr["total_spawned"]

    return int(blob["step"])


def latest(run_dir: str | Path) -> Path | None:
    """run_dir/checkpoint/ 内の最新(step 最大)の checkpoint パス。無ければ None。"""
    ckpt_dir = Path(run_dir) / "checkpoint"
    if not ckpt_dir.is_dir():
        return None
    cands = sorted(ckpt_dir.glob("ckpt-*.pkl.gz"))
    if not cands:
        return None

    def _step(p: Path) -> int:
        try:
            return int(p.name[len("ckpt-"):].split(".")[0])
        except ValueError:
            return -1

    return max(cands, key=_step)
