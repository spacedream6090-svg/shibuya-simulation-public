"""環境パック(EnvPack)= 場所の知識の分離(基盤モデル抽出 D1-W2)のテスト。

検証:
- envpack.build_cfg の正準化: 既定(None)は generic かつ地名リテラルを含まない。dict は反映。
- 置換各所が cfg 値を反映する: place_name を変えると header/persona/観光文/駅名フィルタ/
  年中行事/気候/地下地名/番組名/場所ヒント が変わる。既定(渋谷)は従来と同一文字列。
- env/shibuya/env.yaml が YAML としてロード可能で必須キーを持つ。
検証は mock のみ(実LLM 禁止)。
"""
from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from society import diversity, envpack, media, schedule, weather
from society.cognition import deliberate
from society.config import load_config
from society.engine.simulation import Simulation

REPO = Path(__file__).resolve().parents[1]
_FORBIDDEN = ("渋谷", "スクランブル", "ハチ公", "道玄坂", "センター街")

# 現行(渋谷)ヘッダ先頭行 = ゴールデン・キャッシュ整合の基準(deliberate.py:14 由来)。
_SHIBUYA_HEAD = "あなたは渋谷の街で暮らす一人の人間です。状況に対して自然に振る舞ってください。"


def _sim(tmp_path, name="ep", n=2, **ov) -> Simulation:
    dot = ["run.seed=42", f"run.n_agents={n}", "run.n_steps=1", f"run.name={name}",
           "observer.snapshot_every=1"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


# --------------------------------------------------------------- build_cfg 単体
def test_build_cfg_defaults_are_generic_and_placeless():
    """build_cfg(None) は generic 既定で成立し、地名リテラルを一切含まない。"""
    cfg = envpack.build_cfg(None)
    # 構造キーが揃う
    for key in ("lexicon", "transit", "culture", "media", "origin", "climate"):
        assert key in cfg
    lx = cfg["lexicon"]
    assert lx["place_name"] == "この街" and lx["place_hints"] == []
    assert cfg["transit"]["station_filters"] == []
    # 既定に渋谷系の地名が紛れていない(基盤側に地名を残さない原則)
    blob = repr(cfg)
    for term in _FORBIDDEN:
        assert term not in blob, f"envpack 既定に地名 {term} が混入"


def test_build_cfg_canonicalizes_values():
    """dict を渡すと lexicon/transit/culture/media/climate が反映・正準化される。"""
    raw = {
        "lexicon": {"place_name": "架空市", "underground_name": "地下街X",
                    "place_hints": ["架空駅", "架空広場"]},
        "transit": {"station_filters": ["架空", "Fictis"]},
        "culture": {"events": [{"name": "祭", "month": "7", "day": "7",
                                "crowd": True}]},
        "media": {"tv": ["番組A"], "video": ["動画B"], "game": ["ゲームC"]},
        "climate": {"monthly": {"1": [1, 2, 0.3, 0.4]}, "neutral": [9, 8, 0.1, 0.0]},
        "origin": {"landmark": "中央広場"},
    }
    cfg = envpack.build_cfg(raw)
    assert cfg["lexicon"]["place_name"] == "架空市"
    assert cfg["lexicon"]["place_hints"] == ["架空駅", "架空広場"]
    assert cfg["transit"]["station_filters"] == ["架空", "Fictis"]
    ev = cfg["culture"]["events"][0]
    assert ev == {"name": "祭", "month": 7, "day": 7, "crowd": True}  # 型強制
    assert cfg["media"]["tv"] == ["番組A"]
    assert cfg["climate"]["monthly"][1] == [1.0, 2.0, 0.3, 0.4]       # 月キー=int
    assert cfg["origin"]["landmark"] == "中央広場"


# --------------------------------------------------------- 既定 config = 渋谷
def test_default_config_envpack_is_shibuya():
    """基底 config.yaml の envpack 既定は渋谷(=ゴールデンの値)。"""
    cfg = load_config([])
    ep = envpack.build_cfg(OmegaConf.to_container(cfg.envpack, resolve=True))
    assert ep["lexicon"]["place_name"] == "渋谷"
    assert ep["transit"]["station_filters"] == ["渋谷", "Shibuya"]
    names = {e["name"] for e in ep["culture"]["events"]}
    assert {"正月", "ハロウィン", "年末"} <= names
    assert ep["climate"]["monthly"][1] == [10.0, 2.0, 0.15, 0.08]


# --------------------------------------------------------------- header 反映
def test_header_reflects_place_name(tmp_path):
    """build_prompt 冒頭の街名が city_name を反映。既定(渋谷)は従来と一字一句同一。"""
    sim = _sim(tmp_path, "hdr")
    ag = sim.agents[0]
    assert sim.place_name == "渋谷"
    shibuya = deliberate.build_prompt(ag, place_name="路上", surprise="solo",
                                      nearby_names=[], city_name="渋谷")
    assert shibuya.splitlines()[0] == _SHIBUYA_HEAD
    other = deliberate.build_prompt(ag, place_name="路上", surprise="solo",
                                    nearby_names=[], city_name="架空市")
    assert other.splitlines()[0] == \
        "あなたは架空市の街で暮らす一人の人間です。状況に対して自然に振る舞ってください。"
    assert shibuya != other


# --------------------------------------------------------------- persona 反映
def test_persona_uses_place_name(tmp_path):
    """envpack.lexicon.place_name を変えると procedural persona 文がその街名になる。"""
    sim = _sim(tmp_path, "pers", **{"envpack.lexicon.place_name": "架空市"})
    assert sim.place_name == "架空市"
    assert any("架空市" in a.persona for a in sim.agents), "persona に街名が反映されない"
    assert all("渋谷" not in a.persona for a in sim.agents), "旧地名が残っている"


# --------------------------------------------------------------- map 地下地名
def test_map_underground_label_from_cfg(tmp_path):
    """CityMap の地下フォールバック地名が envpack から来る(既定=渋谷の値)。"""
    sim = _sim(tmp_path, "ug")
    assert sim.city.underground_label == "地下通路(渋谷ちかみち)"


# --------------------------------------------------------------- 駅名フィルタ
def test_transit_station_filters_from_cfg(tmp_path):
    """Transit が envpack の駅名フィルタを保持する(既定=渋谷/Shibuya)。"""
    sim = _sim(tmp_path, "tr")
    assert sim.transit.station_filters == ["渋谷", "Shibuya"]


# --------------------------------------------------------------- 年中行事
def test_annual_events_default_from_envpack(tmp_path):
    """annualcfg の既定行事は envpack.culture.events(渋谷=正月/ハロウィン/年末)由来。"""
    sim = _sim(tmp_path, "an")
    names = {e["name"] for e in sim.annualcfg["events"]}
    assert {"正月", "ハロウィン", "年末"} <= names
    hw = next(e for e in sim.annualcfg["events"] if e["name"] == "ハロウィン")
    assert hw["month"] == 10 and hw["day"] == 31 and hw["crowd"] is True


def test_annual_build_cfg_uses_default_events():
    """annual.build_cfg は default_events(envpack)を events 既定に採る。"""
    from society import annual
    custom = [{"name": "テスト祭", "month": 5, "day": 5, "crowd": True}]
    cfg = annual.build_cfg({"enabled": True}, default_events=custom)
    assert cfg["events"] == custom


# --------------------------------------------------------------- 気候
def test_weather_climate_from_cfg():
    """weather.build_cfg は envpack の monthly/neutral を採る。"""
    cfg = weather.build_cfg({"enabled": True}, monthly={1: [1, 2, 0.3, 0.4]},
                            neutral=[9, 8, 0.1, 0.0])
    assert cfg["monthly"][1] == (1, 2, 0.3, 0.4)
    assert cfg["neutral"] == (9, 8, 0.1, 0.0)


# --------------------------------------------------------------- 場所ヒント
def test_schedule_place_hints_from_cfg():
    """extract は place_hints(envpack)で地名固有の場所を拾う。無指定だと拾わない。"""
    txt = "明後日、架空スポットでご飯食べよう"
    with_hints = schedule.extract(txt, base_day=0, base_min=8 * 60,
                                  place_hints=("架空スポット",))
    assert with_hints and with_hints[0]["place"] == "架空スポット"
    without = schedule.extract(txt, base_day=0, base_min=8 * 60)
    assert without and without[0]["place"] == "", "地名固有ヒントが基盤に残っている"


# --------------------------------------------------------------- メディア番組名
def test_media_titles_from_cfg():
    """pick_title は渡された titles(envpack)から選ぶ。"""
    import numpy as np
    rng = np.random.default_rng(0)
    title = media.pick_title("tv", rng, titles={"tv": ["唯一の番組"]})
    assert title == "唯一の番組"


# --------------------------------------------------------------- 観光文
def test_diversity_context_line_place_name():
    """観光客の文脈行が place_name(envpack)を反映する。"""
    from types import SimpleNamespace
    ag = SimpleNamespace(tourist=True, language="")
    line = diversity.context_line(ag, "架空市")
    assert line is not None and "架空市を訪れている" in line


# --------------------------------------------------------------- env.yaml
def test_env_yaml_loads_and_has_required_keys():
    """env/shibuya/env.yaml が YAML としてロード可能で必須キーを持つ。"""
    path = REPO / "env" / "shibuya" / "env.yaml"
    assert path.exists(), "env/shibuya/env.yaml が無い"
    doc = OmegaConf.load(path)
    assert doc.env.name == "shibuya"
    assert doc.data.map and doc.data.transit                 # データ参照(実体は移さない)
    assert doc.institutions.pref == "tokyo"                  # ② pref セレクタ
    assert str(doc.institutions.ref).endswith("institutions_jp.yaml")
    assert doc.culture.lexicon.place_name == "渋谷"
    assert doc.origin.landmark == "スクランブル交差点"
    assert len(doc.culture.events) == 3
    assert doc.attribution and any("ODPT" in str(a) for a in doc.attribution)


def test_ref_institutions_jp_loads():
    """ref/institutions_jp.yaml(②共有参照)がロード可能で東京都の最賃を持つ。"""
    path = REPO / "ref" / "institutions_jp.yaml"
    assert path.exists()
    doc = OmegaConf.load(path)
    assert doc.prefectures.tokyo.min_wage_hourly == 1226
    assert doc.national.consumption.rate == 0.10
