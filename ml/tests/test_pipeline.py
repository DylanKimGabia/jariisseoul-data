"""가짜 데이터로 전체 파이프라인 확인 (실제 데이터 없이 1분 안에 돈다).

    cd ml && python -m pytest -q
"""
import copy

import numpy as np
import pandas as pd
import pytest

from jariml import Predictor, load_config
from jariml.engines import ENGINES
from jariml.fee import parking_fee
from jariml.train import run

LOTS = [f"{i:06d}" for i in range(1, 7)]


@pytest.fixture(scope="module")
def fake_data(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    rng = np.random.default_rng(0)
    t = pd.date_range("2026-10-01", "2026-10-15 23:55", freq="5min")
    tod = (t.hour * 60 + t.minute) / 1440
    for d in ["live", "static", "reference"]:
        (root / d).mkdir()
    rows = []
    for k, code in enumerate(LOTS):
        cap = 50 + 30 * k
        occ = 0.3 + 0.5 * np.clip(np.sin(np.pi * (tod - 0.3) / 0.5), 0, None) * np.where(t.dayofweek >= 5, 0.5, 1)
        cnt = np.clip(np.round(cap * occ + rng.normal(0, 2, len(t))), 0, cap)
        rows.append(pd.DataFrame({"collected_at": t, "PKLT_CD": code, "PKLT_NM": f"테스트{k} 공영주차장(시)",
                                  "TPKCT": cap, "NOW_PRK_VHCL_CNT": cnt, "NOW_PRK_VHCL_UPDT_TM": t, "PRK_STTS_YN": 1}))
    live = pd.concat(rows)
    for day, g in live.groupby(live.collected_at.dt.date):
        g.to_csv(root / "live" / f"{day}.csv", index=False)
    static = pd.DataFrame({"PKLT_CD": LOTS, "PKLT_NM": [f"테스트{k} 공영주차장(시)" for k in range(6)], "ADDR": "중구 테스트동 1",
                           "PKLT_TYPE": "NW", "OPER_SE_NM": "시간제 주차장", "PAY_YN": "Y", "BSC_PRK_CRG": 300, "BSC_PRK_HR": 5,
                           "ADD_PRK_CRG": 300, "ADD_PRK_HR": 5, "DAY_MAX_CRG": 0, "PRD_AMT": 100000, "SAT_CHGD_FREE_SE": "Y",
                           "LHLDY_CHGD_FREE_SE": "N", **{c: "0000" for c in ["WD_OPER_BGNG_TM", "WE_OPER_BGNG_TM", "LHLDY_OPER_BGNG_TM"]},
                           **{c: "2400" for c in ["WD_OPER_END_TM", "WE_OPER_END_TM", "LHLDY_OPER_END_TM"]}})
    static.to_csv(root / "static" / "2026-10-15.csv", index=False)
    pd.DataFrame({"PKLT_CD": LOTS, "LAT": 37.5, "LNG": 127.0}).to_csv(root / "reference" / "parking_lots.csv", index=False)
    pd.DataFrame({"PKLT_CD": LOTS[:2], "discount_rate": 0.5}).to_csv(root / "reference" / "pnr_lots.csv", index=False)
    return root


@pytest.fixture(scope="module")
def trained(fake_data, tmp_path_factory):
    cfg = copy.deepcopy(load_config())
    cfg["data"]["data_dir"] = fake_data
    cfg["registry"]["dir"] = tmp_path_factory.mktemp("models")
    cfg["split"].update(test_days=3, val_days=3)
    cfg["model"]["max_iter"] = 50
    return cfg, run(cfg)


def test_split_has_no_overlap(trained):
    _, out = trained
    sp = out["split"]
    assert sp.train.max() < sp.val.min() and sp.val.max() < sp.test.min()


def test_model_beats_persistence(trained):
    _, out = trained
    s = out["summary"]
    assert s["test_mae_free_1h"] < s["test_mae_free_1h_baseline_persist"]


def test_predictor_outputs(trained, fake_data):
    cfg, out = trained
    p = Predictor.load(cfg["registry"]["dir"])
    recent = p.recent_from_data_dir(fake_data)
    now = recent.t.max()
    r = p.predict(recent, now + pd.Timedelta("1h"))
    assert len(r) == len(LOTS)
    assert r.prob_free_ge_3.between(0, 1).all()
    assert (r.pred_free >= 0).all() and (r.pred_free <= r.cap).all()
    assert (p.predict(recent, now + pd.Timedelta("5h")).method == "평소 패턴(2시간 초과)").all()
    assert len(p.deadline(recent)) == len(LOTS)


@pytest.mark.parametrize("name", list(ENGINES))
def test_engine_contract(name, trained):
    """엔진 규약: 등록된 모든 엔진이 같은 입력을 받아 같은 형태의 출력을 낸다."""
    cfg, out = trained
    occ, cap, sp = out["tables"].occ, out["tables"].cap, out["split"]
    eng = ENGINES[name](cfg)
    eng.fit(occ, cap, sp)
    pred = eng.predict(occ, cap, sp.test[:5], [1, 12])
    assert list(pred.columns) == ["t", "PKLT_CD", "h", "pred"]
    assert set(pred.h) == {1, 12} and set(pred.PKLT_CD) == set(occ.columns)
    assert pred.pred.between(0, 1).all()
    assert eng.history_steps >= 1


def test_fee():
    rule = pd.Series({"PAY_YN": "Y", "BSC_CRG": 300, "BSC_MIN": 5, "ADD_CRG": 300, "ADD_MIN": 5, "DAY_MAX": 0,
                      "SAT_PAY": "Y", "HOLIDAY_PAY": "N", "pnr_discount": 0.5})
    assert parking_fee(rule, "2026-10-14 09:00", 60) == 3600
    assert parking_fee(rule, "2026-10-14 09:00", 60, transfer=True) == 1800
    assert parking_fee(rule, "2026-10-18 09:00", 60) == 0          # 일요일 무료
