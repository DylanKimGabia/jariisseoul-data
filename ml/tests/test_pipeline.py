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
    cfg["engine_params"] = {"rf": {"n_estimators": 10}, "lgbm": {"n_estimators": 50}, "xgb": {"n_estimators": 50},
                            "mlp": {"max_iter": 3}, "gru": {"epochs": 2, "stride": 4}}
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
    pytest.importorskip({"xgb": "xgboost", "gru": "torch"}.get(name, "numpy"))   # 운영 서버에는 비교용 라이브러리가 없다
    cfg, out = trained
    occ, cap, sp = out["tables"].occ, out["tables"].cap, out["split"]
    eng = ENGINES[name](cfg)
    eng.fit(occ, cap, sp)
    pred = eng.predict(occ, cap, sp.test[:5], [1, 12])
    assert list(pred.columns) == ["t", "PKLT_CD", "h", "pred"]
    assert set(pred.h) == {1, 12} and set(pred.PKLT_CD) == set(occ.columns)
    assert pred.pred.between(0, 1).all()
    assert eng.history_steps >= 1


def test_benchmark_runs(trained):
    """비교 도구가 끝까지 돌고, 모든 엔진의 정확도와 응답 시간이 나온다."""
    pytest.importorskip("torch")
    from jariml.benchmark import run_benchmark
    cfg, out = trained
    r = run_benchmark(cfg, ["gbm", "ridge", "gru"], repeat=2, data=(out["tables"], out["split"]))["result"]
    assert {"gbm", "ridge", "gru"} <= set(r.index)
    assert (r.loc[["gbm", "ridge", "gru"], "예측 응답(ms)"] > 0).all()


def test_fee():
    rule = pd.Series({"PAY_YN": "Y", "BSC_CRG": 300, "BSC_MIN": 5, "ADD_CRG": 300, "ADD_MIN": 5, "DAY_MAX": 0,
                      "SAT_PAY": "Y", "HOLIDAY_PAY": "N", "pnr_discount": 0.5})
    assert parking_fee(rule, "2026-10-14 09:00", 60) == 3600
    assert parking_fee(rule, "2026-10-14 09:00", 60, transfer=True) == 1800
    assert parking_fee(rule, "2026-10-18 09:00", 60) == 0          # 일요일 무료


def test_tuning_tools(trained):
    """lgbm 튜닝, 시계열 교차검증, SHAP, 확률 신뢰도가 끝까지 돈다."""
    pytest.importorskip("optuna")
    from jariml import tuning
    from jariml.engines import ENGINES
    cfg, out = trained
    tables, sp = out["tables"], out["split"]
    study = tuning.tune_lgbm(cfg, tables, sp, n_trials=3, sample_rows=5000)
    assert len(study.trials) == 3 and study.best_value > 0
    tuned = tuning.with_params(cfg, study.best_params)
    df, summary = tuning.walk_forward(cfg, tables, {"기본": ("lgbm", cfg), "튜닝": ("lgbm", tuned)}, n_folds=3, fold_days=1)
    assert len(df) >= 2 and {"기본", "튜닝"} <= set(summary.index.get_level_values(1))
    eng = ENGINES["lgbm"](tuned)
    eng.fit(tables.occ, tables.cap, sp)
    imp, S, X = tuning.shap_table(eng, tables, sp, cfg, n=500)
    assert len(imp) == 16 and imp.iloc[0] >= imp.iloc[-1]
    prob, cal = tuning.calibration(eng, tables, sp, cfg)
    assert 0 <= prob["Brier"] <= 1 and len(cal) > 0


def test_promote_compares_same_period(trained, tmp_path):
    """교체 규칙: 같은 평가 기간 점수로 비교하고, 비교할 수 없으면 최신 모델로 바꾼다."""
    import json
    from jariml import registry
    for v, m in [("old", 3.0), ("new", 4.0)]:
        (tmp_path / v).mkdir()
        (tmp_path / v / "metrics.json").write_text(json.dumps({"summary": {"m": m}}))
    (tmp_path / "LATEST").write_text("old")
    assert registry.promote_if_better(tmp_path / "new", tmp_path, "m", 4.5)[0]        # 같은 기간에선 새 모델이 더 좋음
    (tmp_path / "LATEST").write_text("old")
    assert not registry.promote_if_better(tmp_path / "new", tmp_path, "m", 3.5)[0]
    assert registry.promote_if_better(tmp_path / "new", tmp_path, "m", None)[0]


def test_artifact_has_no_path_objects(trained):
    """저장 파일에 Path 객체가 없어야 다른 Python 버전 서버에서도 열린다."""
    from pathlib import PurePath
    cfg, out = trained
    def walk(x):
        if isinstance(x, dict):
            return any(walk(v) for v in x.values())
        return isinstance(x, PurePath)
    assert not walk(out["artifact"]["config"]) and not walk(out["artifact"]["engine"].cfg)


def test_recommend_mock(trained, fake_data):
    """추천: 가짜 응답 모드(TMAP 호출 없음)로 상위 5곳이 정해진 열로 나온다."""
    from jariml.recommend import recommend, to_korean
    from jariml.tmap import Tmap
    cfg, out = trained
    p = Predictor.load(cfg["registry"]["dir"])
    recent = p.recent_from_data_dir(fake_data)
    r = recommend((37.45, 127.0), (37.5, 127.0), p, recent, Tmap(live=False), depart=recent.t.max())
    t = r["table"]
    assert 1 <= len(t) <= 5 and list(t["rank"]) == list(range(1, len(t) + 1))
    assert t.prob.between(0, 100).all() and r["meta"]["tmap_calls"] == {"poi": 0, "route": 0, "transit": 0}
    assert "자리 있을 확률(%)" in to_korean(t).columns


def test_tmap_map_html(trained, fake_data):
    """TMAP 지도 HTML: 키, 경로선, 순위 표시가 들어간다."""
    from jariml.recommend import recommend
    from jariml.tmap import Tmap
    from jariml.tmapview import tmap_html
    cfg, out = trained
    p = Predictor.load(cfg["registry"]["dir"])
    recent = p.recent_from_data_dir(fake_data)
    r = recommend((37.45, 127.0), (37.5, 127.0), p, recent, Tmap(live=False), depart=recent.t.max())
    h = tmap_html(r, "TESTKEY")
    assert "appKey=TESTKEY" in h and "Tmapv2.Polyline" in h and h.count('"rank"') == len(r["table"])


def test_parse_voice():
    """음성 문장 해석: 출발지, 목적지, 출발 시각."""
    from jariml.voice import parse_voice
    now = pd.Timestamp("2026-11-18 09:20")
    q = parse_voice("수원 SK스카이뷰에서 교대역까지 내일 아침 8시 반에 출발", now)
    assert (q["origin"], q["dest"], q["depart"]) == ("수원 SK스카이뷰", "교대역", pd.Timestamp("2026-11-19 08:30"))
    q = parse_voice("강남역 근처 주차할 곳 알려줘", now)
    assert (q["origin"], q["dest"], q["depart"]) == (None, "강남역", None)
    q = parse_voice("여기서 을지로입구역까지 30분 뒤에", now)
    assert (q["origin"], q["dest"], q["depart"]) == (None, "을지로입구역", pd.Timestamp("2026-11-18 09:50"))
    assert parse_voice("광화문에서 세종로 가는데 오후 3시", now)["dest"] == "세종로"
    assert parse_voice("판교에서 코엑스로 오늘 저녁 7시", now)["depart"] == pd.Timestamp("2026-11-18 19:00")
