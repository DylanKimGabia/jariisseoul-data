"""모델 비교: 같은 데이터, 같은 기간 분할, 같은 지표로 엔진들의 정확도와 속도를 잰다.

    cd ml
    python -m jariml.benchmark                          # 등록된 엔진 전부
    python -m jariml.benchmark --engines gbm lgbm gru   # 일부만

재는 것
    정확도  평가 기간 빈자리 MAE(30분 / 1시간 / 2시간 뒤), RMSE, MSLE, 확률 Brier
    속도    서비스와 같은 경로(Predictor)로 요청 1번 처리 시간
            - 예측: 주차장 전부, 도착 시각 1개 (/predict)
            - 마지막 도착 시각: 주차장 전부, 5분 간격 24개 (/deadline)
            학습 시간, 모델 파일 크기
결과는 reports/benchmark_<시각>.csv 에 저장한다.
"""
import argparse
import io
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd

from .config import ML_ROOT, load_config
from .data import resolve_data_dir
from .engines import ENGINES, INTERFACE_VERSION
from .evaluate import by_lot, compare, eval_frame, probability_check, truth_frame
from .features import fit_profile
from .predictor import Predictor
from .preprocess import build_tables
from .probability import fit_residuals
from .split import time_split


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def _latency(fn, repeat):
    fn()                                                        # 첫 호출(준비 시간)은 빼고 잰다
    ts = []
    for _ in range(repeat):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return float(np.median(ts) * 1000)


def run_benchmark(cfg: dict, engines=None, repeat: int = 20, data=None) -> dict:
    hol, hs, k = cfg["preprocess"]["holidays"], cfg["target"]["horizons"], cfg["target"]["free_threshold"]
    if data is None:
        tables = build_tables(resolve_data_dir(cfg), cfg)
        s = cfg["split"]
        sp = time_split(tables.occ.index, s["test_days"], s["val_days"], s["gap_steps"], cfg["preprocess"]["step_min"])
    else:
        tables, sp = data
    occ, cap = tables.occ, tables.cap
    base_profile = fit_profile(occ[occ.index < sp.val_start], hol)
    truth_val = truth_frame(occ, cap, sp.val, hs, base_profile, hol)
    truth_test = truth_frame(occ, cap, sp.test, hs, base_profile, hol)

    # 속도 측정용 요청: 평가 기간 마지막 날 오전 7시 30분 기준 (출근 시간대)
    now = sp.test.normalize()[-1] + pd.Timedelta(hours=7, minutes=30)
    win = lambda w, col: w.loc[:now].tail(288).rename_axis("t").reset_index().melt("t", var_name="PKLT_CD", value_name=col)
    recent = win(occ, "occ")
    recent["cap"] = win(cap, "cap").cap.values
    recent["cnt"] = recent.occ * recent.cap

    rows, tables_out = [], {}
    for name in engines or list(ENGINES):
        log(f"[{name}] 학습 시작")
        eng = ENGINES[name](cfg)
        t0 = time.perf_counter()
        try:
            eng.fit(occ, cap, sp)
        except ImportError as e:
            log(f"[{name}] 건너뜀: {e}")
            continue
        fit_sec = time.perf_counter() - t0
        E_val = eval_frame(truth_val, eng.predict(occ, cap, sp.val, hs))
        E_test = eval_frame(truth_test, eng.predict(occ, cap, sp.test, hs))
        res = fit_residuals(E_val.h.to_numpy(), E_val.pred.to_numpy(), E_val.y.to_numpy(), cfg["probability"])
        table = compare(E_test)
        prob, _ = probability_check(E_test, res, k)
        tables_out[name] = table

        art = {"interface_version": INTERFACE_VERSION, "engine": eng, "profile": base_profile, "residuals": res,
               "lots": list(occ.columns), "lot_master": tables.lot_master, "fee_rule": tables.fee_rule,
               "confidence": pd.DataFrame({"label": "보통"}, index=occ.columns), "config": cfg,
               "trained_until": sp.val_start}
        p = Predictor(art)
        pred_ms = _latency(lambda: p.predict(recent, now + pd.Timedelta(minutes=60), now=now), repeat)
        dl_ms = _latency(lambda: p.deadline(recent, now=now), max(3, repeat // 4))
        buf = io.BytesIO(); joblib.dump(eng, buf, compress=3)

        m = lambda h, col="MAE(대)": float(table.loc[(h, "모델"), col])
        rows.append({"엔진": name,
                     "MAE 30분(대)": m(30), "MAE 1시간(대)": m(60), "MAE 2시간(대)": m(120),
                     "RMSE 1시간(대)": m(60, "RMSE(대)"), "MSLE 1시간": m(60, "MSLE"), "Brier": prob["Brier"],
                     "예측 응답(ms)": pred_ms, "마지막 도착 시각 응답(ms)": dl_ms,
                     "학습(초)": fit_sec, "모델 크기(MB)": buf.tell() / 1e6, "반복/에폭": eng.info.get("best_iter")})
        log(f"[{name}] 1시간 MAE {m(60):.3f}대 / 예측 {pred_ms:.0f}ms / 학습 {fit_sec:.0f}초")

    first = next(iter(tables_out.values()))
    for base in ("지금값 유지", "지금+평소변화"):
        rows.append({"엔진": f"(기준선) {base}", "MAE 30분(대)": float(first.loc[(30, base), "MAE(대)"]),
                     "MAE 1시간(대)": float(first.loc[(60, base), "MAE(대)"]), "MAE 2시간(대)": float(first.loc[(120, base), "MAE(대)"]),
                     "RMSE 1시간(대)": float(first.loc[(60, base), "RMSE(대)"]), "MSLE 1시간": float(first.loc[(60, base), "MSLE"])})
    result = pd.DataFrame(rows).set_index("엔진")
    return {"result": result, "tables": tables_out, "split": sp, "request_time": now}


def main():
    ap = argparse.ArgumentParser(description="엔진 정확도 / 속도 비교")
    ap.add_argument("--config", default=None)
    ap.add_argument("--engines", nargs="*", default=None, help=f"비교할 엔진 (기본: 전부 {list(ENGINES)})")
    ap.add_argument("--repeat", type=int, default=20, help="응답 시간 측정 반복 횟수")
    a = ap.parse_args()
    out = run_benchmark(load_config(a.config), a.engines, a.repeat)
    r = out["result"]
    pd.set_option("display.width", 250)
    print(r.round(3).to_string())
    d = ML_ROOT / "reports"
    d.mkdir(exist_ok=True)
    f = d / f"benchmark_{datetime.now(ZoneInfo('Asia/Seoul')):%Y%m%d-%H%M}.csv"
    r.to_csv(f, encoding="utf-8-sig")
    log(f"저장: {f}")


if __name__ == "__main__":
    main()
