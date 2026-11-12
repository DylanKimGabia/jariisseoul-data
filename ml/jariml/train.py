"""학습 파이프라인 (엔진 공통, 명령 한 줄로 실행)

    cd ml
    python -m jariml.train                  # 학습 → 평가 → 저장 → 더 좋으면 운영 모델로 교체
    python -m jariml.train --refresh-data   # GitHub에서 최신 데이터를 받아서

순서: 원본 읽기 → 전처리 → 기간 분할 → 엔진 학습(train, val) → 확률용 오차 분포(val)
      → 평가(test, 기준선과 비교) → 전체 기간 재학습 → 저장 → 운영 모델 교체 판단
엔진은 config.yaml의 model.engine으로 고른다. 나머지 과정은 엔진과 상관없이 같다.
"""
import argparse
import time

import numpy as np
import pandas as pd

from . import registry
from .config import load_config
from .data import resolve_data_dir
from .engines import INTERFACE_VERSION, make_engine
from .evaluate import by_hour, by_lot, compare, eval_frame, probability_check, truth_frame
from .features import fit_profile
from .preprocess import build_tables
from .probability import fit_residuals
from .split import time_split


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def run(cfg: dict, refresh_data=False, refit_all=True, save=True, promote=True) -> dict:
    hol, hs = cfg["preprocess"]["holidays"], cfg["target"]["horizons"]
    data_dir = resolve_data_dir(cfg, refresh_data)
    log(f"데이터: {data_dir}")

    tables = build_tables(data_dir, cfg)
    occ, cap = tables.occ, tables.cap
    log(f"전처리: 예측 대상 {occ.shape[1]}곳, {occ.index.min()} ~ {occ.index.max()}")
    log("제외: " + str(tables.lot_master.exclude_reason.replace("", "사용").value_counts().to_dict()))

    s = cfg["split"]
    sp = time_split(occ.index, s["test_days"], s["val_days"], s["gap_steps"], cfg["preprocess"]["step_min"])
    log("기간 분할\n" + sp.describe().to_string())

    engine = make_engine(cfg)
    engine.fit(occ, cap, sp)
    log(f"엔진 학습: {engine.name} {({k: v for k, v in engine.info.items() if not isinstance(v, list)})}")

    base_profile = fit_profile(occ[occ.index < sp.val_start], hol)     # 기준선(지금+평소변화) 계산용, 학습 기간만
    E_val = eval_frame(truth_frame(occ, cap, sp.val, hs, base_profile, hol), engine.predict(occ, cap, sp.val, hs))
    E_test = eval_frame(truth_frame(occ, cap, sp.test, hs, base_profile, hol), engine.predict(occ, cap, sp.test, hs))
    log(f"평가 표: 검증 {len(E_val):,}행 / 평가 {len(E_test):,}행")

    residuals = fit_residuals(E_val.h.to_numpy(), E_val.pred.to_numpy(), E_val.y.to_numpy(), cfg["probability"])
    lot_mae = by_lot(E_val)                                          # 주차장별 신뢰도 (검증 기간 1시간 뒤 오차)
    c = cfg["confidence"]
    confidence = pd.DataFrame({"val_mae_free_1h": lot_mae,
                               "label": np.select([lot_mae <= c["high_max_mae"], lot_mae <= c["mid_max_mae"]], ["높음", "보통"], "낮음")})

    table = compare(E_test)
    prob_summary, calib = probability_check(E_test, residuals, cfg["target"]["free_threshold"])
    hourly = by_hour(E_test)
    one_h = table.xs(60, level=0)
    summary = {
        "engine": engine.name,
        "test_mae_free_1h": float(one_h.loc["모델", "MAE(대)"]),
        "test_mae_free_1h_baseline_persist": float(one_h.loc["지금값 유지", "MAE(대)"]),
        "test_mae_free_1h_baseline_profile": float(one_h.loc["지금+평소변화", "MAE(대)"]),
        **{f"test_{k}": v for k, v in prob_summary.items()},
    }
    log("평가(평가 기간, 빈자리 대수)\n" + table.round(3).to_string())
    log(f"확률 검증: {prob_summary}")

    if refit_all:                                                    # 운영용: 전체 기간으로 다시 학습
        engine.fit_full(occ, cap)
        log("전체 기간 재학습 완료")

    artifact = {
        "interface_version": INTERFACE_VERSION, "engine": engine,
        "profile": fit_profile(occ, hol),                            # 2시간 초과 예측(평소 패턴)용, 전체 기간
        "residuals": residuals, "lots": list(occ.columns),
        "lot_master": tables.lot_master, "fee_rule": tables.fee_rule, "confidence": confidence,
        "config": cfg, "trained_until": occ.index.max(),
    }
    metrics = {
        "summary": summary,
        "engine": {"name": engine.name, "interface_version": INTERFACE_VERSION,
                   "info": {k: v for k, v in engine.info.items() if k != "val_curve_mae"}},
        "split": sp.describe().astype(str).to_dict(),
        "data": {"start": str(occ.index.min()), "end": str(occ.index.max()), "lots": occ.shape[1]},
        "comparison": table.round(4).reset_index().to_dict("records"),
        "calibration": calib.round(4).reset_index().rename(columns={"index": "구간"}).to_dict("records"),
        "hourly_mae_1h": hourly.round(3).reset_index().to_dict("records"),
        "refit_all": refit_all,
    }
    out = {"tables": tables, "split": sp, "engine": engine, "eval_val": E_val, "eval_test": E_test,
           "comparison": table, "calibration": calib, "hourly": hourly, "summary": summary,
           "confidence": confidence, "artifact": artifact, "metrics": metrics}
    if save:
        d = registry.save(artifact, metrics, cfg["registry"]["dir"])
        log(f"저장: {d}")
        out["model_dir"] = d
        if promote:
            ok, msg = registry.promote_if_better(d, cfg["registry"]["dir"], cfg["registry"]["promote_metric"])
            log(msg)
            out["promoted"] = ok
    return out


def main():
    ap = argparse.ArgumentParser(description="자리있서울 빈자리 예측 모델 학습")
    ap.add_argument("--config", default=None, help="설정 파일 (기본 ml/config.yaml)")
    ap.add_argument("--refresh-data", action="store_true", help="GitHub에서 최신 데이터를 다시 받는다")
    ap.add_argument("--no-refit", action="store_true", help="전체 기간 재학습을 건너뛴다")
    ap.add_argument("--no-promote", action="store_true", help="저장만 하고 운영 모델은 바꾸지 않는다")
    a = ap.parse_args()
    run(load_config(a.config), a.refresh_data, not a.no_refit, True, not a.no_promote)


if __name__ == "__main__":
    main()
