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
from pathlib import PurePath

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


def plain(x):
    """설정 안의 Path를 문자열로 바꾼다. [주] Path를 그대로 저장하면 저장한 Python 버전에서만 열린다
    (3.13에서 저장한 파일을 3.11 서버에서 열면 pathlib._local 오류)."""
    if isinstance(x, dict):
        return {k: plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(plain(v) for v in x)
    return str(x) if isinstance(x, PurePath) else x


def current_on_same_test(cfg, occ, cap, sp, hs, base_profile, hol):
    """운영 모델을 새 모델과 같은 평가 기간에서 다시 채점한다 (1시간 뒤 빈자리 MAE, 대).

    [주] 저장된 점수끼리 비교하면 평가 기간이 달라 공정하지 않다 (기간마다 난이도가 다르다).
    운영 모델이 이미 그 평가 기간을 학습에 쓴 경우(같은 날 재학습 등)나 주차장 구성이 달라 채점할 수 없으면 None.
    """
    try:
        cur, _ = registry.load(cfg["registry"]["dir"])
    except FileNotFoundError:
        return None
    except Exception as e:                                           # 다른 Python 버전에서 저장된 파일 등
        log(f"운영 모델을 불러올 수 없어 같은 기간 비교 불가: {type(e).__name__}: {e}")
        return None
    if pd.Timestamp(cur["trained_until"]) >= sp.test_start:
        log("운영 모델이 평가 기간을 이미 학습해서 같은 기간 비교 불가")
        return None
    try:
        E = eval_frame(truth_frame(occ, cap, sp.test, hs, base_profile, hol), cur["engine"].predict(occ, cap, sp.test, hs))
    except (ValueError, KeyError) as e:
        log(f"운영 모델 재채점 불가: {e}")
        return None
    g = E[E.h == 12]
    return float((np.abs(g.pred - g.y) * g.cap).mean())


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
    lot_mae = by_lot(E_val)                                          # 주차장별 신뢰도: 검증 기간 1시간 뒤 오차
    lot_pct = lot_mae / tables.lot_master.TPKCT.reindex(lot_mae.index) * 100
    c = cfg["confidence"]
    confidence = pd.DataFrame({"val_mae_free_1h": lot_mae, "val_err_pct_1h": lot_pct,
                               "label": np.select([lot_pct <= c["high_max_err_pct"], lot_pct <= c["mid_max_err_pct"]],
                                                  ["높음", "보통"], "낮음")})
    log("신뢰도: " + str(confidence.label.value_counts().to_dict()))

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

    engine.cfg = plain(engine.cfg)                                   # 엔진이 들고 있는 설정 사본도 Path 제거
    artifact = {
        "interface_version": INTERFACE_VERSION, "engine": engine,
        "profile": fit_profile(occ, hol),                            # 2시간 초과 예측(평소 패턴)용, 전체 기간
        "residuals": residuals, "lots": list(occ.columns),
        "lot_master": tables.lot_master, "fee_rule": tables.fee_rule, "confidence": confidence,
        "config": plain(cfg), "trained_until": occ.index.max(),
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
            cur = current_on_same_test(cfg, occ, cap, sp, hs, base_profile, hol)
            ok, msg = registry.promote_if_better(d, cfg["registry"]["dir"], cfg["registry"]["promote_metric"], cur)
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
