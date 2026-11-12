"""학습 파이프라인 (명령 한 줄로 실행)

    cd ml
    python -m jariml.train                  # 학습 → 평가 → 저장 → 더 좋으면 운영 모델로 교체
    python -m jariml.train --refresh-data   # GitHub에서 최신 데이터를 받아서

순서: 원본 읽기 → 전처리 → 기간 분할 → 특징 → 학습(검증으로 반복 횟수 결정) → 확률용 오차 분포(검증)
      → 평가(평가 기간, 기준선과 비교) → 전체 기간으로 다시 학습 → 저장 → 운영 모델 교체 판단
"""
import argparse
import time

import numpy as np
import pandas as pd

from . import registry
from .config import load_config
from .data import resolve_data_dir
from .evaluate import by_hour, by_lot, compare, probability_check
from .features import FEATURES, build_rows, fit_profile
from .model import fit_with_val, make_model, predict_occ
from .preprocess import build_tables
from .probability import fit_residuals
from .split import time_split


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def run(cfg: dict, refresh_data=False, refit_all=True, save=True, promote=True) -> dict:
    hol, tg = cfg["preprocess"]["holidays"], cfg["target"]
    data_dir = resolve_data_dir(cfg, refresh_data)
    log(f"데이터: {data_dir}")

    tables = build_tables(data_dir, cfg)
    occ, cap = tables.occ, tables.cap
    log(f"전처리: 예측 대상 {occ.shape[1]}곳, {occ.index.min()} ~ {occ.index.max()}")
    log("제외: " + tables.lot_master.exclude_reason.replace("", "사용").value_counts().to_dict().__str__())

    s = cfg["split"]
    sp = time_split(occ.index, s["test_days"], s["val_days"], s["gap_steps"], cfg["preprocess"]["step_min"])
    log("기간 분할\n" + sp.describe().to_string())

    profile = fit_profile(occ[occ.index < sp.val_start], hol)          # 평소 곡선은 학습 기간으로만
    rows = lambda origins: build_rows(occ, cap, profile, hol, tg["horizons"], origins)
    tr, va, te = rows(sp.train), rows(sp.val), rows(sp.test)
    log(f"특징: 학습 {len(tr):,}행 / 검증 {len(va):,}행 / 평가 {len(te):,}행, 특징 {len(FEATURES)}개")

    model, best_iter, curve = fit_with_val(tr, va, cfg["model"])
    log(f"학습: 반복 {best_iter}회 선택 (검증 점유율 MAE {curve[best_iter - 1] * 100:.2f}%p)")

    pva = predict_occ(model, va)
    residuals = fit_residuals(va.h.to_numpy(), pva, va.y.to_numpy(), cfg["probability"])
    lot_mae = by_lot(va, pva)                                          # 주차장별 신뢰도 (검증 기간 1시간 뒤 오차)
    c = cfg["confidence"]
    confidence = pd.DataFrame({"val_mae_free_1h": lot_mae,
                               "label": np.select([lot_mae <= c["high_max_mae"], lot_mae <= c["mid_max_mae"]], ["높음", "보통"], "낮음")})

    pte = predict_occ(model, te)
    table = compare(te, pte)
    prob_summary, calib = probability_check(te, pte, residuals, tg["free_threshold"])
    hourly = by_hour(te, pte)
    one_h = table.xs(60, level=0)
    summary = {
        "test_mae_free_1h": float(one_h.loc["모델", "MAE(대)"]),
        "test_mae_free_1h_baseline_persist": float(one_h.loc["지금값 유지", "MAE(대)"]),
        "test_mae_free_1h_baseline_profile": float(one_h.loc["지금+평소변화", "MAE(대)"]),
        **{f"test_{k}": v for k, v in prob_summary.items()},
        "best_iter": best_iter,
    }
    log("평가(평가 기간, 빈자리 대수)\n" + table.round(3).to_string())
    log(f"확률 검증: {prob_summary}")

    if refit_all:                                                      # 운영용: 전체 기간으로 다시 학습
        profile = fit_profile(occ, hol)
        full = build_rows(occ, cap, profile, hol, tg["horizons"])
        model = make_model(cfg["model"], best_iter).fit(full[FEATURES], full.y - full.occ)
        log(f"전체 기간 재학습: {len(full):,}행")

    artifact = {
        "model": model, "profile": profile, "residuals": residuals, "features": FEATURES,
        "lots": list(occ.columns), "lot_master": tables.lot_master, "fee_rule": tables.fee_rule,
        "confidence": confidence, "config": cfg, "trained_until": occ.index.max(),
    }
    metrics = {
        "summary": summary,
        "split": sp.describe().astype(str).to_dict(),
        "data": {"start": str(occ.index.min()), "end": str(occ.index.max()), "lots": occ.shape[1]},
        "comparison": table.round(4).reset_index().to_dict("records"),
        "calibration": calib.round(4).reset_index().rename(columns={"index": "구간"}).to_dict("records"),
        "hourly_mae_1h": hourly.round(3).reset_index().to_dict("records"),
        "refit_all": refit_all,
    }
    out = {"tables": tables, "split": sp, "train": tr, "val": va, "test": te, "pred_test": pte,
           "comparison": table, "calibration": calib, "hourly": hourly, "summary": summary,
           "confidence": confidence, "artifact": artifact, "metrics": metrics, "curve": curve}
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
