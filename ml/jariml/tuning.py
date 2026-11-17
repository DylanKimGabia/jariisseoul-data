"""lgbm 강화: ① 하이퍼파라미터 튜닝 ② 시계열 교차검증 ③ 예측 근거(SHAP) ④ 확률 신뢰도.

공정성 규칙: 튜닝은 검증 기간만 보고, 평가 기간(test)은 튜닝이 끝난 뒤 한 번만 본다.
"""
import copy
import time

import numpy as np
import pandas as pd

from .data import resolve_data_dir
from .engines import ENGINES
from .evaluate import eval_frame, probability_check, truth_frame
from .features import FEATURES, build_rows, fit_profile
from .preprocess import build_tables
from .probability import fit_residuals
from .split import time_split
from .train import log

H1 = 12                                   # 1시간 뒤 = 5분 칸 12개. 튜닝 목표와 교차검증 지표의 기준

FEATURE_KO = {
    "h": "예측 거리", "occ": "지금 점유율", "d1": "5분 변화", "d3": "15분 변화", "d6": "30분 변화", "d12": "60분 변화",
    "tod": "지금 시각", "ttod": "도착 시각", "dow": "요일", "off": "지금 휴일", "toff": "도착 휴일", "cap": "총면",
    "p_now": "평소(지금)", "p_fut": "평소(도착)", "p_delta": "평소 변화", "gap": "오늘-평소 차이",
}


def load(cfg):
    tables = build_tables(resolve_data_dir(cfg), cfg)
    s = cfg["split"]
    return tables, time_split(tables.occ.index, s["test_days"], s["val_days"], s["gap_steps"], cfg["preprocess"]["step_min"])


def mae_cars(occ, pred, y, cap):
    return float((np.abs(np.clip(occ + pred, 0, 1) - y) * cap).mean())


def with_params(cfg, params, engine="lgbm"):
    c = copy.deepcopy(cfg)
    c.setdefault("engine_params", {})[engine] = {**c["engine_params"].get(engine, {}), **params}
    return c


# ① 하이퍼파라미터 튜닝 ------------------------------------------------------------------

def tune_lgbm(cfg, tables, sp, n_trials=60, timeout_min=30, sample_rows=400_000):
    """Optuna(TPE)로 검증 기간 1시간 뒤 MAE(대)를 최소화하는 lgbm 설정을 찾는다.

    [주1] 탐색 중에는 학습 행을 40만 개만 쓴다 (시도 1번 20~30초). 최종 평가는 전체 행으로 다시 학습한다.
    [주2] 손실함수도 후보로 고른다: l2(MSE), l1(MAE, 평가 지표와 같음), huber(작은 오차는 MSE, 큰 오차는 MAE).
          huber_delta는 MSE에서 MAE로 바뀌는 오차 크기(점유율 변화량, 0.01 = 1%p).
    [주2-1] 나무 개수는 탐색하지 않는다. 검증 기간에서 50번 연속 나아지지 않으면 멈추는 조기 종료로 정한다.
    """
    import lightgbm as lgb
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    hol, hs = cfg["preprocess"]["holidays"], cfg["target"]["horizons"]
    occ, cap = tables.occ, tables.cap
    prof = fit_profile(occ[occ.index < sp.val_start], hol)
    tr = build_rows(occ, cap, prof, hol, hs, sp.train, True)
    tr = tr.sample(min(sample_rows, len(tr)), random_state=0)
    va = build_rows(occ, cap, prof, hol, hs, sp.val, True)
    v1 = va[va.h == H1]

    def objective(trial):
        p = {"objective": trial.suggest_categorical("objective", ["l2", "l1", "huber"])}
        if p["objective"] == "huber":
            p["huber_delta"] = trial.suggest_float("huber_delta", 0.005, 0.1, log=True)
        p |= {
            "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 500, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        }
        kw = {k: v for k, v in p.items() if k != "huber_delta"}
        if "huber_delta" in p:
            kw["alpha"] = p["huber_delta"]                    # LightGBM에서 huber 경계값 이름은 alpha
        m = lgb.LGBMRegressor(n_estimators=3000, subsample_freq=1, verbose=-1, random_state=0, **kw)
        m.fit(tr[FEATURES], tr.y - tr.occ, eval_set=[(va[FEATURES], va.y - va.occ)], eval_metric="l1",
              callbacks=[lgb.early_stopping(50, verbose=False)])
        trial.set_user_attr("best_iter", int(m.best_iteration_))
        return mae_cars(v1.occ.to_numpy(), m.predict(v1[FEATURES]), v1.y.to_numpy(), v1.cap.to_numpy())

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
    base = cfg["engine_params"].get("lgbm", {})
    study.enqueue_trial({"objective": "l2", "num_leaves": base.get("num_leaves", 63), "learning_rate": base.get("learning_rate", 0.05),
                         "min_child_samples": 100, "subsample": 0.8, "colsample_bytree": 0.9, "reg_lambda": 1.0})  # [주3] 1번 시도 = 지금 설정
    t0 = time.perf_counter()
    study.optimize(objective, n_trials=n_trials, timeout=timeout_min * 60,
                   callbacks=[lambda s, t: log(f"시도 {t.number + 1}: {t.value:.3f}대 (최고 {s.best_value:.3f}대)")])
    log(f"튜닝 끝: {len(study.trials)}회, {time.perf_counter() - t0:.0f}초")
    return study


# ② 시계열 교차검증 ------------------------------------------------------------------------

def walk_forward(cfg, tables, variants: dict, n_folds=4, fold_days=2):
    """평가 기간을 fold_days일씩 과거로 옮기며 n_folds번 평가한다.

    variants: {이름: (엔진 이름, cfg)}. 각 fold는 그 fold 끝 시각까지의 데이터만 쓴다 (미래 정보 차단).
    [주4] 학습 기간이 5일보다 짧아지는 fold는 만들지 않는다.
    [주5] TMAP 도착 시간은 5분 단위 아무 값이나 나온다 (예: 35분). 그래서 1시간 뒤 MAE와 함께
          5분~2시간 24칸 전체 평균 MAE도 본다. 학습에 없던 예측 거리도 잘 맞히는지 확인하기 위해서다.
    반환: (fold별 표, 요약 표). 열은 (지표, 방법).
    """
    s, hol = cfg["split"], cfg["preprocess"]["holidays"]
    occ_all, cap_all = tables.occ, tables.cap
    last_day = occ_all.index.max().normalize()
    hs = list(range(1, cfg["target"]["max_horizon_steps"] + 1))
    err = lambda E, p: np.abs(p - E.y) * E.cap
    rows = []
    for k in range(n_folds):
        end = last_day - pd.Timedelta(days=k * fold_days) + pd.Timedelta(hours=23, minutes=55)
        occ, cap = occ_all.loc[:end], cap_all.loc[:end]
        try:
            sp = time_split(occ.index, fold_days, s["val_days"], s["gap_steps"], cfg["preprocess"]["step_min"])
        except ValueError:
            log(f"fold {k + 1}: 학습 기간이 짧아 중단")
            break
        prof = fit_profile(occ[occ.index < sp.val_start], hol)
        truth = truth_frame(occ, cap, sp.test, hs, prof, hol)
        preds = {"(기준선) 지금+평소변화": (truth.assign(pred=(truth.occ + truth.p_delta).clip(0, 1)))}
        for name, (engine, c) in variants.items():
            eng = ENGINES[engine](c)
            eng.fit(occ, cap, sp)
            preds[name] = eval_frame(truth, eng.predict(occ, cap, sp.test, hs))
        row = {("fold", ""): k + 1, ("평가 기간", ""): f"{sp.test_start:%m/%d}~{sp.test.max():%m/%d}"}
        for name, E in preds.items():
            e = err(E, E.pred)
            row[("1시간 MAE(대)", name)] = float(e[E.h == H1].mean())
            row[("5분~2시간 평균 MAE(대)", name)] = float(e.mean())
        log(f"fold {k + 1} {row[('평가 기간', '')]}: " + ", ".join(f"{n} {row[('1시간 MAE(대)', n)]:.3f}" for n in preds))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    df = df.set_index([("fold", ""), ("평가 기간", "")]).rename_axis(["fold", "평가 기간"])
    summary = pd.concat({"평균": df.mean(), "표준편차": df.std(), "최소": df.min(), "최대": df.max()}, axis=1)
    return df, summary


# ③ 예측 근거 (SHAP) ----------------------------------------------------------------------

def shap_table(engine, tables, sp, cfg, n=3000):
    """평가 기간 1시간 뒤 예측에서 변수별 기여도(SHAP)를 빈자리 대수로 계산한다.

    [주6] 계산량은 행 수 × 나무 수 × 잎 수에 비례한다. 튜닝 모델(잎 254개, 나무 수천 그루)은 무거워서 3천 행만 쓴다.
          평균 영향 순위는 3천 행이면 안정적이다.
    [주7] LightGBM 내장 TreeSHAP(pred_contrib)을 쓴다. 별도 shap 라이브러리가 필요 없다.
    기여도는 점유율 변화량 단위라서 총면을 곱해 '대'로 바꾼다 (부호를 뒤집어 + = 빈자리 늘어남).
    """
    hol = cfg["preprocess"]["holidays"]
    X = build_rows(tables.occ, tables.cap, engine.profile, hol, [H1], sp.test, True)
    X = X.sample(min(n, len(X)), random_state=0)
    contrib = engine.model.booster_.predict(X[FEATURES], pred_contrib=True)[:, :-1]
    cars = -contrib * X.cap.to_numpy()[:, None]
    S = pd.DataFrame(cars, columns=[FEATURE_KO[f] for f in FEATURES], index=X.index)
    imp = S.abs().mean().sort_values(ascending=False).rename("평균 영향(대)")
    return imp, S, X.rename(columns=FEATURE_KO)


# ④ 확률 신뢰도 ---------------------------------------------------------------------------

def calibration(engine, tables, sp, cfg):
    """검증 기간 오차로 확률을 만들고, 평가 기간에서 '확률 p%라고 한 경우 실제로 p% 자리가 있었나'를 본다."""
    hol, hs, k = cfg["preprocess"]["holidays"], cfg["target"]["horizons"], cfg["target"]["free_threshold"]
    occ, cap = tables.occ, tables.cap
    prof = fit_profile(occ[occ.index < sp.val_start], hol)
    E_val = eval_frame(truth_frame(occ, cap, sp.val, hs, prof, hol), engine.predict(occ, cap, sp.val, hs))
    E_test = eval_frame(truth_frame(occ, cap, sp.test, hs, prof, hol), engine.predict(occ, cap, sp.test, hs))
    res = fit_residuals(E_val.h.to_numpy(), E_val.pred.to_numpy(), E_val.y.to_numpy(), cfg["probability"])
    return probability_check(E_test, res, k)
