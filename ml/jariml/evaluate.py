"""평가: 빈자리 대수 기준 MSE / RMSE / MAE / MSLE, 기준선 비교, 확률 검증."""
import numpy as np
import pandas as pd

from .probability import calibration_table, prob_free_at_least


def free_metrics(y_occ, p_occ, cap) -> dict:
    """점유율을 빈자리 대수로 바꿔 계산. MSLE는 log(1 + 빈자리)로 계산해 빈자리가 적을 때 오차를 더 크게 본다."""
    ft, fp = cap * (1 - y_occ), cap * (1 - p_occ)
    e = fp - ft
    return {"MAE(대)": np.abs(e).mean(), "MSE": (e ** 2).mean(), "RMSE(대)": np.sqrt((e ** 2).mean()),
            "MSLE": ((np.log1p(np.clip(fp, 0, None)) - np.log1p(np.clip(ft, 0, None))) ** 2).mean(),
            "점유율 MAE(%p)": np.abs(p_occ - y_occ).mean() * 100}


def compare(X: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    """모델과 기준선 2개를 예측 거리별로 비교."""
    methods = {
        "지금값 유지": X.occ.to_numpy(),
        "지금+평소변화": np.clip(X.occ + X.p_delta, 0, 1).to_numpy(),
        "모델": pred,
    }
    rows = []
    for h, g in X.groupby("h"):
        m = g.index.to_numpy()
        for name, p in methods.items():
            rows.append({"예측 거리(분)": h * 5, "방법": name, **free_metrics(g.y.to_numpy(), p[m], g.cap.to_numpy())})
    return pd.DataFrame(rows).set_index(["예측 거리(분)", "방법"])


def by_hour(X: pd.DataFrame, pred: np.ndarray, h: int = 12) -> pd.DataFrame:
    """도착 시각대별 오차 (기본: 1시간 뒤 예측)."""
    g = X[X.h == h].copy()
    g["오차(대)"] = np.abs(pred[g.index] - g.y) * g.cap
    g["도착 시각"] = (g.ttod // 12)
    return g.groupby("도착 시각")["오차(대)"].agg(["mean", "size"]).rename(columns={"mean": "MAE(대)", "size": "건수"})


def by_lot(X: pd.DataFrame, pred: np.ndarray, h: int = 12) -> pd.Series:
    """주차장별 1시간 뒤 빈자리 오차(대)."""
    g = X[X.h == h]
    return (np.abs(pred[g.index] - g.y) * g.cap).groupby(g.PKLT_CD).mean()


def probability_check(X: pd.DataFrame, pred: np.ndarray, residuals: dict, k: int) -> tuple[dict, pd.DataFrame]:
    """확률 검증: Brier 점수(낮을수록 좋음)와 구간별 예측 확률 vs 실제 비율."""
    prob = prob_free_at_least(residuals, X.h.to_numpy(), pred, X.cap.to_numpy(), k)
    actual = ((1 - X.y) * X.cap >= k).astype(float).to_numpy()
    base = actual.mean()
    brier = float(((prob - actual) ** 2).mean())
    return ({"Brier": brier, "Brier(항상 평균 확률)": float(((base - actual) ** 2).mean()), "실제 자리 있음 비율": float(base)},
            calibration_table(prob, actual))
