"""평가 (엔진 공통): 빈자리 대수 기준 MSE / RMSE / MAE / MSLE, 기준선 비교, 확률 검증.

엔진의 예측(t, PKLT_CD, h, pred)을 정답 표와 붙여서 평가한다. 어떤 엔진이든 같은 표, 같은 지표로 비교된다.
"""
import numpy as np
import pandas as pd

from .preprocess import is_offday
from .probability import calibration_table, prob_free_at_least


def truth_frame(occ: pd.DataFrame, cap: pd.DataFrame, origins, horizons, profile: dict, holidays) -> pd.DataFrame:
    """정답 표: 기준 시각 t × 예측 거리 h × 주차장. 기준선 계산용 평소 곡선 값도 붙인다."""
    lots, idx = list(occ.columns), occ.index
    O, C = occ.to_numpy(), cap.to_numpy()
    pos = idx.get_indexer(origins)
    col = pd.Index(profile["lots"]).get_indexer(lots)
    prof = profile["values"][:, :, col]
    off = is_offday(idx[pos], holidays).values.astype(int)
    tod = (idx[pos].hour * 12 + idx[pos].minute // 5).values
    frames = []
    for h in horizons:
        ok = pos + h < len(idx)
        p = pos[ok]
        tf = idx[p] + pd.Timedelta(minutes=5 * h)
        toff, ttod = is_offday(tf, holidays).values.astype(int), (tf.hour * 12 + tf.minute // 5).values
        k, n = len(p), len(lots)
        frames.append(pd.DataFrame({
            "t": np.repeat(idx[p].values, n), "PKLT_CD": np.tile(lots, k), "h": h,
            "occ": O[p].reshape(-1), "y": O[p + h].reshape(-1), "cap": C[p].reshape(-1), "ttod": np.repeat(ttod, n),
            "p_delta": (prof[toff, ttod] - prof[off[ok], tod[ok]]).reshape(-1),
        }))
    return pd.concat(frames, ignore_index=True).dropna(subset=["occ", "y"])


def eval_frame(truth: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    """정답 + 엔진 예측. 엔진이 예측하지 못한 행은 뺀다."""
    E = truth.merge(pred, on=["t", "PKLT_CD", "h"], how="inner")
    return E.dropna(subset=["pred"]).reset_index(drop=True)


def free_metrics(y_occ, p_occ, cap) -> dict:
    """점유율을 빈자리 대수로 바꿔 계산. MSLE는 log(1 + 빈자리)로 계산해 빈자리가 적을 때 오차를 더 크게 본다."""
    ft, fp = cap * (1 - y_occ), cap * (1 - p_occ)
    e = fp - ft
    return {"MAE(대)": np.abs(e).mean(), "MSE": (e ** 2).mean(), "RMSE(대)": np.sqrt((e ** 2).mean()),
            "MSLE": ((np.log1p(np.clip(fp, 0, None)) - np.log1p(np.clip(ft, 0, None))) ** 2).mean(),
            "점유율 MAE(%p)": np.abs(p_occ - y_occ).mean() * 100}


def compare(E: pd.DataFrame) -> pd.DataFrame:
    """엔진과 기준선 2개를 예측 거리별로 비교."""
    methods = {"지금값 유지": E.occ, "지금+평소변화": (E.occ + E.p_delta).clip(0, 1), "모델": E.pred}
    rows = []
    for h, g in E.groupby("h"):
        for name, p in methods.items():
            rows.append({"예측 거리(분)": h * 5, "방법": name,
                         **free_metrics(g.y.to_numpy(), p[g.index].to_numpy(), g.cap.to_numpy())})
    return pd.DataFrame(rows).set_index(["예측 거리(분)", "방법"])


def by_hour(E: pd.DataFrame, h: int = 12) -> pd.DataFrame:
    """도착 시각대별 빈자리 오차 (기본: 1시간 뒤 예측)."""
    g = E[E.h == h]
    err = (g.pred - g.y).abs() * g.cap
    return err.groupby(g.ttod // 12).agg(["mean", "size"]).rename(columns={"mean": "MAE(대)", "size": "건수"}).rename_axis("도착 시각")


def by_lot(E: pd.DataFrame, h: int = 12) -> pd.Series:
    """주차장별 1시간 뒤 빈자리 오차(대)."""
    g = E[E.h == h]
    return ((g.pred - g.y).abs() * g.cap).groupby(g.PKLT_CD).mean()


def probability_check(E: pd.DataFrame, residuals: dict, k: int) -> tuple[dict, pd.DataFrame]:
    """확률 검증: Brier 점수(낮을수록 좋음)와 구간별 예측 확률 vs 실제 비율."""
    prob = prob_free_at_least(residuals, E.h.to_numpy(), E.pred.to_numpy(), E.cap.to_numpy(), k)
    actual = ((1 - E.y) * E.cap >= k).astype(float).to_numpy()
    base = actual.mean()
    return ({"Brier": float(((prob - actual) ** 2).mean()), "Brier(항상 평균 확률)": float(((base - actual) ** 2).mean()),
             "실제 자리 있음 비율": float(base)}, calibration_table(prob, actual))
