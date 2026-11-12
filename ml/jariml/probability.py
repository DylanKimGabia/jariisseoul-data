"""'빈자리 K대 이상일 확률' 계산.

검증 기간에서 모델이 실제로 틀린 양(실제 - 예측 점유율)을 예측 거리 × 예측 점유율 구간별로 모아 둔다.
새 예측에 그 오차 분포를 적용해서 실제 빈자리가 K대 이상일 확률을 계산한다.
"""
import numpy as np
import pandas as pd


def _bin(values, edges):
    return np.searchsorted(np.asarray(edges), values, side="left").clip(0, len(edges) - 1)


def fit_residuals(h, pred, y, cfg_prob: dict, max_per_bin: int = 5000, seed: int = 0) -> dict:
    """검증 기간 오차 분포를 구간별로 저장. 구간마다 정렬된 오차 배열."""
    hb = _bin(h, cfg_prob["horizon_bins"])
    ob = _bin(pred, cfg_prob["occ_bins"])
    resid = np.asarray(y) - np.asarray(pred)
    rng = np.random.default_rng(seed)
    table = {}
    for i in range(len(cfg_prob["horizon_bins"])):
        for j in range(len(cfg_prob["occ_bins"])):
            r = resid[(hb == i) & (ob == j)]
            if len(r) < 200:                                 # 표본이 적으면 같은 예측 거리 전체를 쓴다
                r = resid[hb == i]
            if len(r) > max_per_bin:
                r = rng.choice(r, max_per_bin, replace=False)
            table[(i, j)] = np.sort(r)
    return {"edges": cfg_prob, "table": table}


def prob_free_at_least(res: dict, h, pred, cap, k: int) -> np.ndarray:
    """P(실제 빈자리 >= k) = P(실제 점유율 <= 1 - k/총면) = P(오차 <= 1 - k/총면 - 예측)."""
    h, pred, cap = map(np.atleast_1d, (h, pred, cap))
    hb = _bin(h, res["edges"]["horizon_bins"])
    ob = _bin(pred, res["edges"]["occ_bins"])
    limit = 1 - k / cap - pred
    out = np.empty(len(pred))
    for n in range(len(pred)):
        r = res["table"][(hb[n], ob[n])]
        out[n] = np.searchsorted(r, limit[n], side="right") / len(r)
    out[cap < k] = 0.0
    return out


def calibration_table(prob, actual, bins=10) -> pd.DataFrame:
    """확률이 맞는지 확인: 예측 확률 구간별 실제 비율. 두 값이 비슷해야 한다."""
    b = np.minimum((np.asarray(prob) * bins).astype(int), bins - 1)
    df = pd.DataFrame({"구간": b, "예측": prob, "실제": actual})
    t = df.groupby("구간").agg(예측확률평균=("예측", "mean"), 실제비율=("실제", "mean"), 건수=("예측", "size"))
    t.index = [f"{i * 100 // bins}~{(i + 1) * 100 // bins}%" for i in t.index]
    return t
