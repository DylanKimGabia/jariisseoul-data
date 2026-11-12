"""빈자리 예측 모델: Gradient Boosting으로 지금 대비 점유율 변화량을 맞힌다."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .features import FEATURES


def make_model(p: dict, max_iter=None) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=max_iter or p["max_iter"], learning_rate=p["learning_rate"],
        max_leaf_nodes=p["max_leaf_nodes"], min_samples_leaf=p["min_samples_leaf"],
        l2_regularization=p["l2_regularization"], early_stopping=False, random_state=p["random_state"])


def fit_with_val(tr: pd.DataFrame, va: pd.DataFrame, p: dict):
    """학습 기간으로 학습하고, 검증 기간 오차가 가장 작은 반복 횟수를 고른다.

    돌려주는 값: (모델, 고른 반복 횟수, 반복 횟수별 검증 MAE)
    """
    m = make_model(p).fit(tr[FEATURES], tr.y - tr.occ)
    curve = []
    for pred in m.staged_predict(va[FEATURES]):
        curve.append(np.abs(np.clip(va.occ + pred, 0, 1) - va.y).mean())
    curve = np.array(curve)
    best = int(curve.argmin()) + 1
    if best < len(curve):                                 # 과적합 전 지점으로 다시 학습
        m = make_model(p, best).fit(tr[FEATURES], tr.y - tr.occ)
    return m, best, curve


def predict_occ(model, X: pd.DataFrame) -> np.ndarray:
    """예측 점유율 (0~1)."""
    return np.clip(X.occ.to_numpy() + model.predict(X[FEATURES]), 0, 1)
