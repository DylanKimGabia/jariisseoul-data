"""엔진 1호: Gradient Boosting (scikit-learn HistGradientBoosting), 기본 설정값.

직접 만든 입력 16개(features.FEATURES)로 지금 대비 점유율 변화량을 맞힌다.
"""
import numpy as np
import pandas as pd

from ..features import FEATURES, build_rows, fit_profile
from ..model import fit_with_val, make_model
from .base import Engine


class GBMEngine(Engine):
    name = "gbm"
    history_steps = 13               # 60분 전 변화량까지 쓰므로 지금 포함 13칸

    def _rows(self, occ, cap, origins, horizons, with_target):
        return build_rows(occ, cap, self.profile, self.cfg["preprocess"]["holidays"], horizons, origins, with_target)

    def fit(self, occ, cap, split):
        hol, hs = self.cfg["preprocess"]["holidays"], self.cfg["target"]["horizons"]
        self.profile = fit_profile(occ[occ.index < split.val_start], hol)      # 평소 곡선은 학습 기간으로만
        tr = self._rows(occ, cap, split.train, hs, True)
        va = self._rows(occ, cap, split.val, hs, True)
        self.model, best, curve = fit_with_val(tr, va, self.cfg["model"])
        self.info = {"best_iter": best, "val_curve_mae": curve.tolist(), "train_rows": len(tr), "val_rows": len(va),
                     "features": FEATURES}
        return self.info

    def feature_frame(self, occ, cap, origins, horizons, with_target=True) -> pd.DataFrame:
        """모델 입력 표 (분석용: 입력 중요도 등). with_target=True면 정답 y도 붙인다."""
        return self._rows(occ, cap, origins, horizons, with_target)

    def predict(self, occ, cap, origins, horizons):
        X = self._rows(occ, cap, origins, horizons, False)
        X["pred"] = np.clip(X.occ.to_numpy() + self.model.predict(X[FEATURES]), 0, 1)
        return X[["t", "PKLT_CD", "h", "pred"]]

    def fit_full(self, occ, cap):
        self.profile = fit_profile(occ, self.cfg["preprocess"]["holidays"])
        full = self._rows(occ, cap, None, self.cfg["target"]["horizons"], True)
        self.model = make_model(self.cfg["model"], self.info["best_iter"]).fit(full[FEATURES], full.y - full.occ)
        self.info["full_rows"] = len(full)
