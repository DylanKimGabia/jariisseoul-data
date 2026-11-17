"""표 형태 입력(features.FEATURES 16개)을 쓰는 ML 엔진들.

모두 gbm 엔진과 같은 입력, 같은 정답(지금 대비 점유율 변화량)을 쓰고 모델 종류만 다르다.
그래서 정확도 차이는 모델 차이에서만 나온다.

    ridge   선형 회귀 (가장 단순, 기준용)
    rf      랜덤 포레스트
    lgbm    LightGBM
    xgb     XGBoost
    mlp     다층 퍼셉트론 (sklearn 신경망)
"""
import numpy as np
import pandas as pd

from ..features import FEATURES, build_rows, fit_profile
from .base import Engine


class TabularEngine(Engine):
    history_steps = 13               # 60분 전 변화량까지 쓰므로 지금 포함 13칸
    max_train_rows = None            # 학습이 느린 모델은 학습 행을 무작위로 줄인다

    def params(self) -> dict:
        return self.cfg.get("engine_params", {}).get(self.name, {})

    def make_estimator(self, n_iter=None):
        raise NotImplementedError

    def _rows(self, occ, cap, origins, horizons, with_target):
        return build_rows(occ, cap, self.profile, self.cfg["preprocess"]["holidays"], horizons, origins, with_target)

    def _sample(self, df):
        n = self.params().get("max_train_rows", self.max_train_rows)
        return df.sample(n, random_state=0) if n and len(df) > n else df

    def _fit_estimator(self, est, tr, va):
        """기본: 그냥 학습. 검증 기간으로 반복 횟수를 고르는 모델은 덮어쓴다."""
        est.fit(tr[FEATURES], tr.y - tr.occ)
        return est, None

    def fit(self, occ, cap, split):
        hol, hs = self.cfg["preprocess"]["holidays"], self.cfg["target"]["horizons"]
        self.profile = fit_profile(occ[occ.index < split.val_start], hol)
        tr = self._sample(self._rows(occ, cap, split.train, hs, True))
        va = self._rows(occ, cap, split.val, hs, True)
        self.model, best = self._fit_estimator(self.make_estimator(), tr, va)
        self.info = {"best_iter": best, "train_rows": len(tr), "val_rows": len(va), "features": FEATURES}
        return self.info

    def predict(self, occ, cap, origins, horizons):
        X = self._rows(occ, cap, origins, horizons, False)
        X["pred"] = np.clip(X.occ.to_numpy() + self.model.predict(X[FEATURES]), 0, 1)
        return X[["t", "PKLT_CD", "h", "pred"]]

    def fit_full(self, occ, cap):
        self.profile = fit_profile(occ, self.cfg["preprocess"]["holidays"])
        full = self._sample(self._rows(occ, cap, None, self.cfg["target"]["horizons"], True))
        self.model = self.make_estimator(self.info.get("best_iter")).fit(full[FEATURES], full.y - full.occ)
        self.info["full_rows"] = len(full)


def _imputed(model):
    """결측값을 0(변화 없음)으로 채운 뒤 표준화하는 파이프라인 (결측을 못 다루는 모델용)."""
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(SimpleImputer(strategy="constant", fill_value=0), StandardScaler(), model)


class RidgeEngine(TabularEngine):
    name = "ridge"

    def make_estimator(self, n_iter=None):
        from sklearn.linear_model import Ridge
        return _imputed(Ridge(alpha=self.params().get("alpha", 1.0)))


class RFEngine(TabularEngine):
    name = "rf"
    max_train_rows = 300_000         # 랜덤 포레스트는 행 수에 비례해 느려서 30만 행으로 학습

    def make_estimator(self, n_iter=None):
        from sklearn.ensemble import RandomForestRegressor
        p = self.params()
        return _imputed(RandomForestRegressor(
            n_estimators=p.get("n_estimators", 100), max_depth=p.get("max_depth", 18),
            min_samples_leaf=p.get("min_samples_leaf", 50), max_features=p.get("max_features", 0.5),
            n_jobs=-1, random_state=0))


class MLPEngine(TabularEngine):
    name = "mlp"
    max_train_rows = 400_000

    def make_estimator(self, n_iter=None):
        from sklearn.neural_network import MLPRegressor
        p = self.params()
        # sklearn MLP는 외부 검증 집합을 못 받아서, 학습 행 안에서 10%를 떼어 조기 종료에 쓴다
        return _imputed(MLPRegressor(hidden_layer_sizes=tuple(p.get("hidden", [128, 64])), alpha=p.get("alpha", 1e-4),
                                     batch_size=1024, learning_rate_init=1e-3, max_iter=p.get("max_iter", 40),
                                     early_stopping=True, n_iter_no_change=4, random_state=0))


class LGBMEngine(TabularEngine):
    name = "lgbm"

    def make_estimator(self, n_iter=None):
        import lightgbm as lgb
        p = self.params()
        return lgb.LGBMRegressor(
            n_estimators=n_iter or p.get("n_estimators", 2000), learning_rate=p.get("learning_rate", 0.05),
            num_leaves=p.get("num_leaves", 63), min_child_samples=p.get("min_child_samples", 100),
            subsample=p.get("subsample", 0.8), subsample_freq=1, colsample_bytree=p.get("colsample_bytree", 0.9),
            reg_lambda=p.get("reg_lambda", 1.0), verbose=-1, random_state=0)

    def _fit_estimator(self, est, tr, va):
        import lightgbm as lgb
        est.fit(tr[FEATURES], tr.y - tr.occ, eval_set=[(va[FEATURES], va.y - va.occ)], eval_metric="l1",
                callbacks=[lgb.early_stopping(50, verbose=False)])
        return est, int(est.best_iteration_)


class XGBEngine(TabularEngine):
    name = "xgb"

    def make_estimator(self, n_iter=None):
        import xgboost as xgb
        p = self.params()
        kw = {} if n_iter else {"early_stopping_rounds": 50}
        return xgb.XGBRegressor(
            n_estimators=n_iter or p.get("n_estimators", 2000), learning_rate=p.get("learning_rate", 0.05),
            max_depth=p.get("max_depth", 8), min_child_weight=p.get("min_child_weight", 50),
            subsample=0.8, colsample_bytree=0.9, tree_method="hist", eval_metric="mae", random_state=0, **kw)

    def _fit_estimator(self, est, tr, va):
        est.fit(tr[FEATURES], tr.y - tr.occ, eval_set=[(va[FEATURES], va.y - va.occ)], verbose=False)
        return est, int(est.best_iteration) + 1
