"""예측 (서비스에서 쓰는 부분)

    from jariml import Predictor
    p = Predictor.load()                                 # models/LATEST 모델
    recent = p.recent_from_data_dir()                    # 최근 수집 기록 (또는 백엔드 DB에서 같은 형태로 전달)
    p.predict(recent, arrival="2026-11-14 08:30")        # 주차장별 도착 시점 빈자리와 확률
    p.deadline(recent, min_prob=0.8)                     # 확률 80% 이상으로 자리 있는 마지막 도착 시각

recent 형태 (긴 표): t(5분 단위 시각), PKLT_CD, cnt(주차 대수), cap(총면).
필요한 길이는 엔진마다 다르다 (Predictor.history_minutes). 넉넉히 최근 하루치를 넘기면 된다.
엔진이 바뀌어도(ML → DL) 이 클래스의 입력과 출력은 같다.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from . import registry
from .config import load_config
from .data import load_live, resolve_data_dir
from .engines import INTERFACE_VERSION
from .preprocess import is_offday
from .probability import prob_free_at_least

STEP = pd.Timedelta(minutes=5)


class Predictor:
    def __init__(self, artifact: dict, metrics: dict | None = None):
        if artifact.get("interface_version") != INTERFACE_VERSION:
            raise ValueError(f"엔진 규약 버전이 다르다: 모델 {artifact.get('interface_version')} / 코드 {INTERFACE_VERSION}. 다시 학습한다.")
        self.a, self.metrics = artifact, metrics or {}
        self.engine = artifact["engine"]
        self.cfg = artifact["config"]
        self.lots = artifact["lots"]
        self.hol = self.cfg["preprocess"]["holidays"]
        self.k = self.cfg["target"]["free_threshold"]
        self.max_h = self.cfg["target"]["max_horizon_steps"]
        self.history_minutes = self.engine.history_steps * 5     # 예측에 필요한 최근 기록 길이

    @classmethod
    def load(cls, models_dir=None, version=None) -> "Predictor":
        models_dir = models_dir or load_config()["registry"]["dir"]
        return cls(*registry.load(models_dir, version))

    # ---------- 입력 ----------
    def recent_from_data_dir(self, data_dir=None, refresh=False) -> pd.DataFrame:
        """수집 저장소의 최근 live 파일 2개(어제, 오늘)에서 recent 표를 만든다."""
        data_dir = Path(data_dir) if data_dir else resolve_data_dir(self.cfg, refresh)
        files = sorted(Path(data_dir, "live").glob("*.csv"))[-2:]
        live = load_live(data_dir, files)
        live = live[live.PRK_STTS_YN == 1]
        return live.rename(columns={"NOW_PRK_VHCL_CNT": "cnt", "TPKCT": "cap"})[["t", "PKLT_CD", "cnt", "cap"]]

    def _grid(self, recent: pd.DataFrame, now=None):
        r = recent[recent.PKLT_CD.isin(self.lots)].copy()
        r["t"] = pd.to_datetime(r.t).dt.floor("5min")
        now = pd.Timestamp(now).floor("5min") if now is not None else r.t.max()
        idx = pd.date_range(now - STEP * (self.engine.history_steps - 1), now, freq="5min")
        wide = lambda c: r.pivot_table(index="t", columns="PKLT_CD", values=c, aggfunc="last").reindex(idx).reindex(columns=self.lots)
        cnt, cap = wide("cnt"), wide("cap")
        cap = cap.ffill().bfill().fillna(self.a["lot_master"].TPKCT.reindex(self.lots))
        cnt = cnt.interpolate(limit=3, limit_area="inside").ffill(limit=6).clip(upper=cap)   # 30분 넘게 끊긴 주차장은 비움
        return cnt / cap, cap, now

    # ---------- 예측 ----------
    def _predict_steps(self, occ, cap, now, steps):
        """steps(5분 칸)별 예측. 2시간을 넘으면 평소 곡선 값을 쓴다."""
        out = []
        model_steps = [h for h in steps if 1 <= h <= self.max_h]
        if model_steps:
            r = self.engine.predict(occ, cap, [now], model_steps)
            r["occ"] = r.PKLT_CD.map(occ.iloc[-1])
            r["cap"] = r.PKLT_CD.map(cap.iloc[-1])
            r["method"] = "모델"
            out.append(r[["PKLT_CD", "h", "occ", "cap", "pred", "method"]])
        far = [h for h in steps if h > self.max_h]
        if far:
            prof = self.a["profile"]
            col = pd.Index(prof["lots"]).get_indexer(self.lots)
            for h in far:
                t = now + STEP * h
                off, tod = int(is_offday([t], self.hol).iloc[0]), t.hour * 12 + t.minute // 5
                out.append(pd.DataFrame({"PKLT_CD": self.lots, "h": h, "occ": occ.iloc[-1].values, "cap": cap.iloc[-1].values,
                                         "pred": prof["values"][off, tod, col], "method": "평소 패턴(2시간 초과)"}))
        res = pd.concat(out, ignore_index=True).dropna(subset=["pred"])
        res["prob"] = prob_free_at_least(self.a["residuals"], np.minimum(res.h, self.max_h), res.pred, res.cap, self.k)
        return res

    def predict(self, recent: pd.DataFrame, arrival, now=None, lots=None) -> pd.DataFrame:
        """도착 시각의 주차장별 예상 빈자리와 '빈자리 K대 이상' 확률."""
        occ, cap, now = self._grid(recent, now)
        arrival = pd.Timestamp(arrival)
        h = max(1, int(round((arrival - now) / STEP)))
        r = self._predict_steps(occ, cap, now, [h])
        lm, conf = self.a["lot_master"], self.a["confidence"]
        r = r.assign(
            PKLT_NM=r.PKLT_CD.map(lm.PKLT_NM), is_pnr=r.PKLT_CD.map(lm.is_pnr),
            now_cnt=(r.occ * r.cap).round(), arrival=now + STEP * h, horizon_min=h * 5,
            pred_cnt=(r.pred * r.cap).round(), confidence=r.PKLT_CD.map(conf.label), base_time=now,
        )
        r["now_free"] = r.cap - r.now_cnt
        r["pred_free"] = r.cap - r.pred_cnt
        r = r.rename(columns={"prob": f"prob_free_ge_{self.k}"})
        cols = ["PKLT_CD", "PKLT_NM", "is_pnr", "cap", "base_time", "now_cnt", "now_free", "arrival", "horizon_min",
                "pred_cnt", "pred_free", f"prob_free_ge_{self.k}", "confidence", "method"]
        r = r[cols]
        return r[r.PKLT_CD.isin(lots)] if lots is not None else r

    def deadline(self, recent: pd.DataFrame, now=None, lots=None, min_prob=0.8) -> pd.DataFrame:
        """자리 확보 가능한 마지막 도착 시각: 확률이 min_prob 아래로 처음 떨어지기 직전 시각 (최대 2시간)."""
        occ, cap, now = self._grid(recent, now)
        r = self._predict_steps(occ, cap, now, list(range(1, self.max_h + 1)))
        rows = []
        for code, g in r.sort_values("h").groupby("PKLT_CD"):
            below = g[g.prob < min_prob]
            if below.empty:
                last, note = now + STEP * self.max_h, "2시간 내내 가능"
            elif int(below.h.iloc[0]) == 1:
                last, note = None, "지금 출발해도 불확실"
            else:
                last, note = now + STEP * (int(below.h.iloc[0]) - 1), ""
            rows.append({"PKLT_CD": code, "PKLT_NM": self.a["lot_master"].PKLT_NM.get(code), "base_time": now,
                         "deadline": last, "min_prob": min_prob, "note": note})
        d = pd.DataFrame(rows)
        return d[d.PKLT_CD.isin(lots)] if lots is not None else d
