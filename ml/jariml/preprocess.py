"""전처리: 원본 → 5분 격자 → 제외 기준 → 표 3개 (lot_master / fee_rule / occ_5min)

01_preprocess_eda 노트북 셀 1~5와 같은 규칙이다.
"""
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .data import load_live, load_reference, load_static


@dataclass
class Tables:
    lot_master: pd.DataFrame   # 주차장 1곳 = 1행
    fee_rule: pd.DataFrame     # 주차장 1곳 = 1행
    occ: pd.DataFrame          # 시각 × 예측 대상 주차장, 점유율 0~1 (모델 입력)
    cap: pd.DataFrame          # 시각 × 예측 대상 주차장, 그 시점 총면

    def occ_5min(self, holidays) -> pd.DataFrame:
        """긴 형태 표 (시각, 주차장 1행). 저장과 분석용."""
        melt = lambda w, col: w.rename_axis("t").reset_index().melt("t", var_name="PKLT_CD", value_name=col)
        df = melt(self.occ, "occ")
        df["cap"] = melt(self.cap, "cap").cap.values
        df["cnt"] = df.occ * df.cap
        df["free"] = df.cap - df.cnt
        df["dow"] = df.t.dt.dayofweek
        df["is_offday"] = is_offday(df.t, holidays).astype(int)
        df["tod"] = df.t.dt.hour * 12 + df.t.dt.minute // 5
        return df[["t", "PKLT_CD", "cap", "cnt", "occ", "free", "dow", "is_offday", "tod"]]

    def save(self, out_dir: Path, holidays) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.lot_master.to_csv(out_dir / "lot_master.csv", encoding="utf-8-sig")
        self.fee_rule.to_csv(out_dir / "fee_rule.csv", encoding="utf-8-sig")
        self.occ_5min(holidays).to_csv(out_dir / "occ_5min.csv", index=False, encoding="utf-8-sig")


def is_offday(t, holidays) -> pd.Series:
    """주말 또는 공휴일."""
    t = pd.Series(pd.to_datetime(t))
    return (t.dt.dayofweek >= 5) | t.dt.date.isin(pd.to_datetime(holidays).date)


def to_grid(live: pd.DataFrame, interp_limit: int):
    """긴 형태 원본 → 시각 × 주차장 격자 (주차 대수, 총면, 연계 여부)."""
    idx = pd.date_range(live.t.min(), live.t.max(), freq="5min")
    wide = lambda col: live.pivot(index="t", columns="PKLT_CD", values=col).reindex(idx)
    cnt, cap, link = wide("NOW_PRK_VHCL_CNT"), wide("TPKCT").ffill().bfill(), wide("PRK_STTS_YN")
    cnt = cnt.interpolate(limit=interp_limit, limit_area="inside")   # 짧은 공백만 보간
    cnt = cnt.where(link.ffill() == 1).clip(upper=cap)               # 미연계 시점은 비움, 총면 초과는 자름
    return cnt, cap, link


def exclusion_reason(cnt, cap, link, static, p) -> pd.Series:
    """주차장별 제외 사유. 빈 문자열이면 예측 대상."""
    occ = cnt / cap
    daily_rng = occ.groupby(occ.index.date).agg(lambda s: s.max() - s.min())
    reason = pd.Series("", index=cnt.columns)
    reason[link.max() != 1] = "미연계"
    bus = (static.OPER_SE_NM + static.PKLT_NM).reindex(cnt.columns).str.contains("버스", na=False)
    reason[(reason == "") & bus] = "버스전용"
    reason[(reason == "") & (cap.max() < p["min_cap"])] = "20면 미만"
    reason[(reason == "") & (daily_rng.median() < p["stuck_daily_range"])] = "센서 멈춤"
    reason[(reason == "") & (cnt.nunique() < p["min_unique_values"])] = "센서 이상"
    return reason


def build_tables(data_dir: Path, cfg: dict) -> Tables:
    p = cfg["preprocess"]
    live, static = load_live(data_dir), load_static(data_dir)
    coord, pnr = load_reference(data_dir)
    cnt, cap, link = to_grid(live, p["interp_limit"])
    reason = exclusion_reason(cnt, cap, link, static, p)
    name = static.PKLT_NM.reindex(cnt.columns).fillna(live.groupby("PKLT_CD").PKLT_NM.first())
    pnr = pnr.reindex(cnt.columns)

    lot_master = pd.DataFrame({
        "PKLT_NM": name, "ADDR": static.ADDR, "LAT": coord.LAT, "LNG": coord.LNG,
        "PKLT_TYPE": static.PKLT_TYPE, "TPKCT": cap.iloc[-1],
        "WD_BGN": static.WD_OPER_BGNG_TM, "WD_END": static.WD_OPER_END_TM,
        "WE_BGN": static.WE_OPER_BGNG_TM, "WE_END": static.WE_OPER_END_TM,
        "HD_BGN": static.LHLDY_OPER_BGNG_TM, "HD_END": static.LHLDY_OPER_END_TM,
        "is_pnr": pnr.notna().astype(int),
        "use_for_model": (reason == "").astype(int), "exclude_reason": reason,
    }).reindex(cnt.columns).rename_axis("PKLT_CD")

    num = lambda c: pd.to_numeric(static[c])
    fee_rule = pd.DataFrame({
        "PAY_YN": static.PAY_YN, "BSC_CRG": num("BSC_PRK_CRG"), "BSC_MIN": num("BSC_PRK_HR"),
        "ADD_CRG": num("ADD_PRK_CRG"), "ADD_MIN": num("ADD_PRK_HR"), "DAY_MAX": num("DAY_MAX_CRG"),
        "MONTH_PASS": num("PRD_AMT"), "SAT_PAY": static.SAT_CHGD_FREE_SE, "HOLIDAY_PAY": static.LHLDY_CHGD_FREE_SE,
        "pnr_discount": pnr.fillna(0),
    }).reindex(cnt.columns).rename_axis("PKLT_CD")

    use = reason.index[reason == ""]
    return Tables(lot_master, fee_rule, (cnt[use] / cap[use]), cap[use])
