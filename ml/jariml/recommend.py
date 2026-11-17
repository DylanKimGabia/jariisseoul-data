"""출발지, 목적지 → 도착 시점 기준 목적지 주변 공영주차장 추천 (상위 5곳).

    ① 장소 검색: 출발지, 목적지 텍스트 → 좌표                      TMAP POI 2건
    ② 자동차 경로: 출발지 → 목적지 시간 → 주차장별 도착 시각         TMAP 경로 1건
    ③ 후보: 목적지 반경 1km 공영주차장 (모자라면 2km), 출발지~목적지 사이 환승주차장(P&R)
    ④ 예측: 도착 시각의 예상 빈자리, '빈자리 3대 이상' 확률
    ⑤ 1차 정렬: 확률 등급 → 총 소요시간 → 도보 거리 (도보는 직선거리 추정)
    ⑥ 상위 5곳만 실제 도보 경로                                    TMAP 경로 최대 5건 (동시 호출)
    ⑦ 최종 정렬, 요금

TMAP 경로 호출은 검색 1번에 최대 6건 (Free 하루 1,000건 → 약 160번 검색).
"""
import re
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from .fee import parking_fee
from .tmap import CAR_KMH, DETOUR, WALK_DETOUR, WALK_KMH, Tmap, haversine_m

GRADES = [(0.8, "높음"), (0.5, "보통"), (0.0, "낮음")]     # [주1] 자리 있을 확률 등급 기준
PNR_DETOUR_MAX = 1.3          # [주2] 출발지→P&R→목적지 직선거리가 출발지→목적지의 1.3배 이하면 '가는 길'로 본다
WALK_MAX_M = 1500             # [주5] 걸어서 갈 만한 최대 거리(길 기준). 넘으면 마지막 이동을 대중교통으로 추정
TRANSIT_KMH, TRANSIT_WAIT_MIN = 18, 10   # [주3] 대중교통 API가 없어 P&R 이후 구간은 평균 속도 18km/h + 대기, 도보 10분으로 추정

KO = {"rank": "순위", "name": "주차장", "is_pnr": "환승주차장", "grade": "확률 등급", "prob": "자리 있을 확률(%)",
      "pred_free": "예상 빈자리(대)", "now_free": "지금 빈자리(대)", "cap": "총면", "arrival": "도착 예정",
      "drive_min": "운전(분)", "last_leg": "마지막 이동", "last_min": "마지막 이동(분)", "last_m": "마지막 이동 거리(m)",
      "total_min": "총 소요(분)", "fee": "예상 요금(원)", "confidence": "예측 신뢰도"}


def grade(p):
    return next(g for th, g in GRADES if p >= th)


def kst_now():
    return pd.Timestamp.now(tz="Asia/Seoul").tz_localize(None)


def parse_depart(x, now=None):
    """출발 시각 입력 해석. 시각만 쓰면 오늘 그 시각이고, 이미 지났으면 내일로 본다.
    예: "0830", "8:30", "08:30", "내일 0830", "11/21 08:30", "2026-11-21 08:30". 비우면 None(지금)."""
    if x is None or isinstance(x, pd.Timestamp) or not isinstance(x, str):
        return None if x is None else pd.Timestamp(x)
    s, now = x.strip(), now or kst_now()
    if not s or s == "지금":
        return None
    tomorrow = s.startswith("내일")
    s = s.replace("내일", "").strip()
    m = re.fullmatch(r"(\d{1,2}):?(\d{2})", s)
    if m:
        h, mi = int(m[1]), int(m[2])
        if h > 23 or mi > 59:
            raise ValueError(f"출발 시각 '{x}'를 이해하지 못했다. 예: 0830, 8:30, 내일 0830, 11/21 08:30")
        t = now.normalize() + pd.Timedelta(hours=h, minutes=mi)
        if tomorrow or t < now - pd.Timedelta(minutes=10):
            t += pd.Timedelta(days=1)
        return t
    m = re.fullmatch(r"(\d{1,2})[/.-](\d{1,2})\s+(\d{1,2}):?(\d{2})", s)
    if m:
        return pd.Timestamp(year=now.year, month=int(m[1]), day=int(m[2]), hour=int(m[3]), minute=int(m[4]))
    try:
        t = pd.Timestamp(s)
    except ValueError:
        t = None
    if t is None or not (now.year - 1 <= t.year <= now.year + 1):
        raise ValueError(f"출발 시각 '{x}'를 이해하지 못했다. 예: 0830, 8:30, 내일 0830, 11/21 08:30")
    return t


def recommend(origin, dest, predictor, recent, tmap: Tmap | None = None, depart=None, k=5, radius_m=1000,
              max_radius_m=2000, stay_min=120, with_pnr=True) -> dict:
    """origin, dest: 장소 이름(텍스트) 또는 (위도, 경도). depart: 출발 시각 (없으면 지금).
    반환: {"table": 추천 표(영문 열), "meta": 검색 정보}. 한글 표는 to_korean(table)."""
    t0 = time.perf_counter()
    tm = tmap or Tmap(live=False)
    calls0 = dict(tm.calls)
    o, d = tm.place(origin), tm.place(dest)
    depart = parse_depart(depart) or kst_now()
    data_now = pd.to_datetime(recent.t).max().floor("5min")

    main = tm.car(o, d, depart)                                         # ② 출발지 → 목적지

    lm = predictor.a["lot_master"].reindex(predictor.lots).dropna(subset=["LAT", "LNG"])
    od = float(haversine_m(o["lat"], o["lon"], d["lat"], d["lon"]))
    lm = lm.assign(to_dest=haversine_m(lm.LAT, lm.LNG, d["lat"], d["lon"]),
                   from_origin=haversine_m(o["lat"], o["lon"], lm.LAT, lm.LNG))

    # ③ 후보: 목적지 주변
    near = lm[lm.to_dest <= radius_m]
    if len(near) < k:
        near = lm[lm.to_dest <= max_radius_m]
    if len(near) < k:
        near = lm.nsmallest(k, "to_dest")                               # 주변에 없으면 가장 가까운 k곳
    cand = near.assign(kind="주변")
    # [주4] 주차장까지 운전 시간 = 목적지까지 TMAP 시간 + (주차장과 목적지의 출발지 기준 직선거리 차이) 보정
    cand["drive_s"] = main["time_s"] + (cand.from_origin - od) * DETOUR / (CAR_KMH / 3.6)
    walk_m = cand.to_dest * WALK_DETOUR
    far = walk_m > WALK_MAX_M
    cand["last_leg"] = np.where(far, "대중교통(추정)", "도보")
    cand["last_m"] = np.where(far, cand.to_dest * DETOUR, walk_m)
    cand["last_s"] = np.where(far, cand.last_m / (TRANSIT_KMH / 3.6) + TRANSIT_WAIT_MIN * 60, walk_m / (WALK_KMH / 3.6))
    note = "" if (~far).sum() >= min(k, len(cand)) else (
        f"목적지 도보권({WALK_MAX_M / 1000:.1f}km)에 예측 가능한 공영주차장이 {int((~far).sum())}곳뿐이라 "
        "먼 곳은 대중교통 이동으로 추정했다")

    if with_pnr:                                                        # 가는 길의 환승주차장
        p = lm[(lm.is_pnr == 1) & (lm.to_dest > radius_m) & (lm.from_origin + lm.to_dest <= PNR_DETOUR_MAX * max(od, 1))]
        if len(p):
            p = p.assign(kind="환승")
            p["drive_s"] = main["time_s"] * (p.from_origin / max(od, 1))
            p["last_leg"] = "대중교통(추정)"
            p["last_m"] = p.to_dest * DETOUR
            p["last_s"] = p.last_m / (TRANSIT_KMH / 3.6) + TRANSIT_WAIT_MIN * 60
            cand = pd.concat([cand, p])
    cand = cand[~cand.index.duplicated()].copy()
    cand["drive_s"] = cand.drive_s.clip(lower=60)
    cand["arrival"] = depart + pd.to_timedelta(cand.drive_s, unit="s")

    # ④ 예측: 도착 시각(5분 단위)이 같은 주차장끼리 한 번에
    k_free = predictor.k
    preds = []
    for arr, g in cand.groupby(cand.arrival.dt.round("5min")):
        preds.append(predictor.predict(recent, arr, now=data_now, lots=list(g.index)))
    pr = pd.concat(preds).set_index("PKLT_CD")
    cand = cand.join(pr[[f"prob_free_ge_{k_free}", "pred_free", "now_free", "cap", "confidence", "method"]].rename(
        columns={f"prob_free_ge_{k_free}": "prob"}), how="inner")
    cand["grade"] = cand.prob.map(grade)
    cand["grade_rank"] = cand.grade.map({g: i for i, (_, g) in enumerate(GRADES)})

    def ranked(c):
        c = c.assign(total_s=c.drive_s + c.last_s)
        return c.sort_values(["grade_rank", "total_s", "last_m"])

    # ⑤ 1차 정렬 후 상위 k곳 (환승주차장은 가장 좋은 1곳만 포함)
    r = ranked(cand)
    top = pd.concat([r[r.kind == "주변"].head(k - 1 if (r.kind == "환승").any() else k), r[r.kind == "환승"].head(1)])

    # ⑥ 상위 주변 주차장만 실제 도보 경로
    walk_ids = list(top[(top.kind == "주변") & (top.last_leg == "도보")].index)
    with ThreadPoolExecutor(max_workers=5) as ex:
        walks = list(ex.map(lambda c: tm.walk({"lat": top.at[c, "LAT"], "lon": top.at[c, "LNG"]}, d), walk_ids))
    for c, w in zip(walk_ids, walks):
        top.at[c, "last_m"], top.at[c, "last_s"] = w["dist_m"], w["time_s"]

    # ⑦ 최종 정렬, 요금
    top = ranked(top)
    rules, hol = predictor.a["fee_rule"], predictor.hol

    def fee(c):
        try:
            return parking_fee(rules.loc[c], top.at[c, "arrival"], stay_min, top.at[c, "kind"] == "환승", hol)
        except (KeyError, TypeError, ValueError):
            return None
    out = pd.DataFrame({
        "rank": range(1, len(top) + 1), "PKLT_CD": top.index, "name": top.PKLT_NM.values, "is_pnr": top.kind.eq("환승").values,
        "grade": top.grade.values, "prob": (top.prob * 100).round(1).values,
        "pred_free": top.pred_free.values, "now_free": top.now_free.values, "cap": top.cap.values,
        "arrival": top.arrival.dt.floor("min").values, "drive_min": (top.drive_s / 60).round(1).values,
        "last_leg": top.last_leg.values, "last_min": (top.last_s / 60).round(1).values, "last_m": top.last_m.round(0).values,
        "total_min": (top.total_s / 60).round(1).values, "fee": [fee(c) for c in top.index],
        "confidence": top.confidence.values, "lat": top.LAT.values, "lon": top.LNG.values,
    })
    meta = {"note": note, "origin": o, "dest": d, "depart": depart, "data_time": data_now,
            "data_lag_min": round((kst_now() - data_now).total_seconds() / 60, 1),
            "car": main, "candidates": int(len(cand)), "free_threshold": k_free, "stay_min": stay_min,
            "tmap_calls": {key: tm.calls[key] - calls0[key] for key in tm.calls}, "live": tm.live,
            "elapsed_s": round(time.perf_counter() - t0, 2)}
    return {"table": out, "meta": meta}


def to_korean(table: pd.DataFrame) -> pd.DataFrame:
    t = table.drop(columns=["PKLT_CD", "lat", "lon"]).rename(columns=KO).set_index("순위")
    t["환승주차장"] = np.where(t["환승주차장"], "예", "")
    t["도착 예정"] = pd.to_datetime(t["도착 예정"]).dt.strftime("%H:%M")
    return t
