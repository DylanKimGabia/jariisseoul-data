"""서울 실시간 도시데이터 수집기 (자리있서울)

GitHub Actions에서 10분마다 실행된다. 121개 장소를 하나씩 호출해서
주차 예측에 쓸 항목만 뽑아 날짜별 CSV에 쌓는다.
같은 기준시각 값은 다시 저장하지 않는다(data/citydata/_state.json).

data/citydata/
  population/  실시간 인구 (혼잡도, 인구 범위)
  forecast/    향후 12시간 인구 예측
  road/        장소 주변 평균 도로 소통
  weather/     날씨, 미세먼지, 1시간 뒤 강수확률
  commercial/  상권 결제 활발도
  parking/     도시데이터에만 있는 실시간 주차장 (구영·민영 포함)
  events/      문화행사 목록 (하루 1회)
  log.csv      실행 결과
"""
import csv
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import requests

KST = timezone(timedelta(hours=9))
KEY = os.environ.get("SEOUL_API_KEY", "").strip()
URL = "http://openapi.seoul.go.kr:8088/{key}/json/citydata/1/5/{area}"
AREAS_FILE = os.path.join("data", "reference", "citydata_areas.csv")
OUT = os.path.join("data", "citydata")
STATE_FILE = os.path.join(OUT, "_state.json")
TIME_BUDGET_SEC = 200  # 전체 수집 제한 시간. 넘으면 남은 장소는 다음 회차로 넘긴다

COLS = {
    "population": ["collected_at", "AREA_CD", "AREA_NM", "PPLTN_TIME", "AREA_CONGEST_LVL",
                   "AREA_PPLTN_MIN", "AREA_PPLTN_MAX", "RESNT_PPLTN_RATE", "NON_RESNT_PPLTN_RATE"],
    "forecast": ["collected_at", "AREA_CD", "PPLTN_TIME", "FCST_TIME", "FCST_CONGEST_LVL",
                 "FCST_PPLTN_MIN", "FCST_PPLTN_MAX"],
    "road": ["collected_at", "AREA_CD", "ROAD_TRAFFIC_TIME", "ROAD_TRAFFIC_IDX", "ROAD_TRAFFIC_SPD"],
    "weather": ["collected_at", "AREA_CD", "WEATHER_TIME", "TEMP", "HUMIDITY", "PRECIPITATION",
                "PRECPT_TYPE", "NEXT1H_RAIN_CHANCE", "NEXT1H_SKY_STTS", "PM10", "PM25"],
    "commercial": ["collected_at", "AREA_CD", "CMRCL_TIME", "AREA_CMRCL_LVL", "AREA_SH_PAYMENT_CNT",
                   "AREA_SH_PAYMENT_AMT_MIN", "AREA_SH_PAYMENT_AMT_MAX"],
    "parking": ["collected_at", "AREA_CD", "PRK_CD", "PRK_NM", "PRK_TYPE", "CPCTY",
                "CUR_PRK_CNT", "CUR_PRK_TIME", "LAT", "LNG"],
}


def first(x):
    if isinstance(x, list):
        return x[0] if x else {}
    return x if isinstance(x, dict) else {}


def as_list(x):
    if isinstance(x, list):
        return x
    return [x] if isinstance(x, dict) else []


def load_areas():
    with open(AREAS_FILE, encoding="utf-8") as f:
        return [r["AREA_CD"] for r in csv.DictReader(f)]


def load_state():
    try:
        with open(STATE_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def fetch(area, retries=2):
    last = None
    for i in range(retries):
        try:
            r = requests.get(URL.format(key=KEY, area=area), timeout=(5, 15))
            body = r.json()
            city = body.get("CITYDATA")
            if isinstance(city, dict) and city.get("AREA_NM"):
                return city
            last = RuntimeError(json.dumps(body.get("RESULT", body), ensure_ascii=False)[:150])
        except Exception as e:
            last = e
        time.sleep(2)
    raise last


def append(kind, day, rows, cols):
    if not rows:
        return
    path = os.path.join(OUT, kind, f"{day}.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)


def changed(state, key, value):
    """같은 기준시각이면 False. 새 값이면 state를 갱신하고 True."""
    if not value or state.get(key) == value:
        return False
    state[key] = value
    return True


def main():
    now_dt = datetime.now(KST)
    now = now_dt.strftime("%Y-%m-%d %H:%M:%S")
    day = now_dt.strftime("%Y-%m-%d")
    os.makedirs(OUT, exist_ok=True)

    if not KEY:
        append("", "log", [{"collected_at": now, "status": "error", "message": "SEOUL_API_KEY 없음"}],
               ["collected_at", "status", "ok", "fail", "message"])
        return 1

    state = load_state()
    rows = {k: [] for k in COLS}
    events, seen_prk = [], set()
    ok, fails = 0, []

    started = time.time()
    areas = load_areas()
    # 시간 초과로 끊겼던 장소부터 시작해서, 늘 같은 장소만 빠지지 않게 한다
    start = state.get("_next_start", 0) % len(areas)
    state["_next_start"] = 0
    for idx, area in enumerate(areas[start:] + areas[:start]):
        if time.time() - started > TIME_BUDGET_SEC:
            fails.append(f"{area}부터 {len(areas) - idx}곳:시간초과로 건너뜀")
            state["_next_start"] = (start + idx) % len(areas)
            break
        try:
            c = fetch(area)
        except Exception as e:
            fails.append(f"{area}:{str(e)[:60]}")
            continue
        ok += 1
        base = {"collected_at": now, "AREA_CD": area}

        live = first(c.get("LIVE_PPLTN_STTS"))
        pt = live.get("PPLTN_TIME")
        if changed(state, f"pop:{area}", pt):
            rows["population"].append({**base, **live})
            for fc in as_list(live.get("FCST_PPLTN")):
                rows["forecast"].append({**base, "PPLTN_TIME": pt, **fc})

        road = first(first(c.get("ROAD_TRAFFIC_STTS")).get("AVG_ROAD_DATA"))
        if changed(state, f"road:{area}", road.get("ROAD_TRAFFIC_TIME")):
            rows["road"].append({**base, **road})

        wt = first(c.get("WEATHER_STTS"))
        if changed(state, f"weather:{area}", wt.get("WEATHER_TIME")):
            nxt = first(wt.get("FCST24HOURS"))
            rows["weather"].append({**base, **wt, "NEXT1H_RAIN_CHANCE": nxt.get("RAIN_CHANCE"),
                                    "NEXT1H_SKY_STTS": nxt.get("SKY_STTS")})

        cm = first(c.get("LIVE_CMRCL_STTS"))
        if changed(state, f"cmrcl:{area}", cm.get("CMRCL_TIME")):
            rows["commercial"].append({**base, **cm})

        for p in as_list(c.get("PRK_STTS")):
            cd = p.get("PRK_CD")
            if str(p.get("CUR_PRK_YN", "")).upper() != "Y" or cd in seen_prk:
                continue
            seen_prk.add(cd)
            if changed(state, f"prk:{cd}", p.get("CUR_PRK_TIME")):
                rows["parking"].append({**base, **p})

        if state.get(f"event_day:{area}") != day:
            state[f"event_day:{area}"] = day
            for ev in as_list(c.get("EVENT_STTS")):
                events.append({"collected_at": now, "AREA_CD": area,
                               **{k: v for k, v in ev.items() if not isinstance(v, (list, dict))}})
        time.sleep(0.2)

    for kind, cols in COLS.items():
        append(kind, day, rows[kind], cols)
    if events:
        cols = ["collected_at", "AREA_CD"] + sorted({k for e in events for k in e} - {"collected_at", "AREA_CD"})
        append("events", day, events, cols)

    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, sort_keys=True, indent=0)

    counts = " ".join(f"{k}={len(v)}" for k, v in rows.items())
    status = "ok" if ok and not fails else ("partial" if ok else "error")
    log = os.path.join(OUT, "log.csv")
    new = not os.path.exists(log)
    with open(log, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["collected_at", "status", "ok", "fail", "message"])
        w.writerow([now, status, ok, len(fails), (counts + " " + "|".join(fails))[:400]])
    print(now, status, ok, len(fails), counts)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
