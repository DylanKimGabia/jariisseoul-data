"""서울 실시간 도시데이터 탐색 (한 번만 수동 실행)

121개 장소 응답 구조를 훑어서 정식 수집기를 설계하기 위한 스크립트.
결과는 data/explore/ 에 저장된다.
- summary.csv            : 장소별 응답 여부, 크기, 섹션 목록, 예측/주차장 개수
- parking_citydata.csv   : 도시데이터에 들어 있는 주차장 목록 (좌표 포함 여부 확인용)
- sample_citydata.json   : 첫 번째 정상 응답 원본 1개
- sample_parkinfo.json   : 공영주차장 안내 정보(OA-13122) 원본 5건 (좌표 필드 확인용)
"""
import csv
import json
import os
import sys
import time

import requests

KEY = os.environ.get("SEOUL_API_KEY", "").strip()
CITY_URL = "http://openapi.seoul.go.kr:8088/{key}/json/citydata/1/5/{area}"
PARKINFO_URL = "http://openapi.seoul.go.kr:8088/{key}/json/GetParkInfo/1/5/"
OUT = os.path.join("data", "explore")
MAX_POI = 140  # POI001 ~ POI140 시도 (없는 코드는 건너뜀)


def first(x):
    if isinstance(x, list):
        return x[0] if x else {}
    return x if isinstance(x, dict) else {}


def as_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, dict):
        return [x]
    return []


def get_json(url):
    r = requests.get(url, timeout=40)
    return r, r.json()


def explore_citydata():
    summary, parking = [], {}
    saved_sample = False
    for i in range(1, MAX_POI + 1):
        cd = f"POI{i:03d}"
        row = {"area_cd": cd}
        try:
            r, body = get_json(CITY_URL.format(key=KEY, area=cd))
            row["bytes"] = len(r.content)
        except Exception as e:
            row.update(status="error", note=str(e)[:200])
            summary.append(row)
            continue

        city = body.get("CITYDATA") if isinstance(body, dict) else None
        if not isinstance(city, dict) or not city.get("AREA_NM"):
            row.update(status="no_data", note=json.dumps(body, ensure_ascii=False)[:200])
            summary.append(row)
            time.sleep(0.2)
            continue

        if not saved_sample:
            with open(os.path.join(OUT, "sample_citydata.json"), "w", encoding="utf-8") as f:
                json.dump(body, f, ensure_ascii=False, indent=1)
            saved_sample = True

        live = first(city.get("LIVE_PPLTN_STTS"))
        prk = as_list(city.get("PRK_STTS"))
        live_prk = [p for p in prk if str(p.get("CUR_PRK_YN", "")).upper() == "Y"]
        row.update(
            status="ok",
            area_nm=city.get("AREA_NM"),
            sections="|".join(sorted(city.keys())),
            congest=live.get("AREA_CONGEST_LVL"),
            ppltn_time=live.get("PPLTN_TIME"),
            n_fcst=len(as_list(live.get("FCST_PPLTN"))),
            n_prk=len(prk),
            n_prk_live=len(live_prk),
        )
        summary.append(row)

        for p in prk:
            key = p.get("PRK_CD") or p.get("PRK_NM")
            if key in parking:
                parking[key]["areas"] += "|" + cd
                continue
            rec = {"areas": cd}
            rec.update({k: v for k, v in p.items() if not isinstance(v, (list, dict))})
            parking[key] = rec
        time.sleep(0.2)
    return summary, list(parking.values())


def write_csv(path, rows):
    if not rows:
        return
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def main():
    if not KEY:
        print("SEOUL_API_KEY 없음", file=sys.stderr)
        return 1
    os.makedirs(OUT, exist_ok=True)

    try:
        _, body = get_json(PARKINFO_URL.format(key=KEY))
    except Exception as e:
        body = {"error": str(e)}
    with open(os.path.join(OUT, "sample_parkinfo.json"), "w", encoding="utf-8") as f:
        json.dump(body, f, ensure_ascii=False, indent=1)

    summary, parking = explore_citydata()
    write_csv(os.path.join(OUT, "summary.csv"), summary)
    write_csv(os.path.join(OUT, "parking_citydata.csv"), parking)

    ok = sum(1 for s in summary if s.get("status") == "ok")
    print(f"citydata ok={ok}/{len(summary)}, parking={len(parking)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
