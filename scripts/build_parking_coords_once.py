"""우리 수집 대상 주차장(시영 122곳)의 좌표 정리 (한 번만 수동 실행)

좌표 출처 우선순위
  1. 서울시 공영주차장 안내 정보(GetParkInfo)의 LAT/LOT (0이 아닌 값)
  2. 서울 실시간 도시데이터 주차장 목록(data/explore/parking_citydata.csv)의 LAT/LNG
결과: data/reference/parking_lots.csv
  좌표를 못 찾은 주차장은 LAT/LNG가 비어 있고 coord_source가 'missing'.
"""
import csv
import glob
import os
import sys

import requests

KEY = os.environ.get("SEOUL_API_KEY", "").strip()
URL = "http://openapi.seoul.go.kr:8088/{key}/json/GetParkInfo/{start}/{end}/"
OUT = os.path.join("data", "reference", "parking_lots.csv")


def valid(lat, lng):
    try:
        lat, lng = float(lat), float(lng)
    except (TypeError, ValueError):
        return None
    if 37.3 < lat < 37.8 and 126.7 < lng < 127.3:  # 서울 범위
        return f"{lat:.7f}", f"{lng:.7f}"
    return None


def fetch_parkinfo():
    coords, start = {}, 1
    while True:
        body = requests.get(URL.format(key=KEY, start=start, end=start + 999), timeout=60).json()
        data = body.get("GetParkInfo", {})
        for r in data.get("row", []):
            v = valid(r.get("LAT"), r.get("LOT"))
            if v and r.get("PKLT_CD") not in coords:
                coords[r["PKLT_CD"]] = v
        total = int(data.get("list_total_count", 0))
        start += 1000
        if start > total:
            return coords, total


def main():
    if not KEY:
        print("SEOUL_API_KEY 없음", file=sys.stderr)
        return 1

    static = sorted(glob.glob(os.path.join("data", "static", "*.csv")))[-1]
    with open(static, encoding="utf-8") as f:
        lots = list(csv.DictReader(f))

    parkinfo, total = fetch_parkinfo()

    citydata = {}
    path = os.path.join("data", "explore", "parking_citydata.csv")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                v = valid(r.get("LAT"), r.get("LNG"))
                if v:
                    citydata[r["PRK_CD"]] = v

    rows, stats = [], {"GetParkInfo": 0, "citydata": 0, "missing": 0}
    for lot in lots:
        cd = lot["PKLT_CD"]
        if cd in parkinfo:
            (lat, lng), src = parkinfo[cd], "GetParkInfo"
        elif cd in citydata:
            (lat, lng), src = citydata[cd], "citydata"
        else:
            lat, lng, src = "", "", "missing"
        stats[src] += 1
        rows.append({
            "PKLT_CD": cd, "PKLT_NM": lot["PKLT_NM"], "ADDR": lot["ADDR"],
            "PKLT_TYPE": lot.get("PKLT_TYPE"), "PRK_TYPE_NM": lot.get("PRK_TYPE_NM"),
            "TPKCT": lot["TPKCT"], "PRK_STTS_YN": lot["PRK_STTS_YN"],
            "LAT": lat, "LNG": lng, "coord_source": src,
        })

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"GetParkInfo total={total}, with coords={len(parkinfo)}; lots={len(rows)} {stats}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
