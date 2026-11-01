"""주차장(PKLT_CD) ↔ 도시데이터 장소(AREA_CD) 매핑표 만들기 (API 호출 없음, 1회 실행)

장소 위치: 도시데이터가 장소별로 알려주는 주변 주차장 좌표의 중앙값
(data/explore/parking_citydata.csv)
주차장 위치: data/reference/parking_lots.csv

결과: data/reference/lot_area_map.csv
  주차장마다 가까운 장소 3곳 (rank 1이 가장 가까움), 거리(km)
  rank 1 거리가 2km를 넘으면 within_2km = 0 (주변 도시데이터를 쓰기 애매한 주차장)
"""
import csv
import math
import os
from collections import defaultdict
from statistics import median

REF = os.path.join("data", "reference")
LOTS = os.path.join(REF, "parking_lots.csv")
AREA_PARKING = os.path.join("data", "explore", "parking_citydata.csv")
AREAS = os.path.join(REF, "citydata_areas.csv")
OUT = os.path.join(REF, "lot_area_map.csv")
TOP_N = 3


def km(lat1, lng1, lat2, lng2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def main():
    with open(AREAS, encoding="utf-8") as f:
        area_name = {r["AREA_CD"]: r["AREA_NM"] for r in csv.DictReader(f)}

    pts = defaultdict(list)
    with open(AREA_PARKING, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                lat, lng = float(r["LAT"]), float(r["LNG"])
            except (TypeError, ValueError):
                continue
            for a in r["areas"].split("|"):
                pts[a].append((lat, lng))
    centers = {a: (median(p[0] for p in v), median(p[1] for p in v)) for a, v in pts.items()}

    rows = []
    with open(LOTS, encoding="utf-8") as f:
        for lot in csv.DictReader(f):
            if not lot["LAT"]:
                continue
            lat, lng = float(lot["LAT"]), float(lot["LNG"])
            near = sorted((km(lat, lng, *c), a) for a, c in centers.items())[:TOP_N]
            for rank, (d, a) in enumerate(near, 1):
                rows.append({
                    "PKLT_CD": lot["PKLT_CD"], "PKLT_NM": lot["PKLT_NM"],
                    "rank": rank, "AREA_CD": a, "AREA_NM": area_name.get(a, ""),
                    "distance_km": f"{d:.2f}", "within_2km": int(near[0][0] <= 2.0),
                })

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    lots = {r["PKLT_CD"] for r in rows}
    close = {r["PKLT_CD"] for r in rows if r["within_2km"]}
    print(f"장소 {len(centers)}곳, 주차장 {len(lots)}곳 매핑, 그중 2km 이내 장소가 있는 주차장 {len(close)}곳")


if __name__ == "__main__":
    main()
