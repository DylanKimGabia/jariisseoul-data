"""TMAP API 호출: 장소 검색, 자동차 경로(지금 출발 / 나중에 출발), 보행자 경로.

키: 환경 변수 TMAP_APP_KEY, 없으면 지금 폴더부터 위로 올라가며 찾은 .env 파일의 TMAP_APP_KEY=...
live=False(기본)면 TMAP을 부르지 않고 직선거리로 추정한다 (테스트, 개발용. 하루 한도 절약).

[주1] Free 요금제 하루 한도: POI 검색 20,000건, 경로안내(자동차 + 보행자 + 타임머신 합산) 1,000건. 넘으면 그날은 차단.
      그래서 같은 요청은 10분 동안 저장해 두고 다시 부르지 않는다. 오늘 호출 수는 ~/.jariml_tmap_usage.json에 센다.
"""
import json
import math
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd

BASE = "https://apis.openapi.sk.com/tmap"
ROUTE_LIMIT, POI_LIMIT, TRANSIT_LIMIT = 1000, 20000, 10      # Free 하루 한도 (대중교통은 별도 상품)
USAGE_FILE = Path.home() / ".jariml_tmap_usage.json"

# [주2] 가짜 응답 모드의 추정값. 서울 도심 평균 기준으로 잡은 값이다.
DETOUR = 1.3                 # 실제 길 거리 / 직선거리
CAR_KMH = 22                 # 서울 도심 평균 차량 속도
WALK_KMH = 4.5
WALK_DETOUR = 1.25


def haversine_m(lat1, lon1, lat2, lon2):
    """두 좌표 사이 직선거리(m). 배열도 된다."""
    import numpy as np
    p = np.pi / 180
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 2 * 6_371_000 * np.arcsin(np.sqrt(a))


def find_key(name: str = "TMAP_APP_KEY") -> str | None:
    """환경 변수 name, 없으면 지금 폴더부터 위로 올라가며 찾은 .env 파일의 name=... 값."""
    if os.environ.get(name):
        return os.environ[name].strip()
    for d in [Path.cwd(), *Path.cwd().parents]:
        f = d / ".env"
        if f.exists():
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.startswith(name + "="):
                    return line.split("=", 1)[1].strip()
    return None


class Tmap:
    def __init__(self, live: bool = False, key: str | None = None, cache_sec: int = 600, timeout: float = 5):
        self.live, self.cache_sec, self.timeout = live, cache_sec, timeout
        self.key = key or find_key()
        if live and not self.key:
            raise RuntimeError("TMAP 키가 없다. .env 파일에 TMAP_APP_KEY=... 를 적거나 live=False로 쓴다.")
        self._cache = {}
        self.calls = {"poi": 0, "route": 0, "transit": 0}            # 이 객체가 실제로 부른 횟수

    # ---------- 공통 ----------
    def _count(self, kind):
        self.calls[kind] += 1
        try:
            u = json.loads(USAGE_FILE.read_text()) if USAGE_FILE.exists() else {}
        except (ValueError, OSError):
            u = {}
        today = {"poi": 0, "route": 0, "transit": 0, **u.get(str(date.today()), {})}
        today[kind] += 1
        try:
            USAGE_FILE.write_text(json.dumps({str(date.today()): today}))
        except OSError:
            pass
        if kind == "route" and today["route"] >= ROUTE_LIMIT * 0.9:
            print(f"[주의] 오늘 TMAP 경로 호출 {today['route']}건. 하루 한도 {ROUTE_LIMIT}건에 가깝다.")

    @staticmethod
    def usage_today() -> dict:
        try:
            u = json.loads(USAGE_FILE.read_text()).get(str(date.today()), {})
        except (ValueError, OSError):
            u = {}
        return {"poi": 0, "route": 0, "transit": 0, **u}

    def _cached(self, key, fn):
        hit = self._cache.get(key)
        if hit and time.time() - hit[0] < self.cache_sec:
            return hit[1]
        val = fn()
        self._cache[key] = (time.time(), val)
        return val

    def _request(self, path, kind, params=None, body=None, base=BASE, version=True):
        q = {"version": 1, **(params or {})} if version else (params or {})
        url = f"{base}/{path}" + ("?" + urllib.parse.urlencode(q) if q else "")
        req = urllib.request.Request(url, method="POST" if body else "GET",
                                     data=json.dumps(body).encode() if body else None,
                                     headers={"appKey": self.key, "Accept": "application/json", "Content-Type": "application/json"})
        self._count(kind)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                raw = r.read()
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"TMAP {path} 오류 {e.code}: {e.read().decode('utf-8', 'ignore')[:300]}") from None
        return json.loads(raw) if raw else {}

    # ---------- 장소 검색 ----------
    def poi(self, text: str) -> dict:
        """텍스트 → {name, lat, lon, addr}. 가짜 모드에서는 쓸 수 없다 (좌표를 직접 넣는다)."""
        if not self.live:
            raise RuntimeError("가짜 응답 모드에서는 장소 검색을 못 한다. (위도, 경도)를 직접 넣거나 live=True로 쓴다.")

        def call():
            d = self._request("pois", "poi", {"searchKeyword": text, "count": 1,
                                              "reqCoordType": "WGS84GEO", "resCoordType": "WGS84GEO"})
            pois = (d.get("searchPoiInfo") or {}).get("pois", {}).get("poi", [])
            if not pois:
                raise ValueError(f"'{text}' 검색 결과가 없다")
            p = pois[0]
            lat = float(p.get("frontLat") or p.get("noorLat"))    # [주3] frontLat: 건물 입구 좌표 (없으면 중심 좌표)
            lon = float(p.get("frontLon") or p.get("noorLon"))
            addr = " ".join(str(p.get(k, "")) for k in ["upperAddrName", "middleAddrName", "lowerAddrName"]).strip()
            return {"name": p.get("name", text), "lat": lat, "lon": lon, "addr": addr}
        return self._cached(("poi", text), call)

    def place(self, x) -> dict:
        """텍스트, (위도, 경도), 또는 "위도,경도" 문자열 → {name, lat, lon}."""
        if isinstance(x, str) and re.fullmatch(r"\s*-?\d+(\.\d+)?\s*,\s*-?\d+(\.\d+)?\s*", x):
            x = [float(v) for v in x.split(",")]
        if isinstance(x, (tuple, list)):
            return {"name": f"({x[0]:.5f}, {x[1]:.5f})", "lat": float(x[0]), "lon": float(x[1]), "addr": ""}
        return self.poi(x)

    # ---------- 경로 ----------
    def car(self, a: dict, b: dict, depart=None, arrive_by=None) -> dict:
        """자동차 경로 → {dist_m, time_s, source, path}. depart(미래 출발) 또는 arrive_by(도착 목표)를 주면
        그 시각 교통을 반영한 타임머신 경로."""
        if not self.live:
            d = float(haversine_m(a["lat"], a["lon"], b["lat"], b["lon"])) * DETOUR
            return {"dist_m": d, "time_s": d / (CAR_KMH / 3.6), "source": "추정", "path": [[a["lat"], a["lon"]], [b["lat"], b["lon"]]]}
        when, kind = (arrive_by, "arrival") if arrive_by is not None else (depart, "departure")
        future = when is not None and pd.Timestamp(when) > pd.Timestamp.now() + pd.Timedelta(minutes=10)
        key = ("car", round(a["lat"], 4), round(a["lon"], 4), round(b["lat"], 4), round(b["lon"], 4),
               (kind, pd.Timestamp(when).floor("15min")) if future else None)

        def call():
            if future:
                t = pd.Timestamp(when).strftime("%Y-%m-%dT%H:%M:%S+0900")
                body = {"routesInfo": {"departure": {"name": "start", "lon": str(a["lon"]), "lat": str(a["lat"])},
                                       "destination": {"name": "end", "lon": str(b["lon"]), "lat": str(b["lat"])},
                                       "predictionType": kind, "predictionTime": t, "searchOption": "00"}}
                d = self._request("routes/prediction", "route", body=body)
                src = "TMAP 타임머신"
            else:
                body = {"startX": str(a["lon"]), "startY": str(a["lat"]), "endX": str(b["lon"]), "endY": str(b["lat"]),
                        "reqCoordType": "WGS84GEO", "resCoordType": "WGS84GEO", "searchOption": "0", "trafficInfo": "N"}
                d = self._request("routes", "route", body=body)
                src = "TMAP 실시간"
            pr = _first_total(d)
            return {"dist_m": float(pr["totalDistance"]), "time_s": float(pr["totalTime"]), "source": src, "path": _lines(d)}
        return self._cached(key, call)

    def walk(self, a: dict, b: dict) -> dict:
        """보행자 경로 → {dist_m, time_s, source}."""
        if not self.live:
            d = float(haversine_m(a["lat"], a["lon"], b["lat"], b["lon"])) * WALK_DETOUR
            return {"dist_m": d, "time_s": d / (WALK_KMH / 3.6), "source": "추정", "path": [[a["lat"], a["lon"]], [b["lat"], b["lon"]]]}

        def call():
            body = {"startX": str(a["lon"]), "startY": str(a["lat"]), "endX": str(b["lon"]), "endY": str(b["lat"]),
                    "startName": "start", "endName": "end", "reqCoordType": "WGS84GEO", "resCoordType": "WGS84GEO"}
            d = self._request("routes/pedestrian", "route", body=body)
            pr = _first_total(d)
            return {"dist_m": float(pr["totalDistance"]), "time_s": float(pr["totalTime"]), "source": "TMAP", "path": _lines(d)}
        return self._cached(("walk", round(a["lat"], 4), round(a["lon"], 4), round(b["lat"], 4), round(b["lon"], 4)), call)

    def transit(self, a: dict, b: dict) -> dict | None:
        """대중교통 경로 → {dist_m, time_s, fare, transfers, source, path}. 가짜 모드이거나 오늘 한도(10건)를 다 쓰면 None.

        [주4] 대중교통은 출발 시각을 지정하지 않는다 (지금 기준 경로). 미래 출발이어도 소요시간은 비슷하다고 본다.
        """
        if not self.live or self.usage_today()["transit"] >= TRANSIT_LIMIT:
            return None

        def call():
            body = {"startX": str(a["lon"]), "startY": str(a["lat"]), "endX": str(b["lon"]), "endY": str(b["lat"]),
                    "count": 1, "lang": 0, "format": "json"}
            d = self._request("transit/routes", "transit", body=body, base="https://apis.openapi.sk.com", version=False)
            its = (((d.get("metaData") or {}).get("plan") or {}).get("itineraries") or [])
            if not its:
                return None
            it = its[0]
            path = []
            for leg in it.get("legs", []):
                ls = (leg.get("passShape") or {}).get("linestring")
                if not ls:
                    ls = " ".join(st.get("linestring", "") for st in leg.get("steps", []) or [])
                for xy in (ls or "").split():
                    x, y = xy.split(",")
                    path.append([float(y), float(x)])
            return {"dist_m": float(it.get("totalDistance", 0)), "time_s": float(it["totalTime"]),
                    "fare": (it.get("fare") or {}).get("regular", {}).get("totalFare"),
                    "transfers": it.get("transferCount"), "source": "TMAP", "path": path}
        try:
            return self._cached(("transit", round(a["lat"], 4), round(a["lon"], 4), round(b["lat"], 4), round(b["lon"], 4)), call)
        except (RuntimeError, KeyError, ValueError) as e:
            print("[대중교통 API 실패, 추정으로 대체]", str(e)[:200])
            return None


def _lines(d: dict) -> list:
    """경로 응답의 선(LineString) 좌표를 [위도, 경도] 목록으로 이어 붙인다 (지도 그리기용)."""
    out = []
    for f in d.get("features", []):
        g = f.get("geometry") or {}
        if g.get("type") == "LineString":
            out += [[float(y), float(x)] for x, y in g.get("coordinates", [])]
    return out


def _first_total(d: dict) -> dict:
    """경로 응답에서 전체 거리, 시간이 들어 있는 첫 항목의 properties."""
    for f in d.get("features", []):
        p = f.get("properties", {})
        if "totalTime" in p:
            return p
    raise RuntimeError(f"TMAP 경로 응답에 totalTime이 없다: {str(d)[:300]}")
