"""서울시 시영주차장 실시간 주차대수 수집기 (자리있서울)

GitHub Actions에서 5분마다 실행된다.
- data/live/YYYY-MM-DD.csv   : 실행마다 실시간 값 누적
- data/static/YYYY-MM-DD.csv : 하루 1회 전체 컬럼(요금·운영시간 등) 스냅샷
- data/log.csv               : 실행 결과 기록
"""
import csv
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import requests

KST = timezone(timedelta(hours=9))
KEY = os.environ.get("SEOUL_API_KEY", "").strip()
SERVICE = "GetParkingInfo"
PAGE = 1000  # 1회 호출 최대 1,000건
BASE = "http://openapi.seoul.go.kr:8088/{key}/json/{svc}/{start}/{end}/"

LIVE_COLS = [
    "collected_at",          # 우리가 수집한 시각 (KST)
    "PKLT_CD",               # 주차장 코드
    "PKLT_NM",               # 주차장명
    "TPKCT",                 # 총 주차면
    "NOW_PRK_VHCL_CNT",      # 현재 주차 차량수
    "NOW_PRK_VHCL_UPDT_TM",  # 서울시 쪽 갱신 시각
    "PRK_STTS_YN",           # 실시간 연계 여부 (1=연계, 0=미연계)
]

DATA = "data"


def fetch_page(start, end, retries=3):
    url = BASE.format(key=KEY, svc=SERVICE, start=start, end=end)
    last = None
    for i in range(retries):
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            body = r.json()
            if SERVICE not in body:
                # 인증키 오류 등은 최상위 RESULT로 온다
                raise RuntimeError(f"API 오류: {body.get('RESULT', body)}")
            res = body[SERVICE].get("RESULT", {})
            code = res.get("CODE", "")
            if code not in ("INFO-000", ""):
                raise RuntimeError(f"API 결과 {code}: {res.get('MESSAGE')}")
            return body[SERVICE]
        except Exception as e:  # 네트워크 오류는 잠깐 쉬고 재시도
            last = e
            time.sleep(5 * (i + 1))
    raise last


def fetch_all():
    first = fetch_page(1, PAGE)
    total = int(first.get("list_total_count", 0))
    rows = list(first.get("row", []))
    start = PAGE + 1
    while start <= total:
        page = fetch_page(start, start + PAGE - 1)
        rows.extend(page.get("row", []))
        start += PAGE
    return rows, total


def append_csv(path, cols, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)


def write_log(now, status, count, msg):
    append_csv(
        os.path.join(DATA, "log.csv"),
        ["collected_at", "status", "count", "message"],
        [{"collected_at": now, "status": status, "count": count, "message": msg[:300]}],
    )


def main():
    now_dt = datetime.now(KST)
    now = now_dt.strftime("%Y-%m-%d %H:%M:%S")
    day = now_dt.strftime("%Y-%m-%d")

    if not KEY:
        write_log(now, "error", 0, "SEOUL_API_KEY가 설정되지 않음")
        print("SEOUL_API_KEY 없음", file=sys.stderr)
        return 1

    try:
        rows, total = fetch_all()
    except Exception as e:
        write_log(now, "error", 0, str(e))
        print("수집 실패:", e, file=sys.stderr)
        return 1

    for r in rows:
        r["collected_at"] = now

    # 실시간 값: 실행마다 누적
    append_csv(os.path.join(DATA, "live", f"{day}.csv"), LIVE_COLS, rows)

    # 전체 컬럼: 하루 한 번만 저장
    static_path = os.path.join(DATA, "static", f"{day}.csv")
    if not os.path.exists(static_path) and rows:
        cols = ["collected_at"] + [k for k in rows[0].keys() if k != "collected_at"]
        append_csv(static_path, cols, rows)

    write_log(now, "ok", len(rows), f"total={total}")
    print(now, "ok", len(rows), "rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
