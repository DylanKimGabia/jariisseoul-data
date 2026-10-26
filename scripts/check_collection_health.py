"""수집 상태 감시 (1시간마다 자동 실행)

두 수집기의 로그를 보고 문제가 있으면 실패로 끝난다.
실패하면 GitHub가 계정 메일로 알림을 보낸다.

문제로 보는 경우
- 마지막 수집 기록이 너무 오래됨 (예약 실행이 멈춤)
- 최근 기록 중 오류가 많음 (API 오류, 인증키 문제 등)
"""
import csv
import os
import sys
from datetime import datetime, timedelta, timezone

KST = timezone(timedelta(hours=9))

CHECKS = [
    # 이름, 로그 경로, 이 시간(분) 넘게 기록 없으면 문제, 최근 N줄, 그중 오류 허용 개수
    ("주차장 실시간", os.path.join("data", "log.csv"), 45, 12, 2),
    ("도시데이터", os.path.join("data", "citydata", "log.csv"), 60, 6, 1),
]


def check(name, path, stale_min, recent_n, max_err, now):
    if not os.path.exists(path):
        return [f"[{name}] 로그 파일 없음: {path}"]
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return [f"[{name}] 로그가 비어 있음"]

    problems = []
    last = rows[-1]
    last_t = datetime.strptime(last["collected_at"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=KST)
    gap = (now - last_t).total_seconds() / 60
    if gap > stale_min:
        problems.append(f"[{name}] 마지막 수집이 {gap:.0f}분 전 ({last['collected_at']}). 기준 {stale_min}분")

    recent = rows[-recent_n:]
    errors = [r for r in recent if r.get("status") == "error"]
    if len(errors) > max_err:
        problems.append(f"[{name}] 최근 {len(recent)}회 중 오류 {len(errors)}회. 마지막 오류: "
                        f"{errors[-1].get('message', '')[:150]}")

    print(f"[{name}] 마지막 {last['collected_at']} ({gap:.0f}분 전), 상태 {last.get('status')}, "
          f"최근 {len(recent)}회 중 오류 {len(errors)}회")
    return problems


def main():
    now = datetime.now(KST)
    problems = []
    for c in CHECKS:
        problems += check(*c, now=now)

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    lines = ["## 수집 상태 점검", f"점검 시각: {now:%Y-%m-%d %H:%M} KST", ""]
    lines += [f"- ❌ {p}" for p in problems] or ["- ✅ 이상 없음"]
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    print("이상 없음")
    return 0


if __name__ == "__main__":
    sys.exit(main())
