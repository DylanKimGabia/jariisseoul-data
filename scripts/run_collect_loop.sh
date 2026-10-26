#!/usr/bin/env bash
# 수집 루프 (GitHub Actions 작업 하나 안에서 약 5시간 40분 동안 반복)
# - 주차장: 5분마다  (scripts/collect_parking_realtime.py)
# - 도시데이터: 10분마다 (scripts/collect_citydata_realtime.py)
# 매 회차 결과를 바로 커밋한다. 끝나면 워크플로가 자기 자신을 다시 실행한다.
set -u

LOOP_MINUTES=${LOOP_MINUTES:-340}
end=$(( $(date +%s) + LOOP_MINUTES * 60 ))

git config user.name  "parking-bot"
git config user.email "parking-bot@users.noreply.github.com"

commit_push() {
  git add data
  git diff --cached --quiet && return 0
  git commit -q -m "$1"
  for i in 1 2 3 4 5; do
    git pull -q --rebase --autostash && git push -q && return 0
    sleep $((i * 5))
  done
  echo "push 실패: $1"
  return 1
}

round=0
while [ "$(date +%s)" -lt "$end" ]; do
  stamp=$(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M')

  python scripts/collect_parking_realtime.py || echo "주차장 수집 실패 ($stamp)"
  if [ $((round % 2)) -eq 0 ]; then
    python scripts/collect_citydata_realtime.py || echo "도시데이터 수집 실패 ($stamp)"
  fi
  commit_push "data: $stamp" || true

  round=$((round + 1))
  # 다음 5분 경계까지 대기
  now=$(date +%s)
  sleep $(( 300 - now % 300 ))
done
echo "루프 종료 (총 $round 회)"
