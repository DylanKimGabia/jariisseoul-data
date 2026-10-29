#!/usr/bin/env bash
# 수집 루프 (GitHub Actions 작업 하나 안에서 약 5시간 40분 동안 반복)
# - 주차장: 5분마다  (scripts/collect_parking_realtime.py)
# - 도시데이터: 10분마다 (scripts/collect_citydata_realtime.py)
# 매 회차 시작 전에 저장소를 최신 상태로 맞추고, 수집 결과를 바로 커밋한다.
# 끝나면 워크플로가 자기 자신을 다시 실행한다.
set -u

LOOP_MINUTES=${LOOP_MINUTES:-340}
end=$(( $(date +%s) + LOOP_MINUTES * 60 ))

git config user.name  "parking-bot"
git config user.email "parking-bot@users.noreply.github.com"

# 저장소를 원격 main과 똑같이 맞춘다. 중간에 멈춘 rebase가 있어도 정리한다.
sync_to_remote() {
  git rebase --abort >/dev/null 2>&1 || true
  git merge --abort >/dev/null 2>&1 || true
  for i in 1 2 3; do
    git fetch -q origin main && git reset -q --hard origin/main && return 0
    sleep $((i * 5))
  done
  echo "원격 동기화 실패"
  return 1
}

commit_push() {
  git add data
  git diff --cached --quiet && return 0
  git commit -q -m "$1"
  for i in 1 2 3 4 5; do
    if git pull -q --rebase origin main && git push -q origin HEAD:main; then
      return 0
    fi
    git rebase --abort >/dev/null 2>&1 || true
    sleep $((i * 5))
  done
  echo "push 실패: $1 (이번 회차 데이터는 버리고 다음 회차에 최신 상태로 다시 시작)"
  return 1
}

fails=0
round=0
while [ "$(date +%s)" -lt "$end" ]; do
  stamp=$(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M')
  sync_to_remote || true

  timeout 90 python scripts/collect_parking_realtime.py || echo "주차장 수집 실패 ($stamp)"
  if [ $((round % 2)) -eq 0 ]; then
    timeout 240 python scripts/collect_citydata_realtime.py || echo "도시데이터 수집 실패 ($stamp)"
  fi

  if commit_push "data: $stamp"; then
    fails=0
  else
    fails=$((fails + 1))
    # 연속으로 못 올리면 이 실행을 끝내고 새 실행으로 갈아탄다 (Restart myself 단계)
    if [ "$fails" -ge 3 ]; then
      echo "3회 연속 업로드 실패. 루프를 종료하고 새로 시작한다."
      exit 1
    fi
  fi

  round=$((round + 1))
  now=$(date +%s)
  sleep $(( 300 - now % 300 ))
done
echo "루프 종료 (총 $round 회)"
