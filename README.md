# 자리있서울 – 주차 데이터 수집

2026 서울AI디지털배움터 청년 AI 해커톤에서 "도착 시점 빈자리 예측" 모델의 학습 데이터로 쓰기 위해, 서울시 시영주차장 실시간 주차대수를 10분마다 수집한다.

- 데이터 출처: 서울 열린데이터광장 · 서울시 시영주차장 실시간 주차대수 정보 (공공누리 1유형, 출처표시)
- 실행: GitHub Actions (`.github/workflows/collect.yml`)

## 처음 설정

1. 서울 열린데이터광장에서 일반 인증키 발급
2. 이 저장소 **Settings → Secrets and variables → Actions → New repository secret**
   - Name: `SEOUL_API_KEY`
   - Secret: 발급받은 인증키
3. **Actions** 탭 → `collect-parking` → **Run workflow**로 한 번 수동 실행
4. `data/log.csv`에 `ok`가 찍히면 이후 10분마다 자동으로 쌓인다.

인증키는 절대 코드나 파일에 직접 쓰지 않는다. 공개 저장소라 누구나 볼 수 있다.

## 저장 구조

| 경로 | 내용 | 주기 |
|---|---|---|
| `data/live/YYYY-MM-DD.csv` | 수집시각, 주차장코드, 이름, 총 주차면, 현재 주차 차량수, 갱신시각, 연계여부 | 실행마다 |
| `data/static/YYYY-MM-DD.csv` | API 전체 컬럼 (요금, 운영시간 등) | 하루 1회 |
| `data/log.csv` | 실행 결과 (ok / error, 건수, 메시지) | 실행마다 |

시각은 모두 한국시간(KST)이다.

## 분석할 때 주의

- `PRK_STTS_YN`이 0(미연계)인 주차장은 차량수가 0으로 나온다. 제외한다.
- `NOW_PRK_VHCL_UPDT_TM`(갱신시각)이 오래된 행은 실시간 값이 아니다. 걸러낸다.
- 같은 갱신시각이 여러 번 수집될 수 있다. 주차장코드 + 갱신시각 기준으로 중복 제거한다.
- 빈자리 = `TPKCT - NOW_PRK_VHCL_CNT`. 차량수가 총 주차면보다 큰 경우가 있으니 0으로 자른다.
- 관광버스 전용 주차장은 예측 대상에서 뺀다.
