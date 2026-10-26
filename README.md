# 자리있서울 – 주차 데이터 수집

"도착 시점 빈자리 예측" 모델의 학습 데이터로 쓰기 위해 GitHub Actions로 자동 수집한다.

- 데이터 출처: 서울 열린데이터광장 (공공누리 1유형, 출처표시)
  - 서울시 시영주차장 실시간 주차대수 정보
  - 서울시 실시간 도시데이터 (121개 주요 장소)
  - 서울시 공영주차장 안내 정보 (좌표)

## 폴더 구성

| 경로 | 역할 |
|---|---|
| `scripts/` | 파이썬 코드 |
| `.github/workflows/` | 자동/수동 실행 설정 (GitHub 규칙상 위치 고정) |
| `data/` | 수집된 데이터 (자동 생성, 직접 수정하지 않음) |

### 계속 돌아가는 수집

| 코드 | 실행 설정 | 하는 일 |
|---|---|---|
| `scripts/run_collect_loop.sh` | `collect_loop.yml` | 작업 하나가 약 5시간 40분 동안 켜져서 주차장 5분, 도시데이터 10분마다 수집·커밋. 끝나면 자기 자신을 다시 실행 |
| `scripts/collect_parking_realtime.py` | `collect_parking_realtime.yml` (수동용) | 시영주차장 122곳 실시간 주차대수 1회 수집 |
| `scripts/collect_citydata_realtime.py` | `collect_citydata_realtime.yml` (수동용) | 도시데이터 121개 장소 인구·예측·날씨·도로·상권·주차 1회 수집 |

GitHub 예약 실행(cron)은 지연·누락이 잦아서 수집 간격을 직접 관리하는 루프 방식을 쓴다. `collect_loop.yml`의 2시간 간격 예약은 루프가 끊겼을 때 다시 깨우는 보조 장치다. 루프를 멈추려면 Actions 탭에서 실행 중인 루프를 Cancel 한다.

### 감시

| 코드 | 실행 설정 | 하는 일 |
|---|---|---|
| `scripts/check_collection_health.py` | `check_collection_health.yml` | 1시간마다 두 수집기 로그 점검. 마지막 기록이 오래됐거나(주차장 45분, 도시데이터 60분) 최근 오류가 많으면 실패 처리 |

수집이나 감시가 실패하면 GitHub가 계정 메일로 알림을 보낸다. 수집 실패 시에도 로그는 커밋된다.

### 한 번만 실행하는 작업 (Actions 탭에서 수동 실행)

| 코드 | 실행 설정 | 하는 일 |
|---|---|---|
| `scripts/explore_citydata_once.py` | `explore_citydata_once.yml` | 도시데이터 응답 구조 확인 |
| `scripts/build_parking_coords_once.py` | `build_parking_coords_once.yml` | 수집 대상 주차장 좌표 정리 |

## 데이터 구조

모든 시각은 한국시간(KST)이다. 날짜별 파일은 `YYYY-MM-DD.csv`.

| 경로 | 내용 |
|---|---|
| `data/live/` | 시영주차장 실시간: 총 주차면, 현재 주차 차량수, 서울시 갱신시각 |
| `data/static/` | 시영주차장 전체 컬럼 (요금, 운영시간 등), 하루 1회 |
| `data/log.csv` | 주차장 수집 실행 기록 |
| `data/citydata/population/` | 장소별 실시간 인구, 혼잡도 |
| `data/citydata/forecast/` | 장소별 향후 12시간 인구 예측 |
| `data/citydata/road/` | 장소 주변 평균 도로 소통 |
| `data/citydata/weather/` | 기온, 강수, 1시간 뒤 강수확률, 미세먼지 |
| `data/citydata/commercial/` | 상권 결제 활발도 |
| `data/citydata/parking/` | 도시데이터에만 있는 실시간 주차장 (구영·민영 포함) |
| `data/citydata/events/` | 장소별 문화행사, 하루 1회 |
| `data/citydata/log.csv` | 도시데이터 수집 실행 기록 |
| `data/reference/citydata_areas.csv` | 도시데이터 121개 장소 코드와 이름 |
| `data/reference/parking_lots.csv` | 수집 대상 주차장 좌표 |
| `data/explore/` | 도시데이터 탐색 결과 (1회) |

도시데이터는 기준시각(인구 `PPLTN_TIME`, 날씨 `WEATHER_TIME` 등)이 바뀐 경우에만 저장한다. 마지막 기준시각은 `data/citydata/_state.json`에 있다.

## 처음 설정

1. 서울 열린데이터광장에서 일반 인증키 발급
2. **Settings → Secrets and variables → Actions → New repository secret**
   - Name: `SEOUL_API_KEY`
   - Secret: 발급받은 인증키
3. **Actions** 탭에서 각 작업을 **Run workflow**로 한 번 실행해 확인

인증키는 절대 코드나 파일에 직접 쓰지 않는다. 공개 저장소라 누구나 볼 수 있다.

## 분석할 때 주의

- `PRK_STTS_YN`이 0(미연계)인 주차장은 차량수가 0으로 나온다. 제외한다.
- 관광버스 전용 주차장은 예측 대상에서 뺀다.
- `NOW_PRK_VHCL_UPDT_TM`(갱신시각)이 오래된 행은 실시간 값이 아니다. 걸러낸다.
- 같은 갱신시각이 여러 번 수집될 수 있다. 주차장코드 + 갱신시각 기준으로 중복 제거한다.
- 빈자리 = `TPKCT - NOW_PRK_VHCL_CNT`. 차량수가 총 주차면보다 큰 경우가 있으니 0으로 자른다.
- 도시데이터 일부 장소는 가끔 서버 오류(ERROR-500)로 빠진다. `data/citydata/log.csv`의 fail 칸에서 확인한다.
