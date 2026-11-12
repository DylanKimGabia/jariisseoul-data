# ml: 공영주차장 빈자리 예측

수집 데이터(`../data`)로 **도착 시점 빈자리 수**와 **빈자리 3대 이상일 확률**을 예측한다.
원본 CSV → 전처리 → 기간 분할 → 학습 → 평가 → 모델 저장 → 운영 모델 교체까지 명령 한 줄로 돈다.

## 빠른 시작

```bash
cd ml
pip install -r requirements.txt
python -m jariml.train                      # 학습 (약 1~3분)
python -m pytest -q                         # 가짜 데이터로 파이프라인 점검
uvicorn api.app:app --port 8000             # 예측 API → http://localhost:8000/docs
```

`../data`가 없는 곳(서버, 컨테이너)에서는 GitHub 저장소 zip을 받아 `.cache/`에 푼다. 최신 데이터로 다시 받으려면 `--refresh-data`.

## 폴더 구성

| 경로 | 역할 |
|---|---|
| `config.yaml` | 모든 기준값. 코드에 숫자를 박지 않고 여기서만 바꾼다. 값마다 이유를 적었다 |
| `jariml/data.py` | 수집 데이터 읽기 (로컬 또는 GitHub) |
| `jariml/preprocess.py` | 전처리: 5분 격자, 결측 보간, 제외 기준, 표 3개 (`lot_master`, `fee_rule`, 점유율) |
| `jariml/split.py` | 시간 순서 학습 / 검증 / 평가 분할 |
| `jariml/features.py` | 모델 입력 16개. 학습과 예측이 같은 함수를 쓴다 |
| `jariml/model.py` | Gradient Boosting 학습, 검증 기간으로 반복 횟수 결정 |
| `jariml/probability.py` | 검증 기간 오차 분포로 "빈자리 K대 이상" 확률 계산 |
| `jariml/evaluate.py` | MSE / RMSE / MAE / MSLE (빈자리 대수), 기준선 비교, 확률 검증 |
| `jariml/registry.py` | 모델 저장, 운영 버전(`models/LATEST`) 관리, 더 좋을 때만 교체 |
| `jariml/train.py` | 학습 파이프라인 (`python -m jariml.train`) |
| `jariml/predictor.py` | 서비스용 예측: `predict`, `deadline` |
| `jariml/fee.py` | 예상 주차 요금 (환승 감면 포함) |
| `api/app.py` | FastAPI 예시 |
| `models/` | 학습된 모델. `LATEST`가 운영 버전 |
| `tests/` | 가짜 데이터 파이프라인 테스트 |
| `notebooks/02_model.ipynb` | 학습 결과를 표와 그래프로 보는 노트북 |
| `Dockerfile` | API 컨테이너 예시 |
| `../.github/workflows/retrain_model.yml` | 재학습 워크플로 (수동 실행, 정기 실행은 주석 해제) |

## 데이터 흐름

```
data/live, static, reference (5분마다 수집)
  → preprocess.build_tables     예측 대상 주차장만 남긴 점유율 격자 (시각 × 주차장)
  → split.time_split            |-- train --|gap|-- val --|gap|-- test --|
  → features.build_rows         한 행 = (주차장, 기준 시각 t, 예측 거리 h), 정답 = t+h 점유율
  → model.fit_with_val          train으로 학습, val로 반복 횟수 결정
  → probability.fit_residuals   val 오차 분포 저장 (확률 계산용)
  → evaluate                    test에서 기준선과 비교, 확률 검증
  → 전체 기간 재학습 → registry.save → promote_if_better
```

**누설 방지**
- 기간은 시간 순서로 자르고, 경계 앞 2시간(`gap_steps`)은 기준 시각에서 뺀다. 정답이 다음 기간으로 넘어가지 않는다.
- 평소 곡선(주차장 × 평일/휴일 × 시각 평균)은 평가할 때 학습 기간으로만 만든다.
- 확률용 오차 분포는 학습에 쓰지 않은 검증 기간에서 모은다.

## 서비스 연동 (백엔드)

### 파이썬에서 직접

```python
from jariml import Predictor
p = Predictor.load()                              # models/LATEST
recent = p.recent_from_data_dir()                 # 또는 DB에서 같은 형태로 만든 표
p.predict(recent, arrival="2026-11-14 08:30")
p.deadline(recent, min_prob=0.8)
```

**입력 `recent`** (긴 표, 지금 기준 최소 65분치, 5분 간격)

| 컬럼 | 형식 | 내용 |
|---|---|---|
| t | datetime | 수집 시각 (5분 단위로 내림 처리됨) |
| PKLT_CD | str | 주차장 코드 |
| cnt | float | 주차 대수 (`NOW_PRK_VHCL_CNT`) |
| cap | float | 총면 (`TPKCT`) |

**`predict` 출력** (주차장 1곳 = 1행)

| 컬럼 | 내용 |
|---|---|
| PKLT_CD, PKLT_NM, is_pnr | 주차장 코드, 이름, 환승주차장 여부 |
| cap | 총면 |
| base_time | 기준 시각 (입력의 마지막 시각) |
| now_cnt, now_free | 지금 주차 대수, 빈자리 |
| arrival, horizon_min | 도착 시각 (5분 단위), 기준 시각부터 몇 분 뒤 |
| pred_cnt, pred_free | 도착 시점 예상 주차 대수, 빈자리 |
| prob_free_ge_3 | 도착 시점 빈자리 3대 이상일 확률 (0~1) |
| confidence | 주차장별 신뢰도 (검증 기간 1시간 뒤 오차 3대 이하 높음 / 6대 이하 보통 / 그 외 낮음) |
| method | `모델` (2시간 이내) / `평소 패턴(2시간 초과)` |

**`deadline` 출력**: 주차장별로 확률이 `min_prob` 이상인 마지막 도착 시각 (`deadline`), `note`에 "2시간 내내 가능" / "지금 출발해도 불확실".

### REST API (`api/app.py`)

| 메서드 | 경로 | 내용 |
|---|---|---|
| GET | `/health` | 모델 학습 시점과 평가 요약 |
| GET | `/lots?model_only=true` | 주차장 정보 + 요금 규칙 |
| POST | `/predict` | 본문 `{"arrival": "2026-11-14T08:30:00", "lots": ["1051043"]}` → 위 출력 표 |
| GET | `/deadline?min_prob=0.8&lots=1051043` | 마지막 도착 시각 |

최근 수집 기록은 `recent()` 함수가 읽는다 (지금은 data/live CSV, 60초 캐시). 운영 DB가 생기면 이 함수만 바꾼다.

## 운영 (MLOps)

- **재학습**: `python -m jariml.train`. 새 모델은 `models/<버전>/`에 저장되고, `config.yaml`의 `promote_metric`(평가 기간 1시간 뒤 빈자리 MAE)이 운영 모델보다 작을 때만 `models/LATEST`가 바뀐다.
- **버전**: 폴더 이름이 학습 시각(KST). `metrics.json`에 데이터 기간, 분할, 기준선 비교, 확률 검증, 시간대별 오차가 남는다.
- **롤백**: `models/LATEST` 파일 내용을 이전 버전 폴더 이름으로 바꾼다.
- **자동화**: `.github/workflows/retrain_model.yml` (테스트 → 학습 → 모델 커밋).
- **기준값 변경**: `config.yaml`만 고친다 (예측 거리, 확률 기준 K, 분할 일수, 제외 기준 등).

## 모델 요약

- 대상: 시영주차장 중 실시간 연계, 승용차 가능, 20면 이상, 센서 정상인 곳 (2026-11-13 기준 55곳)
- 예측: 5분~2시간 뒤 점유율. 한 모델이 예측 거리 `h`를 입력으로 받는다. 2시간을 넘으면 평소 곡선 값을 쓴다
  (EDA에서 평소와의 차이가 약 4시간 뒤 사라져, 먼 미래는 평소 패턴이 가장 정직한 추정이다)
- 입력: 지금 점유율, 최근 5/15/30/60분 변화량, 시각, 요일, 휴일 여부, 총면, 평소 곡선(지금 / 도착 시각), 평소와의 차이
- 사용하지 않는 데이터: 도시데이터(인구, 날씨, 도로, 상권, 행사). 넣고 뺀 실험에서 오차가 줄지 않았다
