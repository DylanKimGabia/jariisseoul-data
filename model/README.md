# model/ – 예측 모델과 추천 MVP

| 파일 | 내용 |
|---|---|
| `01_mvp_seq2seq.ipynb` | 데이터 정제 → 평소 곡선 → 도시데이터 결합 → seq2seq 학습 → 성능(MSE·RMSE·MSLE·MAE) → 자리 있을 확률 → 길찾기 → 추천 |
| `app.py` | Streamlit 데모 화면: 출발지·목적지 입력 → 주차장 추천 + 확률, 지도에서 약 10초 이동 애니메이션 |
| `requirements.txt` | 필요한 패키지 |
| `artifacts/` | 실행하면 생김: 학습된 모델(`seq2seq.pt`), 서울 도로망 캐시(`seoul_drive.graphml`). 저장소에 올리지 않음 |

## 실행

```bash
git clone https://github.com/DylanKimGabia/jariisseoul-data
cd jariisseoul-data/model
pip install -r requirements.txt
jupyter notebook 01_mvp_seq2seq.ipynb
```

- 노트북은 `../data`를 읽는다. 최신 데이터로 돌리려면 실행 전에 `git pull`.
- 9번 길찾기 셀은 처음 한 번 OpenStreetMap에서 서울 도로망을 받는다(몇 분, 인터넷 필요). 실패하면 직선거리 근사로 자동 대체.
- 시험 기간은 마지막 1일, 검증은 그 전 1일. 데이터가 2주 이상 쌓이면 `TEST_DAYS`를 늘린다.

## 데모 화면 실행

1. 노트북을 끝까지 실행 (마지막 11번 셀이 `artifacts/app_bundle.pkl` 저장, 9번 셀이 `artifacts/seoul_drive.graphml` 저장)
2. 터미널에서
   ```bash
   cd jariisseoul-data/model
   streamlit run app.py
   ```
3. 왼쪽에서 출발지·목적지·시각을 고르고 **자리 찾기**

- "데이터 최신 시각": 노트북을 마지막으로 실행할 때까지의 데이터 기준 예측
- "과거 재생": 과거 시각을 골라 예측과 실제 빈자리를 비교

## 구조 요약

- 단위: (주차장, 5분 시각). 입력 과거 2시간 → 출력 앞으로 2시간
- 모델은 "지금 값 + 평소 변화량" 기준선에서의 **보정값**을 학습한다
- 제외: 실시간 미연계, 관광버스 전용, 20면 미만, 기간 내내 값이 안 바뀐 주차장
- 도시데이터는 `collected_at`(받은 시각) 기준 as-of join으로만 붙인다 (미래 정보 누설 방지)
- 확률: 검증 기간 오차 분포(시점 × 점유율 구간)로 "빈자리 3개 이상일 확률" 계산, 시험 기간에서 보정 확인
- 길찾기: OSMnx 최단경로 × 도시데이터 실시간 도로속도 보정. 지하철 시간은 근사치
