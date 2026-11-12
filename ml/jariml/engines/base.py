"""엔진 규약: 모든 예측 모델(ML, DL)이 지켜야 하는 틀.

학습, 평가, 확률, 저장, 예측 API는 이 틀만 쓴다. 그래서 엔진을 바꿔도 서비스 코드는 그대로다.

입력 (모든 메서드 공통)
    occ : DataFrame  시각(5분 간격) × 주차장 코드, 점유율 0~1, 결측은 NaN
    cap : DataFrame  occ와 같은 모양, 그 시점 총면
출력 (predict)
    DataFrame  t (기준 시각), PKLT_CD, h (예측 거리, 5분 칸), pred (t + h 시점 예측 점유율 0~1)

새 엔진 만들기
    1. Engine을 상속해 name, history_steps, fit, predict, fit_full을 구현한다.
    2. engines/__init__.py의 ENGINES에 등록한다.
    3. config.yaml의 model.engine을 그 이름으로 바꾸고 python -m jariml.train.
    저장은 엔진 객체를 통째로 joblib으로 한다. 직렬화가 안 되는 부분(예: GPU 텐서)은 __getstate__/__setstate__로 처리한다.
"""
from abc import ABC, abstractmethod

import pandas as pd

INTERFACE_VERSION = 1


class Engine(ABC):
    name: str = "base"
    history_steps: int = 13          # 예측할 때 필요한 과거 길이 (지금 포함, 5분 칸)

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.info: dict = {}         # 학습 정보 (반복 횟수, 학습 곡선 등). metrics.json에 남는다

    @abstractmethod
    def fit(self, occ: pd.DataFrame, cap: pd.DataFrame, split) -> dict:
        """split.train 기간으로 학습하고 split.val 기간으로 조정한다. 평가 기간(split.test)은 보지 않는다."""

    @abstractmethod
    def predict(self, occ: pd.DataFrame, cap: pd.DataFrame, origins, horizons) -> pd.DataFrame:
        """origins(기준 시각들) × horizons(5분 칸) × 주차장 예측. 출력 컬럼: t, PKLT_CD, h, pred."""

    @abstractmethod
    def fit_full(self, occ: pd.DataFrame, cap: pd.DataFrame) -> None:
        """운영용: fit에서 정한 설정 그대로 전체 기간으로 다시 학습한다."""
