"""시간 순서대로 학습 / 검증 / 평가 기간 나누기.

무작위로 섞으면 미래 정보가 학습에 섞여 점수가 실제보다 좋게 나온다. 그래서 시간으로 자른다.
기간 경계 앞 gap_steps 칸은 기준 시각에서 뺀다. 정답(t + h)이 다음 기간으로 넘어가지 않게 하기 위해서다.

    |------ train ------|gap|---- val ----|gap|---- test ----|
"""
from dataclasses import dataclass

import pandas as pd


@dataclass
class Split:
    train: pd.DatetimeIndex
    val: pd.DatetimeIndex
    test: pd.DatetimeIndex
    val_start: pd.Timestamp
    test_start: pd.Timestamp
    end: pd.Timestamp

    def describe(self) -> pd.DataFrame:
        rng = lambda ix: (ix.min(), ix.max(), len(ix), round(len(ix) / 288, 1))
        return pd.DataFrame([rng(self.train), rng(self.val), rng(self.test)],
                            index=["train", "val", "test"], columns=["시작", "끝", "기준 시각 수", "일수"])


def time_split(index: pd.DatetimeIndex, test_days: int, val_days: int, gap_steps: int, step_min: int = 5) -> Split:
    end = index.max()
    test_start = end.normalize() - pd.Timedelta(days=test_days - 1)      # 마지막 날(진행 중인 날 포함)부터 거꾸로
    val_start = test_start - pd.Timedelta(days=val_days)
    gap = pd.Timedelta(minutes=step_min * gap_steps)
    train = index[index < val_start - gap]
    val = index[(index >= val_start) & (index < test_start - gap)]
    test = index[(index >= test_start) & (index <= end - gap)]
    if len(train) < 288 * 5:
        raise ValueError(f"학습 기간이 5일보다 짧다 ({len(train) / 288:.1f}일). test_days/val_days를 줄이거나 데이터를 더 모은다.")
    return Split(train, val, test, val_start, test_start, end)
