"""예상 주차 요금 (예측에는 쓰지 않고 화면의 요금 비교용).

서울시 시영주차장 요금 구조: 기본 시간까지 기본 요금, 이후 추가 시간 단위마다 추가 요금, 일 최대 요금 상한.
환승주차장은 대중교통 환승 시 감면율(pnr_discount)을 적용한다.

주의: 서울시 API 값을 그대로 해석한 추정 요금이다. 토요일/일요일·공휴일 무료 여부(SAT_PAY, HOLIDAY_PAY: Y 유료, N 무료)와
일 최대 요금은 API에 0 또는 N으로 들어온 곳이 많아, 실제 현장 요금과 다를 수 있다. 일요일은 공휴일 기준을 따른다고 가정했다.
"""
import math

import pandas as pd

from .preprocess import is_offday


def parking_fee(rule: pd.Series, start, minutes: int, transfer: bool = False, holidays=()) -> int:
    """rule: fee_rule 한 행. start: 입차 시각. minutes: 주차 시간(분). transfer: 대중교통 환승 여부."""
    start = pd.Timestamp(start)
    if rule.PAY_YN == "N":
        return 0
    if start.dayofweek == 5 and rule.SAT_PAY == "N":                  # 토요일 무료
        return 0
    if (start.dayofweek == 6 or is_offday([start], holidays).iloc[0]) and rule.HOLIDAY_PAY == "N":
        return 0                                                       # 일요일·공휴일 무료
    fee = rule.BSC_CRG
    if minutes > rule.BSC_MIN and rule.ADD_MIN > 0:
        fee += math.ceil((minutes - rule.BSC_MIN) / rule.ADD_MIN) * rule.ADD_CRG
    if rule.DAY_MAX > 0:
        fee = min(fee, rule.DAY_MAX)
    if transfer:
        fee *= 1 - rule.pnr_discount
    return int(round(fee, -1))                                         # 10원 단위
