"""모델 입력(특징) 만들기. 학습과 예측이 같은 함수를 쓴다.

한 행 = (주차장, 기준 시각 t, 예측 거리 h). 정답 y = t + h 시점 점유율.
모델은 지금 대비 변화량(y - occ)을 맞힌다.
"""
import numpy as np
import pandas as pd

from .preprocess import is_offday

FEATURES = [
    "h",                         # 예측 거리 (5분 칸)
    "occ",                       # 지금 점유율
    "d1", "d3", "d6", "d12",     # 5분 / 15분 / 30분 / 60분 전 대비 변화량
    "tod", "ttod",               # 지금 / 도착 시각의 하루 중 5분 칸 번호 (0~287)
    "dow", "off", "toff",        # 요일, 지금이 휴일인지, 도착 시점이 휴일인지
    "cap",                       # 총면
    "p_now", "p_fut", "p_delta", # 평소 곡선: 지금 시각 / 도착 시각 / 그 차이
    "gap",                       # 오늘이 평소보다 얼마나 더 찼나 (EDA: 1시간 뒤까지 66% 이어짐)
]


def fit_profile(occ: pd.DataFrame, holidays) -> dict:
    """평소 곡선: (평일/휴일) × 하루 중 5분 칸 × 주차장 평균 점유율.

    돌려주는 값: {"values": 배열 (2, 288, 주차장 수), "lots": 주차장 코드 목록}
    """
    off = is_offday(occ.index, holidays).values.astype(int)
    tod = (occ.index.hour * 12 + occ.index.minute // 5).values
    g = occ.groupby([off, tod]).mean().reindex(pd.MultiIndex.from_product([[0, 1], range(288)]))
    arr = g.to_numpy(dtype="float32").reshape(2, 288, -1).copy()
    arr[1] = np.where(np.isnan(arr[1]), arr[0], arr[1])   # 휴일 자료가 없는 칸은 같은 시각 평일 값으로
    return {"values": arr, "lots": list(occ.columns)}


def build_rows(occ: pd.DataFrame, cap: pd.DataFrame, profile, holidays, horizons, origins=None, with_target=True):
    """특징 표 만들기.

    occ, cap: 시각 × 주차장 격자. origins: 기준 시각으로 쓸 시각 (None이면 전부).
    with_target=False면 정답 없이 특징만 만든다 (예측용).
    """
    lots = list(occ.columns)
    O = occ.to_numpy(dtype="float32")
    C = cap.to_numpy(dtype="float32")
    T, N = O.shape
    idx = occ.index
    pos = np.arange(T) if origins is None else idx.get_indexer(origins)
    pos = pos[pos >= 12]                                # 60분 전 값이 있어야 함

    off_all = is_offday(idx, holidays).values.astype(int)
    tod_all = (idx.hour * 12 + idx.minute // 5).values
    col = pd.Index(profile["lots"]).get_indexer(lots)
    if (col < 0).any():
        raise ValueError(f"평소 곡선이 없는 주차장: {[l for l, c in zip(lots, col) if c < 0]}")
    prof = profile["values"][:, :, col]                 # (2, 288, N)

    base = {
        "occ": O[pos], "cap": C[pos],
        "d1": O[pos] - O[pos - 1], "d3": O[pos] - O[pos - 3],
        "d6": O[pos] - O[pos - 6], "d12": O[pos] - O[pos - 12],
    }
    off, tod = off_all[pos], tod_all[pos]
    dow = idx.dayofweek.values[pos]
    p_now = prof[off, tod]                              # (행, N)

    frames = []
    for h in horizons:
        tpos = pos + h
        ok = tpos < T if with_target else np.ones(len(pos), bool)
        p = pos[ok]
        t_future = idx[p] + pd.Timedelta(minutes=5 * h)
        toff = is_offday(t_future, holidays).values.astype(int)
        ttod = (t_future.hour * 12 + t_future.minute // 5).values
        p_fut = prof[toff, ttod]
        k = len(p)
        f = {name: v[ok].reshape(-1) for name, v in base.items()}
        f.update(
            h=np.full(k * N, h, "int16"),
            tod=np.repeat(tod[ok], N), ttod=np.repeat(ttod, N), dow=np.repeat(dow[ok], N),
            off=np.repeat(off[ok], N), toff=np.repeat(toff, N),
            p_now=p_now[ok].reshape(-1), p_fut=p_fut.reshape(-1),
        )
        f["p_delta"] = f["p_fut"] - f["p_now"]
        f["gap"] = f["occ"] - f["p_now"]
        f["t"] = np.repeat(idx[p].values, N)
        f["PKLT_CD"] = np.tile(lots, k)
        if with_target:
            f["y"] = O[tpos[ok]].reshape(-1)
        frames.append(pd.DataFrame(f))
    X = pd.concat(frames, ignore_index=True)
    keep = X.occ.notna() & (X.y.notna() if with_target else True)
    return X[keep].reset_index(drop=True)
