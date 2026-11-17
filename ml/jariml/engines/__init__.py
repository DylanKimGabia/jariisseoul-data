"""엔진 목록. 새 엔진을 만들면 여기에 등록하고 config.yaml의 model.engine을 바꾼다."""
from .base import INTERFACE_VERSION, Engine
from .gbm import GBMEngine
from .gru import GRUEngine
from .tabular import LGBMEngine, MLPEngine, RFEngine, RidgeEngine, XGBEngine

# 무거운 라이브러리(lightgbm, xgboost, torch)는 그 엔진을 실제로 쓸 때만 불러온다
ENGINES = {e.name: e for e in (GBMEngine, RidgeEngine, RFEngine, LGBMEngine, XGBEngine, MLPEngine, GRUEngine)}


def make_engine(cfg: dict) -> Engine:
    name = cfg["model"]["engine"]
    if name not in ENGINES:
        raise ValueError(f"등록되지 않은 엔진: {name}. 사용 가능: {list(ENGINES)}")
    return ENGINES[name](cfg)


__all__ = ["Engine", "ENGINES", "INTERFACE_VERSION", "make_engine"]
