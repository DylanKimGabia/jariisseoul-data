"""엔진 목록. 새 엔진을 만들면 여기에 등록하고 config.yaml의 model.engine을 바꾼다."""
from .base import INTERFACE_VERSION, Engine
from .gbm import GBMEngine

ENGINES = {GBMEngine.name: GBMEngine}


def make_engine(cfg: dict) -> Engine:
    name = cfg["model"]["engine"]
    if name not in ENGINES:
        raise ValueError(f"등록되지 않은 엔진: {name}. 사용 가능: {list(ENGINES)}")
    return ENGINES[name](cfg)


__all__ = ["Engine", "ENGINES", "INTERFACE_VERSION", "make_engine"]
