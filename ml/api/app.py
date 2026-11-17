"""FastAPI 예시: 예측 모델을 REST API로 제공한다.

실행:   cd ml && uvicorn api.app:app --reload --port 8000
문서:   http://localhost:8000/docs

최근 수집 기록은 지금은 수집 저장소의 data/live CSV에서 읽는다 (60초 캐시).
운영에서는 recent()를 백엔드 DB 조회로 바꾸면 된다. 형태: t, PKLT_CD, cnt, cap.
"""
import json
import time
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from jariml import Predictor
from jariml.recommend import recommend as run_recommend
from jariml.tmap import Tmap, find_key

app = FastAPI(title="자리있서울 빈자리 예측 API")
predictor = Predictor.load()
_cache = {"at": 0.0, "recent": None}
tmap = Tmap(live=find_key() is not None)        # 키(환경 변수 TMAP_APP_KEY 또는 .env)가 없으면 직선거리 추정 모드


def records(df: pd.DataFrame) -> list[dict]:
    """표 → JSON (시각은 ISO 문자열, 빈 값은 null)."""
    return json.loads(df.to_json(orient="records", date_format="iso", force_ascii=False))


def recent() -> pd.DataFrame:
    if time.time() - _cache["at"] > 60:
        _cache["recent"], _cache["at"] = predictor.recent_from_data_dir(), time.time()
    return _cache["recent"]


class PredictRequest(BaseModel):
    arrival: datetime                       # 도착 예정 시각 (예: 2026-11-14T08:30:00)
    lots: list[str] | None = None           # 주차장 코드 목록. 비우면 전부


@app.get("/health")
def health():
    return {"status": "ok", "trained_until": str(predictor.a["trained_until"]), "summary": predictor.metrics.get("summary")}


@app.get("/lots")
def lots(model_only: bool = True):
    lm = predictor.a["lot_master"].join(predictor.a["fee_rule"])
    if model_only:
        lm = lm[lm.use_for_model == 1]
    return records(lm.reset_index())


@app.post("/predict")
def predict(req: PredictRequest):
    r = predictor.predict(recent(), req.arrival, lots=req.lots)
    if r.empty:
        raise HTTPException(404, "예측할 주차장이 없다")
    return records(r)


@app.get("/deadline")
def deadline(min_prob: float = Query(0.8, ge=0.5, le=0.99), lots: list[str] | None = Query(None)):
    return records(predictor.deadline(recent(), lots=lots, min_prob=min_prob))


@app.get("/recommend")
def recommend(origin: str = Query(..., description="출발지 (예: 강남역)"),
              dest: str = Query(..., description="목적지 (예: 서울시청)"),
              depart: datetime | None = Query(None, description="출발 시각. 비우면 지금"),
              stay_min: int = Query(120, description="주차 시간(분), 요금 계산용")):
    """출발지, 목적지 → 도착 시점 기준 목적지 주변 공영주차장 상위 5곳 (환승주차장 1곳 포함)."""
    try:
        r = run_recommend(origin, dest, predictor, recent(), tmap, depart=depart, stay_min=stay_min)
    except ValueError as e:
        raise HTTPException(404, str(e))
    meta = {k: v for k, v in r["meta"].items()}
    return {"meta": json.loads(json.dumps(meta, default=str, ensure_ascii=False)), "results": records(r["table"])}
