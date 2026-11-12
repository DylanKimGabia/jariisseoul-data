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

app = FastAPI(title="자리있서울 빈자리 예측 API")
predictor = Predictor.load()
_cache = {"at": 0.0, "recent": None}


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
