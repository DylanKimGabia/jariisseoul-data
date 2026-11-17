"""AI 에이전트: 아무렇게나 말한 문장 → 추천 모델 입력값 (GPT 구조화 출력).

    "내일 아침 회의가 9시인데 교대역 근처 세울 데 있을까? 수원 집에서 나가" →
    {origin: "수원 집", dest: "교대역", arrive_by: "2026-11-19 09:00", prefer: "기본", question: null}

- 키: 환경 변수 또는 .env의 OPENAI_API_KEY. 모델: OPENAI_MODEL (기본 gpt-4.1-mini).
- 대화: history(이전 대화)를 함께 보내면 "30분 늦게", "아까 거기서"처럼 이어 말해도 앞 내용을 반영한다.
- 빠진 정보(목적지 등)가 있으면 question에 되물을 말을 담는다.
- 키가 없거나 GPT 호출이 실패하면 규칙 해석기(voice.parse_voice)로 대신한다. [주1] 시연 중 멈추지 않게 하기 위해서다.
"""
import json
import urllib.error
import urllib.request

import pandas as pd

from .recommend import PREFER_SORT, kst_now
from .tmap import find_key
from .voice import parse_voice

URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = "gpt-4.1-mini"          # [주2] 빠르고 저렴한 모델. .env의 OPENAI_MODEL로 바꿀 수 있다
WEEK = "월화수목금토일"

SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["origin", "origin_is_here", "dest", "depart", "arrive_by", "stay_min", "prefer", "question", "understood"],
    "properties": {
        "origin": {"type": ["string", "null"], "description": "출발지 장소 이름. 말하지 않았거나 현재 위치면 null"},
        "origin_is_here": {"type": "boolean", "description": "출발지가 현재 위치(여기, 집이라고만 함, 말 안 함)면 true"},
        "dest": {"type": ["string", "null"], "description": "목적지 장소 이름 (TMAP 장소 검색에 넣을 이름). 모르면 null"},
        "depart": {"type": ["string", "null"], "description": "출발 시각 'YYYY-MM-DD HH:MM'. 지금 출발이면 null"},
        "arrive_by": {"type": ["string", "null"], "description": "목적지에 있어야 하는 시각 'YYYY-MM-DD HH:MM'. 없으면 null"},
        "stay_min": {"type": ["integer", "null"], "description": "주차할 시간(분). 말하지 않았으면 null"},
        "prefer": {"type": "string", "enum": list(PREFER_SORT), "description": "정렬 선호"},
        "question": {"type": ["string", "null"], "description": "목적지를 모르는 등 꼭 필요한 정보가 없을 때 되물을 말 (해요체, 한 문장). 없으면 null"},
        "understood": {"type": "string", "description": "이해한 내용 한 줄 요약 (해요체)"},
    },
}

SYSTEM = """너는 서울 공영주차장 추천 서비스 '자리있서울'의 입력 해석기다.
사용자가 음성으로 말한 문장(음성 인식 오류가 있을 수 있음)을 추천 모델 입력값으로 바꾼다.
지금 시각: {now} ({dow}요일, 한국 시각).

규칙:
- 장소 이름은 TMAP 장소 검색에 넣을 정확한 이름으로 쓴다. 음성 인식 오류는 고친다 (예: '스카이브' → '스카이뷰', '서울시청 역' → '서울시청역').
- 출발지를 말하지 않았거나 '여기', '지금 위치'면 origin=null, origin_is_here=true.
- 'N시까지 가야 돼', 'N시 약속/회의' 처럼 목적지에 있어야 하는 시각이면 arrive_by. 'N시에 출발'이면 depart. 둘 다 없으면 둘 다 null(지금 출발).
- 시각만 말하고 날짜가 없으면 오늘, 이미 지났으면 내일로 본다. '아침', '저녁' 등을 반영한다.
- prefer: '싼 데', '저렴한' → 저렴한 / '가까운', '덜 걷게' → 가까운 / '확실하게 자리 있는' → 확실한 / '환승주차장' → 환승주차장 / 그 외 → 기본.
- 목적지를 알 수 없을 때만 question에 짧게 되묻는다. 출발지는 되묻지 않는다(현재 위치를 쓴다).
- 이전 대화가 있으면 이어서 해석한다 ('30분 늦게', '거기 말고 강남역').
"""


def llm_available() -> bool:
    return find_key("OPENAI_API_KEY") is not None


def ask_gpt(text: str, history: list | None = None, now=None, timeout: float = 8) -> dict:
    """GPT로 해석. history: [{"role": "user"|"assistant", "content": str}, ...]."""
    key = find_key("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY가 없다")
    now = now or kst_now()
    msgs = [{"role": "system", "content": SYSTEM.format(now=f"{now:%Y-%m-%d %H:%M}", dow=WEEK[now.dayofweek])}]
    msgs += (history or [])[-8:] + [{"role": "user", "content": text}]
    body = {"model": find_key("OPENAI_MODEL") or DEFAULT_MODEL, "messages": msgs, "temperature": 0,
            "response_format": {"type": "json_schema", "json_schema": {"name": "parking_query", "strict": True, "schema": SCHEMA}}}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), method="POST",
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"OpenAI 오류 {e.code}: {e.read().decode('utf-8', 'ignore')[:300]}") from None
    return json.loads(out["choices"][0]["message"]["content"])


def understand(text: str, history: list | None = None, now=None) -> dict:
    """문장 → {origin, dest, depart, arrive_by, stay_min, prefer, question, understood, source, error}.
    origin None = 현재 위치. source: 'GPT' 또는 '규칙'(GPT를 못 쓸 때)."""
    now = now or kst_now()
    try:
        g = ask_gpt(text, history, now)
        ts = lambda v: pd.Timestamp(v) if v else None
        return {"origin": None if g["origin_is_here"] else g["origin"], "dest": g["dest"], "depart": ts(g["depart"]),
                "arrive_by": ts(g["arrive_by"]), "stay_min": g["stay_min"], "prefer": g["prefer"],
                "question": g["question"] if not g["dest"] else None, "understood": g["understood"],
                "source": "GPT", "error": None, "text": text}
    except Exception as e:                                       # [주1] 규칙 해석기로 대신
        err = str(e)[:200]
        try:
            q = parse_voice(text, now)
            return {**q, "stay_min": None, "prefer": "기본", "question": None, "understood": None, "source": "규칙", "error": err}
        except ValueError:
            return {"origin": None, "dest": None, "depart": None, "arrive_by": None, "stay_min": None, "prefer": "기본",
                    "question": "어디로 가세요? 목적지를 말해 주세요.", "understood": None, "source": "규칙", "error": err,
                    "text": text}
