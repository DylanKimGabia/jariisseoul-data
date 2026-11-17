"""음성(STT로 받은 문장) → 검색 조건, 추천 결과 → 읽어 줄 문장(TTS).

예: "수원 SK스카이뷰에서 교대역까지 내일 아침 8시 반에 출발"
    → {"origin": "수원 SK스카이뷰", "dest": "교대역", "depart": "2026-11-19 08:30"}
    "강남역 근처 주차할 곳 알려줘" → 출발지 없음(현재 위치 사용), 목적지 강남역, 지금 출발

[주1] 브라우저 음성 인식은 숫자를 보통 아라비아 숫자로 돌려준다 ("8시 30분"). 한글 숫자("여덟 시")는 자주 쓰는 것만 바꾼다.
"""
import re

import pandas as pd

from .recommend import kst_now

FILLER = r"(주차\s*할\s*(곳|데|만한\s*곳)|주차장|추천\s*해\s*줘|알려\s*줘|찾아\s*줘|좀|부탁\s*해|있어\??|어디야\??|가려고|갈\s*건데|가는데|가요|갈래|가고\s*싶어|출발\s*(해|할\s*거야|하면)?)"
HERE = r"(여기|지금\s*위치|현재\s*위치|내\s*위치|집)"
KO_NUM = {"한": 1, "두": 2, "세": 3, "네": 4, "다섯": 5, "여섯": 6, "일곱": 7, "여덟": 8, "아홉": 9, "열한": 11, "열두": 12, "열": 10}
PLACE_END = r"(역|점|관|원|교|청|장|터|소|당|탑|구|동|시|몰|스|타워|센터|공원|병원|학교|마트|아파트|빌딩|호텔|홀|랜드|파크|타운|플라자|월드|스퀘어)"


def _time(s: str, now: pd.Timestamp):
    """문장에서 출발 시각을 꺼내고, 시각 표현을 지운 문장을 돌려준다. 시각이 없으면 None(지금)."""
    for k, v in KO_NUM.items():
        s = re.sub(rf"{k}\s*시", f"{v}시", s)
    m = re.search(r"(\d+)\s*(분|시간)\s*(뒤|후)(에)?", s)
    if m:
        t = now + pd.Timedelta(minutes=int(m[1]) * (60 if m[2] == "시간" else 1))
        return t.floor("min"), s.replace(m[0], " ")
    day = 0
    for word, add in [("모레", 2), ("내일", 1), ("오늘", 0)]:
        if word in s:
            day, s = add, s.replace(word, " ")
            break
    ampm = None
    m = re.search(r"(오전|아침|새벽|오후|저녁|밤|점심)", s)
    if m:
        ampm, s = m[1], s.replace(m[0], " ")
    m = re.search(r"(\d{1,2})\s*(?::\s*(\d{2})|시\s*(?:(\d{1,2})\s*분|(반))?)(에)?", s)
    if not m:
        if day:                                            # "내일"만 말하면 내일 같은 시각
            return (now + pd.Timedelta(days=day)).floor("min"), s
        return None, s
    h = int(m[1])
    mi = int(m[2] or m[3] or 0) + (30 if m[4] else 0)
    if ampm in ("오후", "저녁", "밤") and h < 12:
        h += 12
    if ampm == "점심" and h < 6:
        h += 12
    t = now.normalize() + pd.Timedelta(days=day, hours=h % 24, minutes=mi)
    if not day and ampm is None and t < now - pd.Timedelta(minutes=10) and h < 12 and t + pd.Timedelta(hours=12) > now:
        t += pd.Timedelta(hours=12)                        # "3시"라고만 하고 이미 지났으면 오후 3시로 본다
    elif not day and t < now - pd.Timedelta(minutes=10):
        t += pd.Timedelta(days=1)
    return t, s.replace(m[0], " ")


def _clean_place(p: str) -> str:
    p = re.sub(r"\s+", " ", p).strip(" ,.?!")
    p = re.sub(r"\s*(까지|으로|에)$", "", p)
    if re.search(PLACE_END + r"로$", p):                  # "교대역로"처럼 장소 끝 + 로 일 때만 '로'를 뗀다 (세종로 보호)
        p = p[:-1]
    return re.sub(r"\s*(근처|주변|쪽|부근)$", "", p).strip()


def parse_voice(text: str, now=None) -> dict:
    """반환: {origin (None이면 현재 위치), dest, depart (None이면 지금), text}."""
    now = now or kst_now()
    depart, s = _time(text, now)
    s = re.sub(FILLER, " ", s)
    s = re.sub(r"(에\s*)?출발", " ", s)
    s = re.sub(rf"^\s*{HERE}\s*(에)?서\s+", " ", s)               # "여기서 강남역까지" → 출발지는 현재 위치
    m = re.search(r"(.+?)\s*에서\s+(.+)|(.+?)에서(.+)", s)
    if m:
        origin, dest = (m[1], m[2]) if m[1] else (m[3], m[4])
        origin = _clean_place(origin)
        origin = None if re.fullmatch(HERE, origin or "") else origin
    else:
        origin, dest = None, s
    dest = _clean_place(dest)
    if not dest:
        raise ValueError(f"목적지를 알아듣지 못했다: '{text}'. 예: '강남역에서 교대역까지 내일 8시'")
    return {"origin": origin, "dest": dest, "depart": depart, "text": text}


def speech(result: dict, n: int = 3) -> str:
    """추천 결과를 읽어 줄 문장. 상위 n곳만."""
    t, m = result["table"], result["meta"]
    if t.empty:
        return "추천할 주차장을 찾지 못했어요."
    lines = [f"{m['dest']['name']} 기준 추천이에요."]
    if m.get("note"):
        lines.append("목적지 걸어갈 거리 안의 공영주차장이 적어서, 조금 떨어진 곳도 함께 골랐어요.")
    for _, x in t.head(n).iterrows():
        name = re.sub(r"\s*공영주차장.*$", "", x["name"])
        leg = "걸어서" if x.last_leg == "도보" else "대중교통으로"
        lines.append(f"{x['rank']}순위 {name}, 자리 있을 확률 {x.prob:.0f}퍼센트, 예상 빈자리 {x.pred_free:.0f}대, "
                     f"도착 후 {leg} {x.last_min:.0f}분, 총 {x.total_min:.0f}분이에요.")
    return " ".join(lines)
