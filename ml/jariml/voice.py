"""음성(STT로 받은 문장) → 검색 조건.

예: "수원 SK스카이뷰에서 교대역까지 내일 아침 8시 반에 출발" → 출발지, 목적지, 출발 08:30
    "수원 sk 스카이뷰부터 서울시청역까지 내일 아침 8시 반까지 가야 돼" → 출발지, 목적지, 도착 목표 08:30
    "강남역 근처 주차할 곳 알려줘" → 출발지 없음(현재 위치), 목적지 강남역, 지금 출발

[주1] 브라우저 음성 인식은 숫자를 보통 아라비아 숫자로 돌려준다 ("8시 30분"). 한글 숫자("여덟 시")는 자주 쓰는 것만 바꾼다.
[주2] 시각 바로 뒤에 '까지', '도착'이 오면 출발 시각이 아니라 도착 목표 시각으로 본다.
"""
import re

import pandas as pd

from .recommend import kst_now

HERE = r"(여기|지금\s*위치|현재\s*위치|내\s*위치|집)"
KO_NUM = {"열한": 11, "열두": 12, "다섯": 5, "여섯": 6, "일곱": 7, "여덟": 8, "아홉": 9, "한": 1, "두": 2, "세": 3, "네": 4, "열": 10}
PLACE_END = r"(역|점|관|원|교|청|장|터|소|당|탑|구|동|시|몰|스|타워|센터|공원|병원|학교|마트|아파트|빌딩|호텔|홀|랜드|파크|타운|플라자|월드|스퀘어)"
# 목적지 뒤에 오는 말: 여기서 목적지 이름이 끝난다. '로'는 장소 이름 끝(역, 센터 등) 뒤일 때만 조사로 본다 (세종로, 을지로 보호)
DEST_END = (r"까지|으로|로(?=\s|$)|에\s*(?=가|갈|도착)|\s+(가|갈|가는|가려|가야|가고|도착|근처|주변|쪽|부근|주차)"
            r"|근처|주변|$")


def _time(s: str, now: pd.Timestamp):
    """문장에서 시각을 꺼낸다. 반환: (시각 또는 None, 도착 목표인지, 시각 표현을 지운 문장)."""
    for k, v in KO_NUM.items():
        s = re.sub(rf"{k}\s*시", f"{v}시", s)
    m = re.search(r"(\d+)\s*(분|시간)\s*(뒤|후)(에)?", s)
    if m:
        t = now + pd.Timedelta(minutes=int(m[1]) * (60 if m[2] == "시간" else 1))
        return t.floor("min"), False, s.replace(m[0], " ")
    day = 0
    for word, add in [("모레", 2), ("내일", 1), ("오늘", 0)]:
        if word in s:
            day, s = add, s.replace(word, " ")
            break
    ampm = None
    m = re.search(r"(오전|아침|새벽|오후|저녁|밤|점심)", s)
    if m:
        ampm, s = m[1], s.replace(m[0], " ")
    m = re.search(r"(\d{1,2})\s*(?::\s*(\d{2})|시\s*(?:(\d{1,2})\s*분|(반))?)", s)
    if not m:
        return ((now + pd.Timedelta(days=day)).floor("min") if day else None), False, s
    arrive = bool(re.match(r"\s*(에)?\s*(반)?\s*(까지|도착)", s[m.end():]))      # [주2]
    h, mi = int(m[1]), int(m[2] or m[3] or 0) + (30 if m[4] else 0)
    if ampm in ("오후", "저녁", "밤") and h < 12:
        h += 12
    if ampm == "점심" and h < 6:
        h += 12
    t = now.normalize() + pd.Timedelta(days=day, hours=h % 24, minutes=mi)
    if not day and ampm is None and h < 12 and now - pd.Timedelta(minutes=10) > t > now - pd.Timedelta(hours=12):
        t += pd.Timedelta(hours=12)                        # "3시"라고만 하고 이미 지났으면 오후 3시로 본다
    elif not day and t < now - pd.Timedelta(minutes=10):
        t += pd.Timedelta(days=1)
    tail = re.match(r"\s*(에)?\s*(반)?\s*(까지|도착\S*)?", s[m.end():])         # 시각 뒤 '에', '까지' 등도 지운다
    return t, arrive, s[:m.start()] + " " + s[m.end() + tail.end():]


def _place(p: str) -> str:
    p = re.sub(r"\s+", " ", p).strip(" ,.?!")
    p = re.sub(r"\s+(?=" + PLACE_END + r"$)", "", p)        # "서울시청 역" → "서울시청역"
    return p.strip()


def _dest(rest: str) -> str:
    """출발지 다음 말에서 목적지 이름만 꺼낸다: 처음 나오는 끝말(까지, 으로, 가는 등) 앞까지."""
    for m in re.finditer(DEST_END, rest):
        cand = rest[:m.start()]
        if m[0].startswith("로") and not re.search(PLACE_END + r"\s*$", cand):
            continue
        if cand.strip():
            return _place(cand)
    return _place(rest)


def parse_voice(text: str, now=None) -> dict:
    """반환: {origin (None이면 현재 위치), dest, depart, arrive_by, text}. depart, arrive_by 둘 다 None이면 지금 출발."""
    now = now or kst_now()
    t, arrive, s = _time(text, now)
    s = re.sub(r"(주차\s*할\s*(곳|데|만한\s*곳)|주차장|추천\s*해\s*줘|알려\s*줘|찾아\s*줘)", " ", s)
    s = re.sub(rf"^\s*{HERE}\s*(에)?서\s+", " ", s)          # "여기서 강남역까지" → 현재 위치
    m = re.search(r"^(.*?\S)\s*(에서|부터)\s*(.*)$", s)
    if m:
        origin = _place(m[1])
        origin = None if re.fullmatch(HERE, origin) else origin
        rest = m[3]
    else:
        origin, rest = None, s
    dest = _dest(rest)
    if not dest:
        raise ValueError(f"목적지를 알아듣지 못했다: '{text}'. 예: '강남역에서 교대역까지 내일 8시'")
    return {"origin": origin, "dest": dest, "depart": None if arrive else t, "arrive_by": t if arrive else None, "text": text}
