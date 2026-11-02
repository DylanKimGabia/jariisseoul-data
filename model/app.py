"""자리있서울 데모 화면 (Streamlit)

실행:  cd model  →  streamlit run app.py
필요한 파일 (노트북 01_mvp_seq2seq.ipynb를 끝까지 실행하면 생김)
  artifacts/app_bundle.pkl      학습된 모델 + 전처리 데이터
  artifacts/seoul_drive.graphml 서울 도로망 (없으면 직선 경로로 대체)
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # Windows+아나콘다 torch 충돌 방지

import json
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import torch
import torch.nn as nn

ART = Path(__file__).parent / "artifacts"
st.set_page_config(page_title="자리있서울", page_icon="🅿️", layout="wide")

# 자주 쓰는 장소 (지오코딩 없이 바로 쓰기 위한 좌표)
PLACES = {
    "강남역": (37.4979, 127.0276), "역삼역": (37.5006, 127.0364), "삼성역(코엑스)": (37.5088, 127.0631),
    "잠실역": (37.5133, 127.1001), "양재역": (37.4844, 127.0343), "수서역": (37.4874, 127.1018),
    "서울역": (37.5547, 126.9707), "광화문": (37.5714, 126.9765), "종로3가": (37.5704, 126.9921),
    "동대문(DDP)": (37.5665, 127.0092), "명동": (37.5609, 126.9863), "여의도": (37.5216, 126.9242),
    "홍대입구역": (37.5572, 126.9245), "신촌역": (37.5552, 126.9369), "건대입구역": (37.5404, 127.0692),
    "천호역": (37.5386, 127.1238), "구파발역": (37.6366, 126.9188), "도봉산역": (37.6896, 127.0457),
    "고양 삼송": (37.6530, 126.8955), "성남 판교": (37.3947, 127.1112), "하남 미사": (37.5627, 127.1937),
}


# ---------------------------------------------------------------- 데이터·모델
class Seq2Seq(nn.Module):
    def __init__(self, n_lots, f_enc, f_dec, hidden=64, emb=8):
        super().__init__()
        self.emb = nn.Embedding(n_lots, emb)
        self.enc = nn.GRU(f_enc + emb, hidden, batch_first=True)
        self.dec = nn.GRU(f_dec + emb, hidden, batch_first=True)
        self.out = nn.Linear(hidden, 1)

    def forward(self, xe, xd, lot):
        e = self.emb(lot)
        _, h = self.enc(torch.cat([xe, e[:, None].expand(-1, xe.size(1), -1)], -1))
        z, _ = self.dec(torch.cat([xd, e[:, None].expand(-1, xd.size(1), -1)], -1), h)
        return self.out(z).squeeze(-1)


@st.cache_resource(show_spinner="모델 불러오는 중...")
def load_bundle():
    path = ART / "app_bundle.pkl"
    if not path.exists():
        return None, None
    with open(path, "rb") as f:
        b = pickle.load(f)
    m = b["MODEL"]
    model = Seq2Seq(m["n_lots"], m["f_enc"], m["f_dec"])
    model.load_state_dict(m["state"])
    model.eval()
    return b, model


@st.cache_resource(show_spinner="서울 도로망 불러오는 중...")
def load_graph():
    """도로망 + 가장 가까운 교차로를 빨리 찾기 위한 KD-트리"""
    path = ART / "seoul_drive.graphml"
    if not path.exists():
        return None, None, None
    try:
        import osmnx as ox
        from scipy.spatial import cKDTree
        G = ox.load_graphml(path)
        ids = np.array(list(G.nodes))
        ys = np.array([G.nodes[n]["y"] for n in ids]); xs = np.array([G.nodes[n]["x"] for n in ids])
        tree = cKDTree(np.c_[ys, xs * math.cos(math.radians(37.55))])
        return G, tree, ids
    except Exception:
        return None, None, None


if not (ART / "app_bundle.pkl").exists():     # 파일이 없을 때는 캐시하지 않음
    st.error("artifacts/app_bundle.pkl 이 없습니다. 노트북 01_mvp_seq2seq.ipynb의 마지막 셀(11번)까지 실행하세요.")
    st.stop()
B, MODEL = load_bundle()
if B is None:
    st.error("artifacts/app_bundle.pkl 이 없습니다. 노트북 01_mvp_seq2seq.ipynb의 마지막 셀(11번)까지 실행하세요.")
    st.stop()
GRAPH, TREE, NODE_IDS = load_graph()

CFG = B["CFG"]
STEP, LOOKBACK, HORIZON, FREE_K = CFG["STEP_MIN"], CFG["LOOKBACK"], CFG["HORIZON"], CFG["FREE_K"]
TIMES, NAMES, CAP = B["TIMES"], B["NAMES"], B["CAP"]
LAT, LNG = np.asarray(B["LAT"], float), np.asarray(B["LNG"], float)
OCC, PROF, PROF_TAB, RESID = B["OCC"], B["PROF"], B["PROF_TAB"], B["RESID"]
ENC, FPPL, ERR, OCC_BINS = B["ENC"], B["FPPL"], B["ERR"], B["OCC_BINS"]
HOLIDAYS = set(B["HOLIDAYS"])
T, N = OCC.shape
J = np.arange(1, HORIZON + 1)
FC_IDX = np.minimum((J * STEP) // 60, 11)
STEPS_PER_DAY = 24 * 60 // STEP


# ---------------------------------------------------------------- 예측·확률
def time_feats(ts):
    tod = (ts.hour * 60 + ts.minute) // STEP
    off = int(ts.dayofweek >= 5 or ts.strftime("%Y-%m-%d") in HOLIDAYS)
    ang = 2 * np.pi * tod / STEPS_PER_DAY
    return tod, off, np.sin(ang), np.cos(ang)


def forecast(lots, t_idx):
    """기준 시각 t_idx에서 주차장들의 앞으로 HORIZON칸 점유율 (B, HORIZON)
    미래 시각 특징은 시각으로 직접 계산하므로 데이터 마지막 시각에서도 예측 가능"""
    lots = np.asarray(lots)
    now = TIMES[t_idx]
    fut = [time_feats(now + pd.Timedelta(minutes=STEP * j)) for j in J]
    tod_f = np.array([f[0] for f in fut]); off_f = np.array([f[1] for f in fut])
    sin_f = np.array([f[2] for f in fut], np.float32); cos_f = np.array([f[3] for f in fut], np.float32)
    xe = np.nan_to_num(ENC[t_idx - LOOKBACK + 1: t_idx + 1][:, lots].transpose(1, 0, 2))
    p_now = PROF[t_idx, lots]
    p_fut = PROF_TAB[off_f[None, :], tod_f[None, :], lots[:, None]]
    p_delta = p_fut - p_now[:, None]
    xd = np.stack([p_delta, np.broadcast_to(sin_f, p_delta.shape), np.broadcast_to(cos_f, p_delta.shape),
                   np.broadcast_to(off_f.astype(np.float32), p_delta.shape),
                   FPPL[t_idx, lots][:, FC_IDX]], -1).astype(np.float32)
    occ_now = np.where(np.isnan(OCC[t_idx, lots]), p_now, OCC[t_idx, lots])
    base = np.clip(occ_now[:, None] + p_delta, 0, 1)
    with torch.no_grad():
        corr = MODEL(torch.tensor(xe), torch.tensor(xd), torch.tensor(lots, dtype=torch.long)).numpy()
    return np.clip(base + corr, 0, 1)


def occ_at(n, t_idx, steps, path):
    """steps칸 뒤 점유율. path는 forecast 결과(HORIZON칸). 2시간 넘으면 평소 곡선 + 지금 차이(감쇠)"""
    if steps <= 0:
        v = OCC[t_idx, n]
        return float(PROF[t_idx, n] if np.isnan(v) else v)
    if steps <= HORIZON:
        return float(path[steps - 1])
    ts = TIMES[t_idx] + pd.Timedelta(minutes=STEP * steps)
    tod, off, _, _ = time_feats(ts)
    return float(np.clip(PROF_TAB[off, tod, n] + np.nan_to_num(RESID[t_idx, n]) * 0.5 ** (steps / 24), 0, 1))


def prob_available(pred_occ, cap, step):
    hb = min(max(step, 0) // 6, 3)
    b = int(np.clip(np.digitize(pred_occ, OCC_BINS) - 1, 0, len(OCC_BINS) - 2))
    e = ERR[(hb, b)]
    return float(np.searchsorted(e, (cap - FREE_K) / cap - pred_occ, side="right") / len(e))


def confidence(n, stale_min):
    if stale_min > 30:
        return "낮음 (데이터 지연)"
    m = B["VA_MAE"].get(n, np.nan)
    if m is None or np.isnan(m):
        return "낮음 (기록 부족)"
    return "높음" if m < 0.03 else ("보통" if m < 0.07 else "낮음")


def fee_won(n, minutes):
    p = B["FEES"][n]
    if not p:
        return None
    base, base_m, add, add_m, mx = p
    if base_m <= 0:
        return 0
    fee = base + (max(0, math.ceil((minutes - base_m) / add_m)) * add if add_m > 0 else 0)
    return int(min(fee, mx) if mx > 0 else fee)


# ---------------------------------------------------------------- 길찾기
def haversine_km(lat1, lng1, lat2, lng2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lng2 - lng1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


@st.cache_data(show_spinner=False)
def geocode(text):
    text = text.strip()
    if text in PLACES:
        return PLACES[text]
    try:  # "37.5, 127.0" 형식
        a, b = [float(x) for x in text.split(",")]
        return (a, b)
    except Exception:
        pass
    try:
        import osmnx as ox
        return tuple(ox.geocode(text + ", 서울"))
    except Exception:
        return None


@st.cache_data(show_spinner=False, max_entries=500)
def drive_route(o, d, road_speed):
    """(분, 경로 좌표 [[lat,lng],...], 실제도로여부)"""
    if GRAPH is not None:
        try:
            import osmnx as ox
            k = math.cos(math.radians(37.55))
            a = NODE_IDS[TREE.query([o[0], o[1] * k])[1]]
            b = NODE_IDS[TREE.query([d[0], d[1] * k])[1]]
            sp = getattr(ox, "shortest_path", None) or ox.routing.shortest_path
            route = sp(GRAPH, a, b, weight="travel_time")
            sec = km = 0.0
            for u, v in zip(route[:-1], route[1:]):
                e = min(GRAPH.get_edge_data(u, v).values(), key=lambda x: x.get("travel_time", 1e9))
                sec += float(e["travel_time"]); km += float(e["length"]) / 1000
            free = km / (sec / 3600) if sec > 0 else 30
            factor = float(np.clip(free / road_speed, 1.0, 3.0)) if road_speed and road_speed > 0 else 1.0
            pts = [list(o)] + [[GRAPH.nodes[x]["y"], GRAPH.nodes[x]["x"]] for x in route] + [list(d)]
            return sec / 60 * factor + 2, pts, True
        except Exception:
            pass
    km = float(haversine_km(o[0], o[1], d[0], d[1]))
    spd = road_speed if road_speed and road_speed > 0 else 20
    return km * 1.35 / spd * 60 + 2, [list(o), list(d)], False


def walk_min(a, b):
    return float(haversine_km(a[0], a[1], b[0], b[1])) * 1.25 / 4.5 * 60


def subway_min(a, b):
    return float(haversine_km(a[0], a[1], b[0], b[1])) * 1.25 / 32 * 60 + 8


# ---------------------------------------------------------------- 추천
def recommend(o, d, t_idx, stay_min, near_km, n_near, use_pnr):
    d_dest = haversine_km(LAT, LNG, d[0], d[1])
    d_orig = haversine_km(LAT, LNG, o[0], o[1])
    od = float(haversine_km(o[0], o[1], d[0], d[1]))
    near = d_dest <= near_km
    if near.sum() < n_near:                       # 주변에 적으면 가까운 순으로 채움
        near[np.argsort(d_dest)[:n_near]] = True
    is_station = np.array(["역" in nm for nm in NAMES])
    pnr = (is_station & ~near & (d_orig < od) & (d_dest < od)) if use_pnr else np.zeros(N, bool)
    cand = np.where(near | pnr)[0]
    paths = dict(zip(cand, forecast(cand, t_idx)))
    nearest = int(np.argmin(d_dest))
    rows, routes = [], {}
    for n in cand:
        a = B["LOT_AREA"][n]
        rs = float(B["ROAD_SPD"][t_idx, a]) if a >= 0 and not np.isnan(B["ROAD_SPD"][t_idx, a]) else None
        drive, pts, real = drive_route(tuple(o), (float(LAT[n]), float(LNG[n])), rs)
        steps = int(round(drive / STEP))
        p_occ = occ_at(n, t_idx, steps, paths[n])
        prob = prob_available(p_occ, CAP[n], min(steps, HORIZON) - 1)
        seen = np.where(~np.isnan(OCC[:t_idx + 1, n]))[0]
        stale = (TIMES[t_idx] - TIMES[seen[-1]]).total_seconds() / 60 if len(seen) else 999
        lot = (float(LAT[n]), float(LNG[n]))
        if pnr[n]:
            how, extra = "지하철(추정)", subway_min(lot, d)
        else:
            how, extra = "도보", walk_min(lot, d)
        fee = fee_won(n, stay_min)
        total = drive + extra
        score = prob - 0.5 * total / 60 - 0.05 * (fee or 0) / 10000
        arr = t_idx + steps
        actual = (1 - OCC[arr, n]) * CAP[n] if arr < T and not np.isnan(OCC[arr, n]) else np.nan
        rows.append({
            "n": n, "주차장": NAMES[n] + (" ⭐가장 가까움" if n == nearest else ""),
            "구분": "환승(P&R)" if pnr[n] else "목적지 주변",
            "도착": (TIMES[t_idx] + pd.Timedelta(minutes=drive)).strftime("%H:%M"),
            "지금 빈자리": int(round((1 - np.nan_to_num(OCC[t_idx, n])) * CAP[n])),
            "도착 때 예상 빈자리": int(round((1 - p_occ) * CAP[n])), "총 면수": int(CAP[n]),
            "자리 있을 확률": prob, "신뢰도": confidence(n, stale),
            "자동차(분)": round(drive), "이후 이동": how, "도보/지하철(분)": round(extra),
            "총 소요(분)": round(total), f"요금({stay_min}분)": fee,
            "목적지까지(km)": round(float(d_dest[n]), 2), "점수": score,
            "실제 빈자리(재생)": None if np.isnan(actual) else int(round(actual)),
            "실제 도로": real,
        })
        routes[n] = pts
    df = pd.DataFrame(rows).sort_values("점수", ascending=False).reset_index(drop=True)
    return df, routes, paths


# ---------------------------------------------------------------- 지도 + 이동 애니메이션
def map_html(o, d, df, routes, sel, seconds=10):
    lots = [{"lat": float(LAT[r.n]), "lng": float(LNG[r.n]), "name": r.주차장, "prob": float(r["자리 있을 확률"]),
             "free": int(r["도착 때 예상 빈자리"]), "cap": int(r["총 면수"]), "conf": r.신뢰도,
             "arr": r.도착, "kind": r.구분, "sel": int(r.n) == int(sel)} for _, r in df.iterrows()]
    srow = df[df.n == sel].iloc[0]
    data = {"o": list(o), "d": list(d), "lots": lots, "drive": routes[sel],
            "after": [[float(LAT[sel]), float(LNG[sel])], list(d)],
            "afterKind": srow["이후 이동"], "sec": seconds,
            "label": f"{NAMES[sel]} · {srow['도착']} 도착 · 자리 있을 확률 {srow['자리 있을 확률']:.0%}"}
    return """
<!DOCTYPE html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
 html,body{margin:0;height:100%;font-family:'Malgun Gothic',sans-serif}
 #map{height:520px;border-radius:10px}
 .bar{display:flex;gap:10px;align-items:center;margin:6px 2px;font-size:14px}
 .bar button{padding:6px 14px;border:0;border-radius:8px;background:#2563eb;color:#fff;cursor:pointer}
 .emoji{font-size:26px;line-height:26px;text-align:center}
</style></head><body>
<div class="bar"><button id="play">▶ 다시 재생</button><span id="status"></span></div>
<div id="map"></div>
<script>
const D = __DATA__;
const map = L.map('map');
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:19,attribution:'© OpenStreetMap'}).addTo(map);
const color = p => p>=0.8?'#16a34a':(p>=0.5?'#f59e0b':'#dc2626');
D.lots.forEach(l=>{
  L.circleMarker([l.lat,l.lng],{radius:l.sel?11:7,color:l.sel?'#111':'#fff',weight:l.sel?3:1,
     fillColor:color(l.prob),fillOpacity:.9}).addTo(map)
   .bindPopup(`<b>${l.name}</b><br>${l.kind} · ${l.arr} 도착<br>예상 빈자리 <b>${l.free}</b> / ${l.cap}<br>자리 있을 확률 <b>${Math.round(l.prob*100)}%</b><br>신뢰도 ${l.conf}`);
});
L.marker(D.o,{icon:L.divIcon({className:'emoji',html:'🏠',iconSize:[28,28]})}).addTo(map).bindTooltip('출발');
L.marker(D.d,{icon:L.divIcon({className:'emoji',html:'🏁',iconSize:[28,28]})}).addTo(map).bindTooltip('목적지');
L.polyline(D.drive,{color:'#2563eb',weight:5,opacity:.35}).addTo(map);
L.polyline(D.after,{color:'#6b7280',weight:3,dashArray:'6 8',opacity:.6}).addTo(map);
map.fitBounds(L.latLngBounds(D.drive.concat(D.after).concat(D.lots.map(l=>[l.lat,l.lng]))).pad(0.15));

function cum(path){const c=[0];for(let i=1;i<path.length;i++)c.push(c[i-1]+map.distance(path[i-1],path[i]));return c;}
function at(path,c,t){const target=t*c[c.length-1];let i=1;while(i<c.length-1&&c[i]<target)i++;
  const seg=c[i]-c[i-1]||1, r=(target-c[i-1])/seg;
  return [path[i-1][0]+(path[i][0]-path[i-1][0])*r, path[i-1][1]+(path[i][1]-path[i-1][1])*r];}
const cD=cum(D.drive), cA=cum(D.after);
const car=L.marker(D.o,{icon:L.divIcon({className:'emoji',html:'🚗',iconSize:[28,28]}),zIndexOffset:1000}).addTo(map);
const trail=L.polyline([],{color:'#2563eb',weight:6}).addTo(map);
const st=document.getElementById('status');
let raf=null;
function play(){
  cancelAnimationFrame(raf); trail.setLatLngs([]);
  const total=D.sec*1000, driveMs=total*0.8, start=performance.now();
  const walkIcon=D.afterKind.startsWith('지하철')?'🚇':'🚶';
  car.setIcon(L.divIcon({className:'emoji',html:'🚗',iconSize:[28,28]}));
  function step(now){
    const t=Math.min((now-start)/total,1);
    if(now-start<driveMs){
      const p=at(D.drive,cD,(now-start)/driveMs); car.setLatLng(p); trail.addLatLng(p);
      st.textContent='🚗 주차장으로 이동 중… '+D.label;
    } else {
      car.setIcon(L.divIcon({className:'emoji',html:walkIcon,iconSize:[28,28]}));
      car.setLatLng(at(D.after,cA,(now-start-driveMs)/(total-driveMs)));
      st.textContent=walkIcon+' 주차 후 '+D.afterKind+'로 목적지 이동 중…';
    }
    if(t<1) raf=requestAnimationFrame(step);
    else { car.setIcon(L.divIcon({className:'emoji',html:'✅',iconSize:[28,28]})); st.textContent='✅ 도착! '+D.label; }
  }
  raf=requestAnimationFrame(step);
}
document.getElementById('play').onclick=play;
play();
</script></body></html>""".replace("__DATA__", json.dumps(data, ensure_ascii=False))


# ---------------------------------------------------------------- 화면
st.title("🅿️ 자리있서울")
st.caption("도착할 때 자리가 있을지 미리 알려주는 서울 공영주차장 추천 · 서울 열린데이터광장 실시간 데이터 기반")

with st.sidebar:
    st.header("경로")
    opts = list(PLACES) + ["직접 입력"]
    o_sel = st.selectbox("출발지", opts, index=opts.index("고양 삼송"))
    o_txt = st.text_input("출발지 직접 입력 (장소명 또는 '위도, 경도')", "") if o_sel == "직접 입력" else o_sel
    d_sel = st.selectbox("목적지", opts, index=opts.index("강남역"))
    d_txt = st.text_input("목적지 직접 입력 (장소명 또는 '위도, 경도')", "") if d_sel == "직접 입력" else d_sel

    st.header("시각")
    day_opts = sorted({t.normalize() for t in TIMES}, reverse=True)
    mode = st.radio("기준 시각", ["데이터 최신 시각", "과거 재생 (실제와 비교)"], horizontal=False)
    if mode == "데이터 최신 시각":
        t_idx = T - 1
    else:
        day = st.selectbox("날짜", day_opts, format_func=lambda x: x.strftime("%m/%d (%a)"))
        hhmm = st.slider("출발 시각", 0.0, 23.5, 10.0, 0.5, format="%.1f시")
        want = day + pd.Timedelta(hours=hhmm)
        t_idx = int(np.clip(np.searchsorted(TIMES, want), LOOKBACK, T - 1))

    st.header("옵션")
    stay = st.slider("주차 시간(분)", 30, 480, 120, 30)
    near_km = st.slider("목적지 주변 반경(km)", 0.5, 5.0, 2.0, 0.5)
    use_pnr = st.checkbox("가는 길 환승주차장(P&R)도 추천", True)
    go = st.button("🔍 자리 찾기", type="primary", use_container_width=True)

st.write(f"**기준 시각** {TIMES[t_idx]:%Y-%m-%d %H:%M} · 도로망: {'OpenStreetMap 실제 도로' if GRAPH is not None else '직선 근사 (도로망 파일 없음)'}")

if go or "result" not in st.session_state:
    o, d = geocode(o_txt), geocode(d_txt)
    if o is None or d is None:
        st.error("장소를 찾지 못했습니다. 목록에서 고르거나 '37.5, 127.0' 형식으로 입력하세요.")
        st.stop()
    with st.spinner("주차장별 도착 시각과 빈자리를 예측하는 중..."):
        df, routes, paths = recommend(o, d, t_idx, stay, near_km, 5, use_pnr)
    st.session_state.result = dict(o=o, d=d, df=df, routes=routes, paths=paths, t_idx=t_idx)

R = st.session_state.result
df = R["df"]
if df.empty:
    st.warning("후보 주차장이 없습니다. 반경을 늘려보세요.")
    st.stop()

label = {int(r.n): f"{i+1}위 · {r.주차장} ({r['자리 있을 확률']:.0%})" for i, r in df.iterrows()}
sel = st.selectbox("지도에서 볼 주차장", list(label), format_func=label.get)
row = df[df.n == sel].iloc[0]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("자리 있을 확률", f"{row['자리 있을 확률']:.0%}")
c2.metric("도착 때 예상 빈자리", f"{row['도착 때 예상 빈자리']} / {row['총 면수']}",
          delta=f"{row['도착 때 예상 빈자리'] - row['지금 빈자리']:+d} (지금 대비)")
c3.metric("도착", row["도착"], help=f"자동차 {row['자동차(분)']}분")
c4.metric("총 소요", f"{row['총 소요(분)']}분", help=f"{row['이후 이동']} {row['도보/지하철(분)']}분 포함")
fee_col = [c for c in df.columns if c.startswith("요금")][0]
c5.metric(fee_col, "-" if pd.isna(row[fee_col]) else f"{int(row[fee_col]):,}원", help=f"신뢰도 {row['신뢰도']}")

components.html(map_html(R["o"], R["d"], df, R["routes"], sel), height=580)

left, right = st.columns([3, 2])
with left:
    st.subheader("추천 주차장")
    show = df.drop(columns=["n", "실제 도로"]).copy()
    show["자리 있을 확률"] = (show["자리 있을 확률"] * 100).round().astype(int).astype(str) + "%"
    show["점수"] = show["점수"].round(3)
    if mode == "데이터 최신 시각":
        show = show.drop(columns=["실제 빈자리(재생)"])
    st.dataframe(show, use_container_width=True, hide_index=True)
with right:
    st.subheader("앞으로 2시간 빈자리 예측")
    path = R["paths"][sel]
    tt = [TIMES[R["t_idx"]] + pd.Timedelta(minutes=STEP * j) for j in J]
    chart = pd.DataFrame({"예상 빈자리": (1 - path) * CAP[sel]}, index=pd.Index(tt, name="시각"))
    if mode != "데이터 최신 시각":
        act = [(1 - OCC[R["t_idx"] + j, sel]) * CAP[sel] if R["t_idx"] + j < T else np.nan for j in J]
        chart["실제 빈자리"] = act
    st.line_chart(chart)
    st.caption(f"{NAMES[sel]} · 총 {int(CAP[sel])}면 · 빈자리 {FREE_K}개 이상을 '자리 있음'으로 계산")

with st.expander("계산 방법"):
    st.markdown(
        "- **예상 빈자리**: 지금 차량 수 + 평소 이 시간대 변화량에 seq2seq 모델 보정을 더해 도착 시각 값을 예측\n"
        "- **자리 있을 확률**: 검증 기간의 예측 오차 분포로 빈자리가 3개 이상일 확률을 계산\n"
        "- **신뢰도**: 주차장별 과거 예측 오차와 데이터 신선도\n"
        "- **이동 시간**: OpenStreetMap 도로망 최단경로 × 도시데이터 실시간 도로 속도 보정. 지하철 시간은 거리 기반 추정\n"
        "- **점수**: 확률 − 0.5×(총 소요/60분) − 0.05×(요금/1만원)"
    )
