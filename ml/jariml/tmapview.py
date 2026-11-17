"""추천 결과를 TMAP 지도(Web JS API v2)로 그린다.

    출발지(검은 점), 목적지(빨간 별), 추천 주차장(순위 숫자, 색 = 확률 등급, 파란 테두리 = 환승주차장)
    선: 출발지 → 목적지 자동차(파랑), 주차장 → 목적지 도보(초록 점선), 대중교통(보라). 실제 경로가 없으면 회색 점선 직선.

[주1] 지도 HTML 파일에 TMAP 키가 들어가므로 GitHub에 올리지 않는다 (.gitignore: *_map.html).
"""
import json
from pathlib import Path

COLOR = {"높음": "#2e7d32", "보통": "#ef6c00", "낮음": "#c62828"}
LINE = {"자동차": ("#1565c0", 6, "solid"), "도보": ("#2e7d32", 4, "dot"), "대중교통": ("#6a1b9a", 5, "solid"),
        "추정": ("#9e9e9e", 3, "dash")}


def tmap_html(result: dict, key: str, height: int = 560) -> str:
    t, m = result["table"], result["meta"]
    o, d, P = m["origin"], m["dest"], m["paths"]
    lines = [{"style": "자동차", "path": P["car"]}]
    marks = []
    for _, x in t.iterrows():
        p = P["lots"].get(x.PKLT_CD)
        lines.append({"style": p["mode"], "path": p["path"]} if p and len(p["path"]) > 1
                     else {"style": "추정", "path": [[x.lat, x.lon], [d["lat"], d["lon"]]]})
        marks.append({"lat": x.lat, "lon": x.lon, "rank": int(x["rank"]), "color": COLOR[x.grade], "pnr": bool(x.is_pnr),
                      "tip": f"{x['rank']}. {x['name']} | 확률 {x.prob}% | 빈자리 {x.pred_free:.0f}대 | {x.last_leg} {x.last_min}분"})
    data = json.dumps({"o": o, "d": d, "lines": lines, "marks": marks, "style": LINE}, ensure_ascii=False)
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<script src="https://apis.openapi.sk.com/tmap/jsv2?version=1&appKey={key}"></script>
<style>body{{margin:0;font-family:sans-serif}} #map{{width:100%;height:{height}px}}
.pin{{color:#fff;border-radius:50%;width:26px;height:26px;line-height:22px;text-align:center;font-weight:bold;font-size:13px;box-sizing:border-box}}</style>
</head><body><div id="map"></div><script>
var D = {data};
var LL = function(p) {{ return new Tmapv2.LatLng(p[0], p[1]); }};
var map = new Tmapv2.Map("map", {{center: LL([D.d.lat, D.d.lon]), width: "100%", height: "{height}px", zoom: 14}});
var b = new Tmapv2.LatLngBounds();
D.lines.forEach(function(l) {{
  var s = D.style[l.style];
  var path = l.path.map(LL); path.forEach(function(p) {{ b.extend(p); }});
  new Tmapv2.Polyline({{path: path, strokeColor: s[0], strokeWeight: s[1], strokeStyle: s[2], map: map}});
}});
new Tmapv2.Marker({{position: LL([D.o.lat, D.o.lon]), map: map, title: "출발지: " + D.o.name,
  iconHTML: '<div class="pin" style="background:#212121;width:14px;height:14px"></div>'}});
new Tmapv2.Marker({{position: LL([D.d.lat, D.d.lon]), map: map, title: "목적지: " + D.d.name,
  iconHTML: '<div style="font-size:28px;color:#c62828">&#9733;</div>'}});
D.marks.forEach(function(k) {{
  var border = k.pnr ? "3px solid #1565c0" : "2px solid #fff";
  new Tmapv2.Marker({{position: LL([k.lat, k.lon]), map: map, title: k.tip,
    iconHTML: '<div class="pin" title="' + k.tip + '" style="background:' + k.color + ';border:' + border + '">' + k.rank + '</div>'}});
  b.extend(LL([k.lat, k.lon]));
}});
b.extend(LL([D.o.lat, D.o.lon])); b.extend(LL([D.d.lat, D.d.lon]));
map.fitBounds(b);
</script></body></html>"""


def show(result: dict, key: str, out="recommend_map.html", height: int = 560):
    """지도 HTML을 노트북 폴더에 저장하고 노트북 안에 띄운다. 안 보이면 저장된 파일을 브라우저로 직접 연다."""
    from IPython.display import IFrame, display, HTML
    out = Path(out).resolve()
    out.write_text(tmap_html(result, key, height), encoding="utf-8")
    try:                                                    # 주피터 서버 기준 경로 (/files/...)
        rel = out.relative_to(Path.home()).as_posix()
        display(HTML(f'<a href="/files/{rel}" target="_blank">새 탭에서 지도 열기</a>'))
        return IFrame(f"/files/{rel}", width="100%", height=height + 20)
    except ValueError:
        print("지도 파일:", out)
