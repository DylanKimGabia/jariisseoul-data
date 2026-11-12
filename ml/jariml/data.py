"""수집 데이터 읽기. 로컬 data 폴더가 없으면 GitHub 저장소에서 받는다."""
import io
import shutil
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd


def resolve_data_dir(cfg: dict, refresh: bool = False) -> Path:
    """data 폴더 경로를 돌려준다. 저장소 안에서 실행하면 ../data를, 아니면 GitHub zip을 받아 쓴다."""
    local = Path(cfg["data"]["data_dir"])
    if (local / "live").exists() and not refresh:
        return local
    cache = Path(cfg["data"]["data_cache"])
    if refresh or not (cache / "data" / "live").exists():
        shutil.rmtree(cache, ignore_errors=True)
        z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(cfg["data"]["github_zip"]).read()))
        for m in z.namelist():
            rel = m.split("/", 1)[1]
            if rel.startswith("data/") and not m.endswith("/"):
                out = cache / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(z.read(m))
    return cache / "data"


def load_live(data_dir: Path, files: list | None = None) -> pd.DataFrame:
    """시영주차장 실시간 기록. files를 주면 그 파일만 읽는다 (예측 시 최근 파일만)."""
    files = files or sorted(Path(data_dir, "live").glob("*.csv"))
    live = pd.concat([pd.read_csv(f, dtype={"PKLT_CD": str}) for f in files], ignore_index=True)
    live["t"] = pd.to_datetime(live.collected_at).dt.floor("5min")
    return live.drop_duplicates(["PKLT_CD", "t"], keep="last")   # 같은 5분 칸에 두 번 수집되면 마지막 값


def load_static(data_dir: Path) -> pd.DataFrame:
    """주차장 상세 정보 (최신일)."""
    f = sorted(Path(data_dir, "static").glob("*.csv"))[-1]
    return pd.read_csv(f, dtype=str).set_index("PKLT_CD")


def load_reference(data_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    """주차장 좌표, 환승 감면율."""
    ref = Path(data_dir, "reference")
    coord = pd.read_csv(ref / "parking_lots.csv", dtype={"PKLT_CD": str}).set_index("PKLT_CD")
    pnr = pd.read_csv(ref / "pnr_lots.csv", dtype={"PKLT_CD": str}).set_index("PKLT_CD").discount_rate
    return coord, pnr
