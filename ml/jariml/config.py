"""설정 파일(config.yaml) 읽기. 경로는 ml/ 폴더 기준으로 바꾼다."""
from pathlib import Path
import yaml

ML_ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None) -> dict:
    path = Path(path) if path else ML_ROOT / "config.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    for section, key in (("data", "data_dir"), ("data", "data_cache"), ("registry", "dir")):
        p = Path(cfg[section][key])
        cfg[section][key] = p if p.is_absolute() else (ML_ROOT / p).resolve()
    return cfg
