"""모델 저장과 운영 버전 관리.

models/
  LATEST                      운영 중인 버전 이름 (한 줄)
  20261113-0230/
    artifact.joblib           모델, 평소 곡선, 오차 분포, 주차장 정보
    metrics.json              평가 결과와 학습 정보
"""
import json
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

import joblib


def save(artifact: dict, metrics: dict, root: Path) -> Path:
    root = Path(root)
    version = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d-%H%M")     # 한국 시각
    out = root / version
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, out / "artifact.joblib", compress=3)
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return out


def latest_dir(root: Path) -> Path | None:
    f = Path(root) / "LATEST"
    return Path(root) / f.read_text().strip() if f.exists() else None


def promote_if_better(new_dir: Path, root: Path, metric: str) -> tuple[bool, str]:
    """새 모델의 평가 지표가 운영 모델보다 작을 때만 LATEST를 바꾼다 (자동 재학습용)."""
    new = json.loads((new_dir / "metrics.json").read_text(encoding="utf-8"))["summary"][metric]
    cur_dir = latest_dir(root)
    if cur_dir and (cur_dir / "metrics.json").exists():
        cur = json.loads((cur_dir / "metrics.json").read_text(encoding="utf-8"))["summary"][metric]
        if new >= cur:
            return False, f"유지: 새 모델 {metric}={new:.3f} >= 운영 {cur:.3f} ({cur_dir.name})"
        msg = f"교체: {metric} {cur:.3f} -> {new:.3f}"
    else:
        msg = f"첫 운영 모델: {metric}={new:.3f}"
    (Path(root) / "LATEST").write_text(new_dir.name)
    return True, msg


def load(root: Path, version: str | None = None) -> tuple[dict, dict]:
    d = Path(root) / version if version else latest_dir(root)
    if d is None:
        raise FileNotFoundError(f"{root}에 모델이 없다. 먼저 python -m jariml.train 을 실행한다.")
    return joblib.load(d / "artifact.joblib"), json.loads((d / "metrics.json").read_text(encoding="utf-8"))
