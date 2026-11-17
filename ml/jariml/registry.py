"""모델 저장과 운영 버전 관리.

models/
  LATEST                      운영 중인 버전 이름 (한 줄)
  20261113-0230/
    artifact.joblib           모델, 평소 곡선, 오차 분포, 주차장 정보
    metrics.json              평가 결과와 학습 정보
"""
import json
import pathlib
import sys
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


def promote_if_better(new_dir: Path, root: Path, metric: str, current_same_test: float | None = None) -> tuple[bool, str]:
    """새 모델이 운영 모델보다 나을 때만 LATEST를 바꾼다 (자동 재학습용).

    current_same_test: 운영 모델을 새 모델과 같은 평가 기간에서 다시 채점한 값 (train.current_on_same_test).
    None이면 같은 기간 비교가 불가능한 경우라서, 더 최근 데이터로 학습한 새 모델로 바꾼다.
    """
    new = json.loads((new_dir / "metrics.json").read_text(encoding="utf-8"))["summary"][metric]
    cur_dir = latest_dir(root)
    if cur_dir is None:
        msg = f"첫 운영 모델: {metric}={new:.3f}"
    elif current_same_test is None:
        msg = f"교체 (같은 기간 비교 불가, 최신 데이터 우선): 새 모델 {metric}={new:.3f}"
    elif new >= current_same_test:
        return False, f"유지: 같은 평가 기간에서 새 모델 {new:.3f} >= 운영 {current_same_test:.3f} ({cur_dir.name})"
    else:
        msg = f"교체: 같은 평가 기간에서 운영 {current_same_test:.3f} -> 새 모델 {new:.3f}"
    (Path(root) / "LATEST").write_text(new_dir.name)
    return True, msg


def _load_any_os(f):
    """다른 OS에서 저장한 파일 열기. [주] 리눅스에서 저장한 PosixPath는 윈도우에서 만들 수 없어 오류가 난다.
    열 때만 잠시 '순수 경로' 클래스로 바꿔 읽는다 (경로 값은 예측에 쓰지 않는다)."""
    swap = ("PosixPath", pathlib.PurePosixPath) if sys.platform == "win32" else ("WindowsPath", pathlib.PureWindowsPath)
    mods = [pathlib] + ([sys.modules["pathlib._local"]] if "pathlib._local" in sys.modules else [])
    old = [getattr(m, swap[0]) for m in mods]
    try:
        for m in mods:
            setattr(m, swap[0], swap[1])
        return joblib.load(f)
    finally:
        for m, o in zip(mods, old):
            setattr(m, swap[0], o)


def load(root: Path, version: str | None = None) -> tuple[dict, dict]:
    d = Path(root) / version if version else latest_dir(root)
    if d is None:
        raise FileNotFoundError(f"{root}에 모델이 없다. 먼저 python -m jariml.train 을 실행한다.")
    return _load_any_os(d / "artifact.joblib"), json.loads((d / "metrics.json").read_text(encoding="utf-8"))
