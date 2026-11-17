"""딥러닝 엔진: GRU Seq2Seq (PyTorch).

입력: 주차장마다 최근 2시간(24칸)의 점유율 시계열 + 평소 곡선 + 시각 / 휴일 정보 + 주차장 임베딩
출력: 다음 24칸(5분~2시간) 점유율을 한 번에 예측 (지금 대비 변화량)

표 형태 ML 엔진과 달리 직접 만든 변화량 입력 없이 시계열을 통째로 본다.
"""
import copy
import time

import numpy as np
import pandas as pd

from ..features import fit_profile
from ..preprocess import is_offday
from .base import Engine

H = 24          # 예측 칸 수 (2시간)

try:                                    # torch가 없으면 gru 엔진만 못 쓰고 나머지는 그대로 동작
    import torch
    import torch.nn as nn

    class GRUNet(nn.Module):
        """GRU 인코더(최근 2시간) + 주차장 임베딩 + 앞으로 2시간 평소 곡선 → 24칸 변화량."""
        def __init__(self, n_lots, hidden, emb):
            super().__init__()
            self.emb = nn.Embedding(n_lots, emb)
            self.gru = nn.GRU(5 + emb, hidden, batch_first=True)
            self.head = nn.Sequential(nn.Linear(hidden + emb + 2 * H, 128), nn.ReLU(), nn.Linear(128, H))

        def forward(self, win, fut, lot):
            e = self.emb(lot)
            _, h = self.gru(torch.cat([win, e[:, None, :].expand(-1, win.shape[1], -1)], -1))
            return self.head(torch.cat([h[-1], e, fut], -1))
except ImportError:
    torch = None


def _time_feats(idx, holidays):
    tod = (idx.hour * 12 + idx.minute // 5).values
    ang = 2 * np.pi * tod / 288
    off = is_offday(idx, holidays).values.astype(np.float32)
    return tod, np.sin(ang).astype(np.float32), np.cos(ang).astype(np.float32), off


class GRUEngine(Engine):
    name = "gru"
    history_steps = 24               # 최근 2시간

    def params(self) -> dict:
        p = {"hidden": 64, "emb": 8, "epochs": 15, "patience": 3, "batch": 512, "lr": 2e-3, "stride": 1, "seed": 0}
        p.update(self.cfg.get("engine_params", {}).get(self.name, {}))
        return p

    # ---------- 입력 텐서 ----------
    def _arrays(self, occ):
        """시각 × 주차장 → 시간별 입력 특징 (T, N, 5): 점유율, 평소 곡선, sin, cos, 휴일."""
        hol = self.cfg["preprocess"]["holidays"]
        lots = list(occ.columns)
        col = pd.Index(self.profile["lots"]).get_indexer(lots)
        tod, s, c, off = _time_feats(occ.index, hol)
        prof = self.profile["values"][off.astype(int), tod][:, col]           # (T, N)
        O = occ.ffill(limit=6).to_numpy(dtype=np.float32)
        O = np.where(np.isnan(O), prof, O)                                     # 긴 공백은 평소 값으로
        T, N = O.shape
        X = np.stack([O, prof, np.repeat(s[:, None], N, 1), np.repeat(c[:, None], N, 1),
                      np.repeat(off[:, None], N, 1)], axis=-1).astype(np.float32)
        return X, occ.to_numpy(dtype=np.float32), prof

    def _future(self, idx, pos, lots_col):
        """기준 시각 이후 24칸의 평소 곡선과 휴일 여부 (디코더 입력)."""
        hol = self.cfg["preprocess"]["holidays"]
        out_p, out_o = [], []
        for p in pos:
            tf = idx[p] + pd.to_timedelta(np.arange(1, H + 1) * 5, unit="min")
            tod, _, _, off = _time_feats(tf, hol)
            out_p.append(self.profile["values"][off.astype(int), tod][:, lots_col])   # (H, N)
            out_o.append(off)
        return np.stack(out_p), np.stack(out_o)                                       # (P, H, N), (P, H)

    def _samples(self, occ, origins, with_target):
        lots = list(occ.columns)
        X, Oraw, _ = self._arrays(occ)
        idx = occ.index
        pos = np.arange(len(idx)) if origins is None else idx.get_indexer(origins)
        pos = pos[(pos >= self.history_steps - 1)]
        if with_target:
            pos = pos[pos + 1 < len(idx)]
        col = pd.Index(self.profile["lots"]).get_indexer(lots)
        fp, fo = self._future(idx, pos, col)
        N = len(lots)
        win = np.stack([X[p - self.history_steps + 1: p + 1] for p in pos])           # (P, L, N, F)
        win = win.transpose(0, 2, 1, 3).reshape(-1, self.history_steps, X.shape[-1])   # (P*N, L, F)
        fut = np.concatenate([fp.transpose(0, 2, 1), np.repeat(fo[:, None, :], N, 1)], -1).reshape(-1, 2 * H)
        lot = np.tile(np.arange(N), len(pos))
        now = Oraw[pos].reshape(-1)
        keep = ~np.isnan(now)
        d = {"win": torch.from_numpy(win[keep]), "fut": torch.from_numpy(fut[keep].astype(np.float32)),
             "lot": torch.from_numpy(lot[keep]), "now": torch.from_numpy(now[keep]),
             "t": np.repeat(idx[pos].values, N)[keep], "code": np.tile(lots, len(pos))[keep]}
        if with_target:
            Y = np.full((len(pos), H, N), np.nan, np.float32)
            for k in range(H):
                ok = pos + k + 1 < len(idx)
                Y[ok, k] = Oraw[pos[ok] + k + 1]
            d["y"] = torch.from_numpy(Y.transpose(0, 2, 1).reshape(-1, H)[keep])
        return d

    # ---------- 모델 ----------
    def _net(self, n_lots):
        if torch is None:
            raise ImportError("gru 엔진은 PyTorch가 필요하다: pip install torch")
        p = self.params()
        return GRUNet(n_lots, p["hidden"], p["emb"])

    def _train(self, tr, va, epochs, patience):
        p = self.params()
        torch.manual_seed(p["seed"])
        net = self._net(self.n_lots)
        opt = torch.optim.Adam(net.parameters(), lr=p["lr"])
        n = len(tr["now"])

        def val_mae(model):
            model.eval()
            with torch.no_grad():
                pr = torch.cat([model(va["win"][i:i + 4096], va["fut"][i:i + 4096], va["lot"][i:i + 4096])
                                for i in range(0, len(va["now"]), 4096)])
            pred = (va["now"][:, None] + pr).clamp(0, 1)
            m = ~torch.isnan(va["y"])
            return float((pred - torch.nan_to_num(va["y"])).abs()[m].mean())

        best, best_state, best_ep, curve, bad = 1e9, None, 0, [], 0
        for ep in range(epochs):
            net.train()
            perm = torch.randperm(n)
            for i in range(0, n, p["batch"]):
                b = perm[i:i + p["batch"]]
                y = tr["y"][b]
                m = ~torch.isnan(y)
                pred = tr["now"][b][:, None] + net(tr["win"][b], tr["fut"][b], tr["lot"][b])
                loss = ((pred - torch.nan_to_num(y)) ** 2)[m].mean()
                opt.zero_grad(); loss.backward(); opt.step()
            if va is None:
                continue
            v = val_mae(net)
            curve.append(v)
            if v < best:
                best, best_state, best_ep, bad = v, copy.deepcopy(net.state_dict()), ep + 1, 0
            else:
                bad += 1
                if bad >= patience:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        return net, best_ep, curve

    # ---------- 엔진 규약 ----------
    def fit(self, occ, cap, split):
        hol, p = self.cfg["preprocess"]["holidays"], self.params()
        self.profile = fit_profile(occ[occ.index < split.val_start], hol)
        self.lots, self.n_lots = list(occ.columns), occ.shape[1]
        t0 = time.time()
        tr = self._samples(occ, split.train[::p["stride"]], True)
        va = self._samples(occ, split.val[::3], True)
        self.net, best_ep, curve = self._train(tr, va, p["epochs"], p["patience"])
        self.info = {"best_iter": best_ep, "train_rows": int(len(tr["now"])), "val_rows": int(len(va["now"])),
                     "val_curve_mae": curve, "fit_sec": round(time.time() - t0, 1)}
        return self.info

    def predict(self, occ, cap, origins, horizons):
        if list(occ.columns) != self.lots:
            occ = occ.reindex(columns=self.lots)
        d = self._samples(occ, origins, False)
        with torch.no_grad():
            pr = torch.cat([self.net(d["win"][i:i + 8192], d["fut"][i:i + 8192], d["lot"][i:i + 8192])
                            for i in range(0, len(d["now"]), 8192)]) if len(d["now"]) else torch.zeros(0, H)
        pred = (d["now"][:, None] + pr).clamp(0, 1).numpy()
        hs = np.asarray(horizons)
        out = pd.DataFrame({"t": np.repeat(d["t"], len(hs)), "PKLT_CD": np.repeat(d["code"], len(hs)),
                            "h": np.tile(hs, len(d["now"])), "pred": pred[:, hs - 1].reshape(-1)})
        return out

    def fit_full(self, occ, cap):
        p = self.params()
        self.profile = fit_profile(occ, self.cfg["preprocess"]["holidays"])
        full = self._samples(occ, occ.index[::p["stride"]], True)
        self.net, _, _ = self._train(full, None, max(1, self.info["best_iter"]), 0)
        self.info["full_rows"] = int(len(full["now"]))
