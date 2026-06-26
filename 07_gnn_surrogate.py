"""Stage 6-7: GNN surrogate for the MCMC distance-to-frontier (dtf).

Stage 6 (train): learn a graph-level regressor  G(tract) -> dtf  from the 1000
MCMC-labelled tracts (data/outputs/sampler_spec/summary.json), so the expensive
RJ-MCMC search (~mins/tract) is replaced by a forward pass (~ms/tract).

Stage 7 (predict): score every extracted tract (data/graphs/*.graphml) with the
trained surrogate -> data/outputs/gnn_dtf_predictions.parquet, completing the
~84k national set that full MCMC could never reach (84k x mins = months).

Node features (topology only, so inference needs no precomputed metrics):
    [degree, is_intersection(deg>=3), is_deadend(deg==1), x_norm, y_norm]
Edge features: [length_norm]  (straight-line, per-graph normalized)
Target: dtf, standardized for training (small, right-skewed: median ~0.011).

Model: 3x GraphSAGE + global mean|max pool -> MLP -> scalar.

Usage:
    python 07_gnn_surrogate.py train   [--epochs 300 --hidden 64 --seed 1]
    python 07_gnn_surrogate.py predict [--limit N]

Honest caveat: the 1000 dtf labels come from an under-converged MCMC (R-hat
median ~1.6), so they are noisy; the surrogate's attainable R^2 is bounded by
label quality. Report rank correlation alongside R^2.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

from uoi_common import GRAPH_DIR, OUT_DIR, ROOT

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool, global_max_pool

SAMP = OUT_DIR / "sampler_spec"
MODEL_PATH = OUT_DIR / "gnn_surrogate.pt"
PRED_PATH = OUT_DIR / "gnn_dtf_predictions.parquet"
RES = ROOT / "results" / "gnn"
RES.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# the 6 design-doc UOI metrics, added as graph-level features (dtf is largely a
# function of how much head-room each metric has vs its recommended bound)
METRICS = ["link_node_ratio", "connected_node_ratio", "intersection_density",
           "median_block_length_ft", "walking_circuity", "pedshed_reach"]
N_GFEAT = len(METRICS)


def load_metric_features() -> dict:
    """GEOID -> np.array(6) of the spec metrics (raw; standardized at train)."""
    m = pd.read_parquet(OUT_DIR / "uoi_spec_metrics.parquet")
    m = m[m["status"] == "ok"].copy()
    m["GEOID"] = m["GEOID"].astype(str).str.zfill(11)
    arr = m[METRICS].to_numpy(dtype=np.float32)
    return {g: arr[i] for i, g in enumerate(m["GEOID"].values)}


# ----------------------------------------------------------- graph -> Data
def graph_to_data(geoid: str, gfeat: np.ndarray | None = None,
                  y: float | None = None) -> Data | None:
    """Build a PyG Data from a tract graphml. Uses nx.read_graphml directly
    (much faster than osmnx's typed loader — matters for the 84k inference)."""
    p = GRAPH_DIR / f"{geoid}.graphml"
    if not p.exists():
        return None
    try:
        G = nx.Graph(nx.read_graphml(p))      # collapse multi/dir -> simple undirected
    except Exception:
        return None
    nodes = list(G.nodes)
    if len(nodes) < 3:
        return None
    idx = {u: i for i, u in enumerate(nodes)}
    deg = np.array([G.degree(u) for u in nodes], dtype=np.float32)

    def _f(u, k):
        try:
            return float(G.nodes[u].get(k, 0.0))
        except (TypeError, ValueError):
            return 0.0
    x = np.array([_f(u, "x") for u in nodes], dtype=np.float32)
    yv = np.array([_f(u, "y") for u in nodes], dtype=np.float32)
    xn = (x - x.mean()) / (x.std() + 1e-6)
    yn = (yv - yv.mean()) / (yv.std() + 1e-6)
    feat = np.stack([np.log1p(deg), (deg >= 3).astype(np.float32),
                     (deg == 1).astype(np.float32), xn, yn], axis=1)

    src, dst, elen = [], [], []
    for u, v, d in G.edges(data=True):
        try:
            L = float(d.get("length", 0.0))
        except (TypeError, ValueError):
            L = 0.0
        src += [idx[u], idx[v]]; dst += [idx[v], idx[u]]; elen += [L, L]
    elen = np.array(elen, dtype=np.float32)
    elen = elen / (elen.mean() + 1e-6)

    data = Data(
        x=torch.tensor(feat),
        edge_index=torch.tensor([src, dst], dtype=torch.long),
        edge_attr=torch.tensor(elen).unsqueeze(1),
    )
    g = np.zeros(N_GFEAT, np.float32) if gfeat is None else np.nan_to_num(gfeat)
    data.gfeat = torch.tensor(g, dtype=torch.float32).unsqueeze(0)
    data.geoid = geoid
    if y is not None:
        data.y = torch.tensor([y], dtype=torch.float32)
    return data


def _build_one(arg):
    """Picklable worker for parallel inference graph-building."""
    geoid, gfeat = arg
    return graph_to_data(geoid, gfeat)


# ----------------------------------------------------------------- model
class GNNSurrogate(nn.Module):
    def __init__(self, in_dim=5, hidden=64, n_gfeat=N_GFEAT):
        super().__init__()
        self.c1 = SAGEConv(in_dim, hidden)
        self.c2 = SAGEConv(hidden, hidden)
        self.c3 = SAGEConv(hidden, hidden)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden + n_gfeat, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1))

    def forward(self, d):
        x, ei, b = d.x, d.edge_index, d.batch
        x = F.relu(self.c1(x, ei))
        x = F.relu(self.c2(x, ei))
        x = F.relu(self.c3(x, ei))
        x = torch.cat([global_mean_pool(x, b), global_max_pool(x, b),
                       d.gfeat], dim=1)               # graph emb + 6 UOI metrics
        return self.head(x).squeeze(-1)


def r2(pred, true):
    ss_res = ((true - pred) ** 2).sum()
    ss_tot = ((true - true.mean()) ** 2).sum()
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def spearman(pred, true):
    pr = pd.Series(pred).rank().values
    tr = pd.Series(true).rank().values
    return float(np.corrcoef(pr, tr)[0, 1])


# ------------------------------------------------------------------ train
def train(args):
    summ = json.loads((SAMP / "summary.json").read_text())
    labels = {g: v["distance_to_frontier"] for g, v in summ.items()}
    mfeat = load_metric_features()
    print(f"{len(labels)} labelled tracts; building graphs on {DEVICE} ...", flush=True)

    t0 = time.time()
    data_list = []
    for g, y in labels.items():
        d = graph_to_data(g, mfeat.get(g), y)
        if d is not None:
            data_list.append(d)
    print(f"built {len(data_list)} graphs in {time.time()-t0:.0f}s", flush=True)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(data_list))
    n = len(perm); n_te = int(0.15 * n); n_va = int(0.15 * n)
    te, va, tr = perm[:n_te], perm[n_te:n_te + n_va], perm[n_te + n_va:]
    for d in data_list:                       # log target (dtf is >=0, heavy right skew)
        d.y = torch.log1p(d.y)
    ys = np.array([float(data_list[i].y) for i in tr])
    y_mean, y_std = ys.mean(), ys.std() + 1e-9
    gf = np.vstack([data_list[i].gfeat.numpy() for i in tr])
    g_mean = gf.mean(0); g_std = gf.std(0) + 1e-9
    for d in data_list:                       # standardize target + global feats
        d.y = (d.y - y_mean) / y_std
        d.gfeat = (d.gfeat - torch.tensor(g_mean)) / torch.tensor(g_std)

    def loader(ix, shuffle):
        return DataLoader([data_list[i] for i in ix], batch_size=args.batch,
                          shuffle=shuffle)
    tl, vl, tel = loader(tr, True), loader(va, False), loader(te, False)

    model = GNNSurrogate(hidden=args.hidden).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=15, factor=0.5)

    def evaluate(ld):
        model.eval(); P, T = [], []
        with torch.no_grad():
            for b in ld:
                b = b.to(DEVICE)
                P.append(model(b).cpu().numpy()); T.append(b.y.cpu().numpy())
        P, T = np.concatenate(P), np.concatenate(T)
        return P, T

    best_va, best_state, patience = -1e9, None, 0
    for ep in range(1, args.epochs + 1):
        model.train()
        for b in tl:
            b = b.to(DEVICE); opt.zero_grad()
            loss = F.mse_loss(model(b), b.y)
            loss.backward(); opt.step()
        P, T = evaluate(vl); vr2 = r2(P, T); sched.step(((T - P) ** 2).mean())
        if vr2 > best_va:
            best_va, best_state, patience = vr2, {k: v.cpu().clone()
                                                  for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
        if ep % 20 == 0 or ep == 1:
            print(f"  ep{ep:3d} val R2={vr2:.3f} (best {best_va:.3f})", flush=True)
        if patience >= args.early:
            print(f"  early stop @ep{ep}", flush=True); break

    model.load_state_dict(best_state)
    # report on the held-out test set, in ORIGINAL dtf units
    Pte, Tte = evaluate(tel)
    Pd, Td = np.expm1(Pte * y_std + y_mean), np.expm1(Tte * y_std + y_mean)
    print(f"\nTEST  R2(dtf)={r2(Pd,Td):.3f}  R2(log)={r2(Pte,Tte):.3f}  "
          f"Spearman={spearman(Pd,Td):.3f}  MAE={np.abs(Pd-Td).mean():.4f}  "
          f"(n={len(Td)})", flush=True)

    torch.save({"state": model.state_dict(), "hidden": args.hidden,
                "y_mean": y_mean, "y_std": y_std,
                "g_mean": g_mean, "g_std": g_std}, MODEL_PATH)
    print(f"saved model -> {MODEL_PATH}")

    # pred-vs-true scatter
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 5.5), facecolor="white")
    ax.scatter(Td, Pd, s=14, alpha=0.6)
    lim = [min(Td.min(), Pd.min()), max(Td.max(), Pd.max())]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xlabel("true dtf (MCMC)"); ax.set_ylabel("predicted dtf (GNN)")
    ax.set_title(f"GNN surrogate — test R²={r2(Pte,Tte):.3f}, "
                 f"Spearman={spearman(Pte,Tte):.3f}")
    fig.tight_layout(); fig.savefig(RES / "fig_pred_vs_true.png", dpi=140)
    print(f"saved {RES/'fig_pred_vs_true.png'}")


# ---------------------------------------------------------------- predict
def predict(args):
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    model = GNNSurrogate(hidden=ckpt["hidden"]).to(DEVICE); model.eval()
    model.load_state_dict(ckpt["state"])
    y_mean, y_std = ckpt["y_mean"], ckpt["y_std"]
    g_mean = torch.tensor(ckpt["g_mean"]); g_std = torch.tensor(ckpt["g_std"])
    mfeat = load_metric_features()

    geoids = sorted(p.stem for p in GRAPH_DIR.glob("*.graphml"))
    if args.limit:
        geoids = geoids[: args.limit]
    print(f"predicting dtf for {len(geoids)} tracts on {DEVICE}, "
          f"{args.procs} loader procs ...", flush=True)

    rows, t0 = [], time.time()

    # the surrogate is only trained on the top-1000 (elite urban grids); for
    # tracts far outside that distribution the prediction is extrapolation, so
    # we flag them (max |standardized global feature| > OOD_Z) and clamp.
    OOD_Z, DTF_CAP = 4.0, 0.5

    def infer(data_list, gids):
        if not data_list:
            return
        zmax = []
        for d in data_list:
            z = (d.gfeat - g_mean) / g_std
            zmax.append(float(z.abs().max()))
            d.gfeat = z
        b = next(iter(DataLoader(data_list, batch_size=len(data_list)))).to(DEVICE)
        with torch.no_grad():
            p = np.expm1(model(b).cpu().numpy() * y_std + y_mean)
        for g, v, zz in zip(gids, p, zmax):
            rows.append((g, float(np.clip(v, 0.0, DTF_CAP)), zz > OOD_Z, zz))

    # build graphs in parallel (CPU-bound XML parse), infer on GPU per chunk
    from concurrent.futures import ProcessPoolExecutor
    CH = 4000
    with ProcessPoolExecutor(max_workers=args.procs) as ex:
        for c0 in range(0, len(geoids), CH):
            chunk = geoids[c0:c0 + CH]
            built = list(ex.map(_build_one,
                                [(g, mfeat.get(g)) for g in chunk], chunksize=64))
            dl = [d for d in built if d is not None]
            gd = [d.geoid for d in dl]
            infer(dl, gd)
            print(f"  {min(c0+CH,len(geoids))}/{len(geoids)} "
                  f"({len(rows)/(time.time()-t0):.0f}/s)", flush=True)

    df = pd.DataFrame(rows, columns=["GEOID", "dtf_pred", "ood", "z_max"])
    df["state"] = df.GEOID.str[:2]
    df.to_parquet(PRED_PATH, index=False)
    df.to_csv(OUT_DIR / "gnn_dtf_predictions.csv", index=False)
    n_ood = int(df.ood.sum())
    print(f"\nsaved {len(df)} predictions -> {PRED_PATH}")
    print(f"in-distribution: {len(df)-n_ood}   out-of-distribution (unreliable): "
          f"{n_ood} ({n_ood/len(df):.0%})")
    print(df.loc[~df.ood, "dtf_pred"].describe().round(4).to_string())


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--epochs", type=int, default=300)
    tr.add_argument("--hidden", type=int, default=64)
    tr.add_argument("--batch", type=int, default=32)
    tr.add_argument("--lr", type=float, default=3e-3)
    tr.add_argument("--early", type=int, default=40)
    tr.add_argument("--seed", type=int, default=1)
    pr = sub.add_parser("predict")
    pr.add_argument("--limit", type=int, default=None)
    pr.add_argument("--procs", type=int, default=16)
    args = ap.parse_args()
    (train if args.cmd == "train" else predict)(args)


if __name__ == "__main__":
    main()
