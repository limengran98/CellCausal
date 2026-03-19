# Generic Reference Recipe (DD-MIA)
# Deterministic starter notebook for CellCausal agent-mode.

# ---- cell ----
print("DIGEST: Executing generic DD-MIA reference recipe.")

# ---- cell ----
import os
import json
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except Exception:
    Chem = None
    AllChem = None

STAGE1_H5_PATH = os.environ.get("STAGE1_H5_PATH")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", ".")
os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Setup] Device={DEVICE}")

BATCH_SIZE = 32
MAX_EPOCHS = 120
PATIENCE = 20
LR = 1e-4
FP_BITS = 2048

# ---- cell ----
class CPDataset(Dataset):
    def __init__(self, xm, xc, xd, y, fold):
        self.xm = torch.tensor(xm, dtype=torch.float32)
        self.xc = torch.tensor(xc, dtype=torch.float32)
        self.xd = torch.tensor(xd, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.fold = torch.tensor(fold, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (self.xm[i], self.xc[i], self.xd[i]), self.y[i], self.fold[i]


def _fp(smiles_list, nbits=2048):
    if Chem is None or AllChem is None:
        return np.zeros((len(smiles_list), nbits), dtype=np.float32)
    out = []
    for sm in smiles_list:
        try:
            mol = Chem.MolFromSmiles(sm)
            if mol is None:
                out.append(np.zeros(nbits, dtype=np.float32))
                continue
            bv = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=nbits)
            out.append(np.asarray(bv, dtype=np.float32))
        except Exception:
            out.append(np.zeros(nbits, dtype=np.float32))
    return np.vstack(out)


def load_data():
    if not STAGE1_H5_PATH or not os.path.exists(STAGE1_H5_PATH):
        raise FileNotFoundError(f"Missing STAGE1_H5_PATH: {STAGE1_H5_PATH}")
    with h5py.File(STAGE1_H5_PATH, "r") as f:
        grp = f["combined"]
        mpre = grp["morphology_pre"][:]
        mpost = grp["morphology_post"][:]
        smiles = grp["smiles"][:]
        dose = grp["dose"][:]
        split = grp["split_id"][:]

    if smiles.dtype.kind == "S":
        smiles = [s.decode("utf-8") for s in smiles]
    else:
        smiles = [str(s) for s in smiles]

    xc = _fp(smiles, nbits=FP_BITS)
    xd = np.log10(np.asarray(dose, dtype=np.float32) + 1e-6)
    mean = np.mean(mpre, axis=0)
    std = np.std(mpre, axis=0) + 1e-6
    xm = (mpre - mean) / std
    return xm, xc, xd, mpost, split


XM, XC, XD, Y, SPLIT = load_data()
D_M, D_C, D_O = XM.shape[1], XC.shape[1], Y.shape[1]


def fold_loaders(f):
    vm = SPLIT == f
    tm = ~vm
    tr = CPDataset(XM[tm], XC[tm], XD[tm], Y[tm], SPLIT[tm])
    va = CPDataset(XM[vm], XC[vm], XD[vm], Y[vm], SPLIT[vm])
    tr_mean = torch.tensor(np.mean(Y[tm], axis=0), dtype=torch.float32, device=DEVICE)
    return DataLoader(tr, batch_size=BATCH_SIZE, shuffle=True), DataLoader(va, batch_size=BATCH_SIZE), tr_mean

# ---- cell ----
class GRLFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a):
        ctx.a = a
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.a * g, None


class GRL(nn.Module):
    def __init__(self, a=0.5):
        super().__init__()
        self.a = a

    def forward(self, x):
        return GRLFunc.apply(x, self.a)


class ResBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d), nn.BatchNorm1d(d), nn.ReLU(), nn.Dropout(0.2))

    def forward(self, x):
        return x + self.net(x)


class EquiMorphGAT(nn.Module):
    def __init__(self, dm, dc, do):
        super().__init__()
        h = 512
        self.nt, self.ed = 16, 32
        self.m_enc = nn.Sequential(nn.Linear(dm, h), nn.BatchNorm1d(h), nn.ReLU(), ResBlock(h), ResBlock(h))
        self.c_enc = nn.Sequential(nn.Linear(dc, h), nn.BatchNorm1d(h), nn.ReLU(), ResBlock(h), ResBlock(h))
        self.d_enc = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 64))

        self.grl = GRL(0.5)
        self.disc = nn.Sequential(nn.Linear(h, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 6))

        self.attn = nn.MultiheadAttention(embed_dim=self.ed, num_heads=4, batch_first=True)
        self.chem_gate = nn.Sequential(nn.Linear(h, h), nn.Sigmoid())
        self.dose_gate = nn.Sequential(nn.Linear(64, h), nn.Sigmoid())
        self.norm = nn.LayerNorm(h)

        self.pred = nn.Sequential(
            nn.Linear(h + h + 64, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, do)
        )

    def forward(self, xm, xc, xd):
        hm, hc, hd = self.m_enc(xm), self.c_enc(xc), self.d_enc(xd)
        fold_logits = self.disc(self.grl(hm))

        q = hm.view(-1, self.nt, self.ed)
        kv = hc.view(-1, self.nt, self.ed)
        attn, _ = self.attn(q, kv, kv)
        attn = attn.reshape(-1, 512)

        p = attn * self.chem_gate(hc) * self.dose_gate(hd)
        hf = self.norm(hm + p)
        y = self.pred(torch.cat([hf, hc, hd], dim=1))
        return y, fold_logits


class BaselineMLP(nn.Module):
    def __init__(self, dm, dc, do):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dm + dc + 1, 1024), nn.ReLU(), nn.Dropout(0.2), nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, do))

    def forward(self, xm, xc, xd):
        return self.net(torch.cat([xm, xc, xd], dim=1))

# ---- cell ----
def calc_metrics(y_true, y_pred, mean_train):
    yt = y_true.detach().cpu().numpy()
    yp = y_pred.detach().cpu().numpy()
    mu = mean_train.detach().cpu().numpy()

    mse = mean_squared_error(yt, yp)
    r2 = r2_score(yt, yp)
    pcc = pearsonr(yt.reshape(-1), yp.reshape(-1))[0] if np.std(yp) > 1e-9 else 0.0

    dev = np.abs(yt - mu)
    out = {}
    for k in (20, 50):
        rm, pc = [], []
        for i in range(len(yt)):
            idx = np.argsort(dev[i])[::-1][:k]
            a, b = yt[i, idx], yp[i, idx]
            rm.append(np.sqrt(mean_squared_error(a, b)))
            if np.std(a) > 1e-9 and np.std(b) > 1e-9:
                pc.append(pearsonr(a, b)[0])
            else:
                pc.append(0.0)
        out[f"DEG_RMSE_{k}"] = float(np.mean(rm))
        out[f"DEG_PCC_{k}"] = float(np.mean(pc))

    ytd, ypd = yt - mu, yp - mu
    out.update({
        "MSE": float(mse), "PCC": float(pcc), "R2": float(r2),
        "MSE_DM": float(mean_squared_error(ytd, ypd)),
        "PCC_DM": float(pearsonr(ytd.reshape(-1), ypd.reshape(-1))[0] if np.std(ypd) > 1e-9 else 0.0),
        "R2_DM": float(r2_score(ytd, ypd)),
    })
    return out


def train(name, model_ctor, folds=(1, 2, 3, 4, 5)):
    aggregate = {k: [] for k in ["MSE", "PCC", "R2", "DEG_RMSE_20", "DEG_RMSE_50", "DEG_PCC_20", "DEG_PCC_50", "MSE_DM", "PCC_DM", "R2_DM"]}
    for f in folds:
        tr, va, mean_y = fold_loaders(f)
        model = model_ctor().to(DEVICE)
        opt = optim.Adam(model.parameters(), lr=LR)
        mse = nn.MSELoss()
        ce = nn.CrossEntropyLoss()
        best, bad = 1e18, 0
        best_path = os.path.join(OUTPUT_DIR, f"best_{name}_fold{f}.pt")

        for ep in range(MAX_EPOCHS):
            model.train()
            for (xm, xc, xd), y, fold in tr:
                xm, xc, xd, y, fold = xm.to(DEVICE), xc.to(DEVICE), xd.to(DEVICE), y.to(DEVICE), fold.to(DEVICE)
                opt.zero_grad()
                if name == "EquiMorph-GAT":
                    yp, fp = model(xm, xc, xd)
                    loss = mse(yp, y) + 0.1 * ce(fp, fold - 1)
                else:
                    yp = model(xm, xc, xd)
                    loss = mse(yp, y)
                loss.backward()
                opt.step()

            model.eval()
            vm = 0.0
            with torch.no_grad():
                for (xm, xc, xd), y, _ in va:
                    xm, xc, xd, y = xm.to(DEVICE), xc.to(DEVICE), xd.to(DEVICE), y.to(DEVICE)
                    yp = model(xm, xc, xd)[0] if name == "EquiMorph-GAT" else model(xm, xc, xd)
                    vm += mse(yp, y).item()
            vm /= max(len(va), 1)
            if vm < best:
                best, bad = vm, 0
                torch.save(model.state_dict(), best_path)
            else:
                bad += 1
                if bad >= PATIENCE:
                    break

        model.load_state_dict(torch.load(best_path, map_location=DEVICE))
        model.eval()
        pred, true = [], []
        with torch.no_grad():
            for (xm, xc, xd), y, _ in va:
                xm, xc, xd = xm.to(DEVICE), xc.to(DEVICE), xd.to(DEVICE)
                yp = model(xm, xc, xd)[0] if name == "EquiMorph-GAT" else model(xm, xc, xd)
                pred.append(yp)
                true.append(y.to(DEVICE))
        m = calc_metrics(torch.cat(true), torch.cat(pred), mean_y)
        for k, v in m.items():
            aggregate[k].append(v)

    return {k: float(np.mean(v)) for k, v in aggregate.items()}


baseline = train("Baseline", lambda: BaselineMLP(D_M, D_C, D_O))
innov = train("EquiMorph-GAT", lambda: EquiMorphGAT(D_M, D_C, D_O))
winner = "EquiMorph-GAT" if innov["PCC"] > baseline["PCC"] else "Baseline"

store = {
    "models": {
        "Baseline": {"aggregate": baseline},
        "EquiMorph-GAT": {"aggregate": innov},
    },
    "winner": winner,
}

with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
    json.dump(store, f, indent=2)

with open(os.path.join(OUTPUT_DIR, "analysis_summary.json"), "w", encoding="utf-8") as f:
    json.dump({
        "experiment_name": "DD-MIA Reference",
        "winner": winner,
        "improvement_pcc": float(innov["PCC"] - baseline["PCC"]),
    }, f, indent=2)

md = f"""
# Experiment Report: EquiMorph-GAT vs Baseline

## Winner: {winner}

| Metric | Baseline | EquiMorph-GAT |
|---|---:|---:|
| MSE | {baseline['MSE']:.4f} | {innov['MSE']:.4f} |
| PCC | {baseline['PCC']:.4f} | {innov['PCC']:.4f} |
| R2 | {baseline['R2']:.4f} | {innov['R2']:.4f} |
| DEG_RMSE_20 | {baseline['DEG_RMSE_20']:.4f} | {innov['DEG_RMSE_20']:.4f} |
| DEG_PCC_20 | {baseline['DEG_PCC_20']:.4f} | {innov['DEG_PCC_20']:.4f} |
"""
with open(os.path.join(OUTPUT_DIR, "experiment_report.md"), "w", encoding="utf-8") as f:
    f.write(md)

print(json.dumps(store, indent=2))
print(f"[Output] saved in {os.path.abspath(OUTPUT_DIR)}")
