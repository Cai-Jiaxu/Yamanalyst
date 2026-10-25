"""Learned temporal refiner: a small bidirectional GRU over the tree model's per-frame
probabilities and a few ball/rule signals, trained on external matches with true labels.

The trees score each frame from multi-scale window features; the Viterbi decoder then smooths with
fixed switch costs. A sequence model can learn context-dependent transitions instead (where a
possession is likely to start or end given what came before and after), the role MS-TCN /
ASRF-style refinement stages play in temporal action segmentation and that GRU/UNet heads played
in per-step segmentation competitions (Kaggle CMI sleep states). Runs at 5 fps on CPU.

    Z = inputs(P, B, f, R)             # (n, D) per frame at 25 fps
    net = fit([(Z, codes), ...])       # codes: 0 H, 1 A, 2 D, -1 ungraded
    Q = predict(net, Z)                # (n, 3) refined probabilities at 25 fps
"""
import numpy as np

STEP = 5          # 25 fps -> 5 fps
CHUNK = 512       # training crop, steps (~100 s)
D_IN = 13


def inputs(P, B, f, R, seen_r=12):
    """Per-frame inputs at 25 fps: log P (3), boundary-head logit, ball seen, ball near, rule vote
    x near (3), nearest-team distance margin, ball speed, team pace, frames since ball seen."""
    seen = f["seen"].astype(bool)
    near = np.convolve(seen.astype(float), np.ones(2 * seen_r + 1), "same") > 0
    lp = np.log(np.clip(P, 1e-4, 1.0))
    b = np.clip(B if B is not None else np.full(len(P), 0.1), 1e-4, 1 - 1e-4)
    rv = np.stack([(R == s) & near for s in "HAD"], 1).astype(float)
    margin = np.where(seen, np.clip(np.nan_to_num(f["da"] - f["dh"]), -15, 15) / 15, 0.0)
    bs = np.where(seen, np.clip(np.nan_to_num(f["bspeed"]), 0, 25) / 25, 0.0)
    pace = np.clip(np.nan_to_num((f["sh"] + f["sa"]) / 2), 0, 6) / 6
    idx = np.where(seen, np.arange(len(seen)), -1)
    last = np.maximum.accumulate(idx)
    since = np.minimum(np.where(last >= 0, np.arange(len(seen)) - last, 750), 750) / 750
    return np.column_stack([lp, np.log(b / (1 - b)), seen, near, rv, margin, bs, pace,
                            since]).astype(np.float32)


def mirror(Z, y):
    """Home/away swap: exact symmetry of the inputs (H and A channels swapped, margin negated)."""
    Z = Z.copy()
    Z[:, [0, 1]] = Z[:, [1, 0]]
    Z[:, [6, 7]] = Z[:, [7, 6]]
    Z[:, 9] = -Z[:, 9]
    y = y.copy()
    h, a = y == 0, y == 1
    y[h], y[a] = 1, 0
    return Z, y


def _down(Z, y=None):
    n = len(Z) // STEP * STEP
    Zd = Z[:n].reshape(-1, STEP, Z.shape[1]).mean(1)
    if y is None:
        return Zd
    yd = y[:n].reshape(-1, STEP)[:, STEP // 2]
    return Zd, yd


def _tcn(hidden=64, depth=8):
    """Single-stage MS-TCN: residual dilated convolutions (k=3, dilation 1..2^(depth-1)),
    non-causal; depth 8 at 5 fps sees about +-51 s."""
    import torch.nn as nn

    class Layer(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.conv = nn.Conv1d(hidden, hidden, 3, padding=d, dilation=d)
            self.mix = nn.Conv1d(hidden, hidden, 1)
            self.drop = nn.Dropout(0.1)

        def forward(self, x):
            return x + self.drop(self.mix(self.conv(x).relu()))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Conv1d(D_IN, hidden, 1)
            self.layers = nn.Sequential(*[Layer(2 ** i) for i in range(depth)])
            self.out = nn.Conv1d(hidden, 3, 1)

        def forward(self, x):                      # (b, t, D) -> (b, t, 3)
            return self.out(self.layers(self.inp(x.transpose(1, 2)))).transpose(1, 2)
    return Net()


def _net(hidden=64, layers=2):
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(D_IN, hidden)
            self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True,
                              bidirectional=True, dropout=0.1)
            self.out = nn.Linear(2 * hidden, 3)

        def forward(self, x):
            h, _ = self.gru(self.inp(x).relu())
            return self.out(h)
    return Net()


def fit(seqs, epochs=25, seed=0, hidden=64, lr=2e-3, smooth=0.0, verbose=False,
        arch="gru"):
    """seqs: [(Z (n, D), codes (n,))] at 25 fps. Returns the trained net (CPU, deterministic)."""
    import torch
    torch.manual_seed(seed)
    torch.set_num_threads(8)
    rng = np.random.default_rng(seed)
    data = []
    for Z, y in seqs:
        for z, t in (( Z, y), mirror(Z, y)):
            zd, td = _down(z, t)
            data.append((torch.from_numpy(zd), torch.from_numpy(td.astype(np.int64))))
    net = _tcn(hidden) if arch == "tcn" else _net(hidden)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-3)
    total = sum(len(z) for z, _ in data)
    per_epoch = max(1, total // CHUNK)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * per_epoch // 16 + 1)
    lossf = torch.nn.CrossEntropyLoss(ignore_index=-1)
    for ep in range(epochs):
        tot = 0.0
        for _ in range(per_epoch // 16):
            xb, yb = [], []
            for _ in range(16):
                z, t = data[rng.integers(len(data))]
                s = rng.integers(0, max(1, len(z) - CHUNK))
                xb.append(z[s:s + CHUNK]), yb.append(t[s:s + CHUNK])
            L = min(len(x) for x in xb)
            x = torch.stack([v[:L] for v in xb])
            y = torch.stack([v[:L] for v in yb])
            logit = net(x)
            loss = lossf(logit.reshape(-1, 3), y.reshape(-1))
            if smooth:
                lp = logit.log_softmax(-1)
                d = (lp[:, 1:] - lp[:, :-1].detach()).clamp(-4, 4) ** 2
                loss = loss + smooth * d.mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
        if verbose:
            print("  epoch {} loss {:.4f}".format(ep, tot / max(1, per_epoch // 16)), flush=True)
    net.eval()
    return net


def predict(net, Z):
    """(n, 3) probabilities at 25 fps (5-fps outputs, log-linearly interpolated)."""
    import torch
    zd = _down(Z)
    with torch.no_grad():
        lp = net(torch.from_numpy(zd)[None])[0].log_softmax(-1).numpy()
    t = np.arange(len(zd)) * STEP + STEP // 2
    full = np.arange(len(Z))
    out = np.stack([np.interp(full, t, lp[:, k]) for k in range(3)], 1)
    P = np.exp(out)
    return P / P.sum(1, keepdims=True)
