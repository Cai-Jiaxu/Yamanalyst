"""Tested semi-Markov span decoder with reachability assertions and forced tests.

Tests:
  T1: 5H/5A/5D blocks recovered.
  T2: every-frame boundaries + zero prior = framewise Viterbi.
  T3: >L_MAX constant block must be recovered (regular endpoints guarantee connectivity).
  T4: multiple forced switches with varied lengths.
"""
import numpy as np

STATES = np.array(["H", "A", "D"])


def regular_boundaries(arg, L_MAX):
    """Union of argmax changes and every L_MAX frames. Guarantees Viterbi connectivity."""
    n = len(arg)
    ch = np.flatnonzero(arg[1:] != arg[:-1]) + 1
    reg = np.arange(L_MAX, n, L_MAX)
    cb = np.unique(np.concatenate(([0], ch, reg, [n])))
    return cb


def semimarkov(logP, boundaries, dur_log, T, max_len=None):
    n = len(logP)
    cb = np.asarray(boundaries, dtype=int)
    K = len(cb)
    NEG = -1e9
    dp = np.full((K, 3), NEG)
    bp = np.zeros((K, 3, 2), dtype=np.int32)
    csum = np.vstack([np.zeros((1, 3)), np.cumsum(logP, axis=0)])

    def span(a, b, s, mode="sum"):
        length = b - a
        if mode == "sum":
            return csum[b, s] - csum[a, s] + dur_log[length, s]
        else:  # true mean length-normalized
            return (csum[b, s] - csum[a, s]) / length + dur_log[length, s]

    dp[0] = 0.0
    for i in range(1, K):
        ce = cb[i]
        for j in range(i - 1, -1, -1):
            cs = cb[j]
            length = ce - cs
            if max_len is not None and length > max_len:
                break
            for s in range(3):
                if cs >= n:
                    continue
                sc = span(cs, min(ce, n), s)
                prev = dp[j] + T[:, s]
                bpv = int(prev.argmax())
                cand = prev[bpv] + sc
                if cand > dp[i, s]:
                    dp[i, s] = cand
                    bp[i, s] = (j, bpv)
    # assert reachability before backtrack
    if dp[K - 1].max() <= NEG:
        raise RuntimeError("terminal dp unreachable; boundary gap exceeded max_len")
    out = np.zeros(n, dtype=np.int8)
    i = K - 1
    s = int(dp[i].argmax())
    steps = 0
    while i > 0:
        j, ps = bp[i, s]
        if j >= i:
            raise RuntimeError("backpointer does not progress")
        out[cb[j]:cb[i]] = s
        i, s = j, ps
        steps += 1
    if steps == 0:
        raise RuntimeError("backtrack produced zero segments")
    return out


def framewise_viterbi(logE, T):
    n = len(logE)
    back = np.zeros((n, 3), dtype=np.int8)
    score = logE[0].copy()
    for t in range(1, n):
        cand = score[:, None] + T
        back[t] = cand.argmax(axis=0)
        score = cand.max(axis=0) + logE[t]
    path = np.zeros(n, dtype=np.int8)
    path[-1] = score.argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def _toy_logP(blocks):
    parts = []
    for label, length in blocks:
        p = np.full((length, 3), 0.05)
        p[:, label] = 0.9
        parts.append(np.log(p))
    return np.vstack(parts)


def run_tests():
    T = np.zeros((3, 3))
    T[0, 1] = T[1, 0] = -12
    T[0, 2] = T[2, 0] = T[1, 2] = T[2, 1] = -12

    # T1: 5H/5A/5D
    logP = _toy_logP([(0, 5), (1, 5), (2, 5)])
    dur = np.zeros((16, 3))
    out = semimarkov(logP, np.arange(16), dur, T)
    exp = np.array([0] * 5 + [1] * 5 + [2] * 5)
    print("T1 blocks:", "PASS" if (out == exp).all() else "FAIL", out.tolist())

    # T2: equivalence to framewise Viterbi
    rng = np.random.default_rng(0)
    logP2 = np.log(rng.dirichlet([2, 2, 2], size=50))
    dur2 = np.zeros((51, 3))
    out_sm = semimarkov(logP2, np.arange(51), dur2, T)
    out_v = framewise_viterbi(logP2, T)
    agree = (out_sm == out_v).mean()
    print("T2 viterbi equivalence:", "PASS" if agree == 1.0 else "FAIL", agree)

    # T3: >L_MAX constant block (e.g. 1000 frames H, L_MAX=300)
    logP3 = _toy_logP([(0, 1000), (1, 200), (2, 300)])
    cb3 = regular_boundaries(logP3.argmax(1), 300)
    dur3 = np.zeros((1501, 3))
    out3 = semimarkov(logP3, cb3, dur3, T, max_len=300)
    exp3 = np.array([0] * 1000 + [1] * 200 + [2] * 300)
    print("T3 >L_MAX block:", "PASS" if (out3 == exp3).all() else "FAIL",
          "segments:", int((out3[1:] != out3[:-1]).sum()) + 1)

    # T4: multiple switches varied lengths
    logP4 = _toy_logP([(0, 50), (1, 10), (2, 5), (0, 200), (2, 8), (1, 60)])
    cb4 = regular_boundaries(logP4.argmax(1), 100)
    dur4 = np.zeros((334, 3))
    out4 = semimarkov(logP4, cb4, dur4, T, max_len=100)
    exp4 = np.array([0] * 50 + [1] * 10 + [2] * 5 + [0] * 200 + [2] * 8 + [1] * 60)
    print("T4 multi-switch:", "PASS" if (out4 == exp4).all() else "FAIL",
          "segments:", int((out4[1:] != out4[:-1]).sum()) + 1)


if __name__ == "__main__":
    run_tests()
