"""Synthetic evidence behind the robust filter: self-cancellation, remedies, rank.

Calibrate at s=1, then hold 10 s at a shortened length s1 (no movement, so covariance
lag is excluded). The 12 most separable of about 40 active units are decoded with
``w = R^-1 h / (h^T R^-1 h)`` for templates between the true posture MUAP (lambda=0)
and the static calibration MUAP (lambda=1). ``R`` is a 10 s sample covariance that,
as in Stage B, contains the target unit. Scores are RoA with a 2 ms tolerance at the
MVDR threshold 0.5, with an unsupervised two-means threshold, and at the best threshold.

Usage: ``python -m benchmarks.robust_mvdr.synthetic.study all --output DIR``.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .generator import ElectrodeGrid, MuapModel, discharge_trains, sample_motor_unit_pool

FS, R_EXT, TOL = 2048.0, 10, 4  # tolerance 4 samples ~ 2 ms
SNR_LINEAR = 100.0  # 20 dB per channel


def extend(x, r=R_EXT):
    """Block (delay-major) extension, block i = x(t - i), as adapt_decomp."""
    n, m = x.shape
    out = np.zeros((n, m * r), np.float32)
    for i in range(r):
        out[i:, m * i : m * (i + 1)] = x[: n - i]
    return out


def ext_template(muap, delay, r=R_EXT):
    """Extended vector of a discharge at t0 observed at t0 + delay."""
    m, length = muap.shape
    cols = [muap[:, delay - i] if 0 <= delay - i < length else np.zeros(m) for i in range(r)]
    return np.concatenate(cols)


def all_lags(muap, r=R_EXT):
    return np.stack([ext_template(muap, tau, r) for tau in range(muap.shape[1] + r - 1)])


def render(bank_by_unit, trains, n, length):
    """Per-unit clean components so that target-free covariances are exact."""
    comps = {}
    for u, muap in bank_by_unit.items():
        x = np.zeros((n + length, muap.shape[0]))
        for t in trains[u]:
            x[t : t + length] += muap.T
        comps[u] = x[:n]
    return comps


def roa(ref, pred, tol=TOL):
    i = j = tp = 0
    while i < len(ref) and j < len(pred):
        if abs(ref[i] - pred[j]) <= tol:
            tp, i, j = tp + 1, i + 1, j + 1
        elif ref[i] < pred[j]:
            i += 1
        else:
            j += 1
    return tp / (len(ref) + len(pred) - tp) if len(ref) + len(pred) else 0.0, tp / max(len(ref), 1)


def peaks(y, min_dist=20):
    """Local maxima separated by at least ``min_dist`` samples (greedy by height)."""
    cand = np.flatnonzero((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:])) + 1
    order = cand[np.argsort(-y[cand])]
    taken = np.zeros(len(y), bool)
    keep = []
    for c in order:
        if not taken[max(0, c - min_dist) : c + min_dist].any():
            keep.append(c)
            taken[c] = True
    return np.sort(np.array(keep, int))


def detect_score(y, truth, delay, threshold):
    pk = peaks(y)
    pk = pk[y[pk] > threshold]
    return roa(truth, pk - delay)


def best_threshold_score(y, truth, delay):
    pk = peaks(y)
    heights = y[pk]
    best = (0.0, 0.0, 0.0)
    for thr in np.quantile(heights, np.linspace(0.5, 0.995, 60)):
        score = roa(truth, pk[heights > thr] - delay)
        if score[0] > best[0]:
            best = (score[0], score[1], float(thr))
    return best


def kmeans_threshold(y):
    """Unsupervised two-cluster split of peak heights (what adaptive centroids track)."""
    pk = peaks(y)
    h = np.sort(y[pk])[-max(20, len(pk) // 4) :]  # strongest quarter of local maxima
    lo, hi = np.quantile(h, 0.1), np.quantile(h, 0.9)
    for _ in range(50):
        thr = 0.5 * (lo + hi)
        a, b = h[h <= thr], h[h > thr]
        if not len(a) or not len(b):
            break
        lo, hi = a.mean(), b.mean()
    return 0.5 * (lo + hi)


def mvdr(cov, h):
    w = np.linalg.solve(cov, h)
    return w / (h @ w)


def _best_delay(muap, inverse, length):
    def energy(d):
        c = ext_template(muap, d)
        return c @ inverse @ c

    return max(range(R_EXT, length), key=energy)


def _setup():
    """Pool, trains, calibration noise level and the 12 most separable units at s=1."""
    rng = np.random.default_rng(3)
    grid = ElectrodeGrid()
    model = MuapModel(grid)
    pool = sample_motor_unit_pool(64, rng)
    n_cal, n_hold, length = int(5 * FS), int(10 * FS), grid.n_samples
    trains_cal = discharge_trains(pool, 0.15, n_cal, FS, rng)
    trains_hold = discharge_trains(pool, 0.15, n_hold, FS, rng)
    active = [u for u in range(pool.n_units) if trains_cal[u].size and trains_hold[u].size]
    muap_cal = {u: model.unit_muap(pool, u, 1.0).reshape(grid.n_channels, length) for u in active}
    clean_cal = sum(render(muap_cal, trains_cal, n_cal, length).values())
    noise_sd = np.sqrt(np.mean(clean_cal**2, axis=0) / SNR_LINEAR)
    xe_cal = extend(clean_cal + rng.normal(size=clean_cal.shape) * noise_sd)
    xe_cal -= xe_cal.mean(0)
    r_cal = xe_cal.T @ xe_cal / len(xe_cal)
    inverse = np.linalg.inv(r_cal + 1e-6 * np.trace(r_cal) / len(r_cal) * np.eye(len(r_cal)))
    sir = {}
    for u in active:
        best = _best_delay(muap_cal[u], inverse, length)
        c = ext_template(muap_cal[u], best)
        sir[u] = (float(c @ inverse @ c), best)
    units = sorted(sir, key=lambda u: -sir[u][0])[:12]
    return dict(
        rng=rng,
        grid=grid,
        model=model,
        pool=pool,
        n_hold=n_hold,
        length=length,
        trains_hold=trains_hold,
        active=active,
        muap_cal=muap_cal,
        noise_sd=noise_sd,
        sir=sir,
        units=units,
    )


def _hold(setup, s1):
    grid, length = setup["grid"], setup["length"]
    muaps = {
        u: setup["model"].unit_muap(setup["pool"], u, s1).reshape(grid.n_channels, length)
        for u in setup["active"]
    }
    comps = render(muaps, setup["trains_hold"], setup["n_hold"], length)
    noise = setup["rng"].normal(size=(setup["n_hold"], grid.n_channels)) * setup["noise_sd"]
    xe = extend(sum(comps.values()) + noise)
    xe -= xe.mean(0)
    return muaps, comps, xe


def _without_unit(r_full, xe, comp):
    comp = extend(comp)
    comp -= comp.mean(0)
    return r_full - (comp.T @ xe + xe.T @ comp - comp.T @ comp) / len(xe)


def self_nulling():
    """Covariances with and without the target, diagonal loading and Stage B's mixing."""
    setup = _setup()
    units, sir, n_hold = setup["units"], setup["sir"], setup["n_hold"]
    rates = {u: setup["trains_hold"][u].size / n_hold for u in units}
    results = {"active_units": len(setup["active"]), "postures": {}}
    for s1 in (0.98, 0.96, 0.94, 0.91):
        muap_s1, comps, xe = _hold(setup, s1)
        r_full = xe.T @ xe / len(xe)
        dim = len(r_full)
        eye = np.eye(dim)
        mean_eig = np.trace(r_full) / dim
        posture = {}
        for lam in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):
            names = ("full", "target_free", "loading_0.1", "loading_1", "repo_alpha0.5")
            rows = {k: [] for k in names}
            cosines = []
            for u in units:
                delay = sir[u][1]
                h_true = ext_template(muap_s1[u], delay)
                h = h_true + lam * (ext_template(setup["muap_cal"][u], delay) - h_true)
                w_true = np.linalg.solve(r_full, h_true)
                cosines.append(
                    float(
                        h @ w_true / np.sqrt((h @ np.linalg.solve(r_full, h)) * (h_true @ w_true))
                    )
                )
                hm = []
                for v in units:
                    hv = ext_template(muap_s1[v], sir[v][1])
                    hm.append(hv + lam * (ext_template(setup["muap_cal"][v], sir[v][1]) - hv))
                hm = np.stack(hm)
                model_cov = hm.T @ (hm * np.array([rates[v] for v in units])[:, None])
                residual = max((np.trace(r_full) - np.trace(model_cov)) / dim, 0.01)
                r_repo = 0.5 * r_full + 0.5 * (model_cov + residual * eye)
                r_repo += 0.001 * np.trace(r_repo) / dim * eye
                covs = {
                    "full": r_full + 1e-6 * mean_eig * eye,
                    "target_free": _without_unit(r_full, xe, comps[u]) + 1e-6 * mean_eig * eye,
                    "loading_0.1": r_full + 0.1 * mean_eig * eye,
                    "loading_1": r_full + 1.0 * mean_eig * eye,
                    "repo_alpha0.5": r_repo,
                }
                truth = setup["trains_hold"][u]
                for name, cov in covs.items():
                    w = mvdr(cov, h)
                    y = xe @ w
                    fixed = detect_score(y, truth, delay, 0.5)
                    rows[name].append(
                        dict(
                            true_response=float(w @ h_true),
                            roa_fixed=fixed[0],
                            recall_fixed=fixed[1],
                            roa_best=best_threshold_score(y, truth, delay)[0],
                        )
                    )
            posture[f"lambda={lam}"] = {
                "whitened_cosine_template_vs_true_median": float(np.median(cosines)),
                **{
                    name: {
                        "true_unit_response_median": float(
                            np.median([r["true_response"] for r in rs])
                        ),
                        "roa_fixed_threshold_mean": float(np.mean([r["roa_fixed"] for r in rs])),
                        "recall_fixed_threshold_mean": float(
                            np.mean([r["recall_fixed"] for r in rs])
                        ),
                        "roa_best_threshold_mean": float(np.mean([r["roa_best"] for r in rs])),
                    }
                    for name, rs in rows.items()
                },
            }
            print("self-nulling", s1, lam, flush=True)
        results["postures"][f"s={s1}"] = posture
    return results


def _blanked_covariance(xe, times, before, after):
    keep = np.ones(len(xe), bool)
    for t in times:
        keep[max(0, t - before) : t + after] = False
    kept = xe[keep]
    return kept.T @ kept / max(len(kept), 1), float(keep.mean())


def remedies():
    """Label-free remedies: detection blanking, model subtraction, loading."""
    setup = _setup()
    units, sir, n_hold, length = setup["units"], setup["sir"], setup["n_hold"], setup["length"]
    results = {}
    for s1 in (0.96, 0.94, 0.91):
        muap_s1, comps, xe = _hold(setup, s1)
        r_full = xe.T @ xe / len(xe)
        dim = len(r_full)
        eye = np.eye(dim)
        mean_eig = np.trace(r_full) / dim
        tiny = 1e-6 * mean_eig * eye
        for lam in (0.0, 0.5, 1.0):
            scores, kept_fraction = {}, []
            for u in units:
                delay, truth = sir[u][1], setup["trains_hold"][u]
                h_muap = muap_s1[u] + lam * (setup["muap_cal"][u] - muap_s1[u])
                h = ext_template(h_muap, delay)
                lags = all_lags(h_muap)
                r_model = r_full - truth.size / n_hold * lags.T @ lags
                ev, evec = np.linalg.eigh(r_model)
                r_model = (evec * np.maximum(ev, 1e-3 * mean_eig)) @ evec.T
                # Decision-directed blanking: exclude [t - (R-1), t + L + R) around the
                # unit's detections, re-solve, twice.
                y = xe @ mvdr(r_full + tiny, h)
                for _ in range(2):
                    pk = peaks(y)
                    detected = pk[y[pk] > 0.5] - delay
                    r_blank, fraction = _blanked_covariance(xe, detected, R_EXT, length + R_EXT)
                    y = xe @ mvdr(r_blank + tiny, h)
                kept_fraction.append(fraction)
                covs = {
                    "full": r_full + tiny,
                    "oracle_free": _without_unit(r_full, xe, comps[u]) + tiny,
                    "blank_detected": r_blank + tiny,
                    "model_subtract": r_model,
                    "loading_0.1": r_full + 0.1 * mean_eig * eye,
                    "blank_plus_loading": r_blank + 0.03 * mean_eig * eye,
                }
                for name, cov in covs.items():
                    yy = xe @ mvdr(cov, h)
                    scores.setdefault(name, []).append(
                        (
                            detect_score(yy, truth, delay, 0.5)[0],
                            best_threshold_score(yy, truth, delay)[0],
                        )
                    )
            row = {
                name: {
                    "roa_fixed": float(np.mean([v[0] for v in vals])),
                    "roa_best_threshold": float(np.mean([v[1] for v in vals])),
                }
                for name, vals in scores.items()
            }
            row["blank_kept_sample_fraction"] = float(np.mean(kept_fraction))
            results[f"s={s1},lambda={lam}"] = row
            print("remedies", s1, lam, flush=True)
    return results


def _subtract(r, lags, rate, rho, floor):
    model = r - rho * rate * lags.T @ lags
    ev, vec = np.linalg.eigh(model)
    return (vec * np.maximum(ev, floor)) @ vec.T


def combinations():
    """Partial subtraction plus loading, scored at fixed, two-means and best thresholds."""
    setup = _setup()
    units, sir, n_hold = setup["units"], setup["sir"], setup["n_hold"]
    results = {}
    for s1 in (0.96, 0.94, 0.91):
        muap_s1, _, xe = _hold(setup, s1)
        r = xe.T @ xe / len(xe)
        dim = len(r)
        eye = np.eye(dim)
        me = np.trace(r) / dim
        for lam in (0.0, 0.5, 1.0):
            scores = {}
            for u in units:
                delay, truth = sir[u][1], setup["trains_hold"][u]
                h_muap = muap_s1[u] + lam * (setup["muap_cal"][u] - muap_s1[u])
                h = ext_template(h_muap, delay)
                lags = all_lags(h_muap)
                rate = truth.size / n_hold
                covs = {
                    "full": r + 1e-6 * me * eye,
                    "loading_0.03": r + 0.03 * me * eye,
                    "loading_0.1": r + 0.1 * me * eye,
                    "subtract_0.5+load_0.03": _subtract(r, lags, rate, 0.5, 1e-3 * me)
                    + 0.03 * me * eye,
                    "subtract_1+load_0.1": _subtract(r, lags, rate, 1.0, 1e-3 * me)
                    + 0.1 * me * eye,
                }
                for name, cov in covs.items():
                    y = xe @ mvdr(cov, h)
                    scores.setdefault(name, []).append(
                        (
                            detect_score(y, truth, delay, 0.5)[0],
                            detect_score(y, truth, delay, kmeans_threshold(y))[0],
                            best_threshold_score(y, truth, delay)[0],
                        )
                    )
            results[f"s={s1},lambda={lam}"] = {
                "columns": ["roa_fixed_0.5", "roa_two_means", "roa_best_threshold"],
                **{
                    n: [round(float(np.mean([v[i] for v in vals])), 3) for i in range(3)]
                    for n, vals in scores.items()
                },
            }
            print("combinations", s1, lam, flush=True)
    return results


def _covariance(lag_sets, rates, noise_var):
    dim = lag_sets[0].shape[1]
    cov = np.zeros((dim, dim))
    for lags, rate in zip(lag_sets, rates, strict=True):
        cov += rate * lags.T @ lags
    return cov + np.diag(np.tile(noise_var, R_EXT))


def _kl_to_identity(eigs):
    return 0.5 * float(np.sum(eigs - 1 - np.log(eigs)))


def whitening_rank():
    """KL drift of calibration-ZCA coordinates left after rank-r or diagonal corrections.

    Exact extended covariances R(s) = sum_u rate_u sum_tau M_u(tau) M_u(tau)^T + noise.
    In calibration ZCA coordinates C(s) = W0 R(s) W0 the drift is KL(C(s) || I).
    """
    rng = np.random.default_rng(3)
    grid = ElectrodeGrid()
    model = MuapModel(grid)
    pool = sample_motor_unit_pool(64, rng)
    excitation = 0.15
    active = np.flatnonzero(pool.threshold <= excitation)
    rates = np.minimum(8 + 60 * (excitation - pool.threshold[active]), 35) / FS
    shape = (grid.n_channels, grid.n_samples)

    def lags_at(s):
        return [all_lags(model.unit_muap(pool, int(u), s).reshape(shape)) for u in active]

    cal = lags_at(1.0)
    signal = _covariance(cal, rates, np.zeros(grid.n_channels))
    noise_var = np.diag(signal)[: grid.n_channels] / SNR_LINEAR
    evals, evecs = np.linalg.eigh(_covariance(cal, rates, noise_var))
    w0 = evecs @ np.diag(evals**-0.5) @ evecs.T  # ZCA, adapt_decomp's default
    results = {"active_units": int(active.size), "postures": {}}
    for s1 in (0.99, 0.98, 0.96, 0.94, 0.91, 0.88):
        c = w0 @ _covariance(lags_at(s1), rates, noise_var) @ w0
        c = 0.5 * (c + c.T)
        eigs = np.linalg.eigvalsh(c)
        total = _kl_to_identity(eigs)
        f = np.sort(eigs - 1 - np.log(eigs))[::-1]
        row = {"kl_total": total}
        for r in (3, 6, 18, 50):
            row[f"rank{r}_residual_fraction_fix_r_dirs"] = float(0.5 * f[r:].sum() / total)
            row[f"rank{r}_residual_fraction_lower_bound_2r"] = float(0.5 * f[2 * r :].sum() / total)
        d = np.diag(c)
        corr = c / np.sqrt(np.outer(d, d))
        row["diagonal_residual_fraction"] = float(-0.5 * np.linalg.slogdet(corr)[1] / total)
        # One gain per channel shared by its ten delays (100 parameters, block order).
        g = d.reshape(R_EXT, grid.n_channels).mean(0)
        scale = np.tile(g**-0.5, R_EXT)
        row["channel_gain_residual_fraction"] = float(
            _kl_to_identity(np.linalg.eigvalsh(c * np.outer(scale, scale))) / total
        )
        e2 = np.linalg.eigvalsh(corr)
        f2 = np.sort(e2 - 1 - np.log(e2))[::-1]
        row["diagonal_plus_rank3_residual_fraction"] = float(0.5 * f2[3:].sum() / total)
        results["postures"][f"s={s1}"] = row
        print("whitening-rank", s1, flush=True)
    return results


STUDIES = {
    "self-nulling": self_nulling,
    "remedies": remedies,
    "combinations": combinations,
    "whitening-rank": whitening_rank,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", choices=[*STUDIES, "all"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name, study in STUDIES.items():
        if args.study in (name, "all"):
            path = args.output / f"robust_mvdr_synthetic_{name.replace('-', '_')}.json"
            path.write_text(json.dumps(study(), indent=1) + "\n", encoding="utf-8")
            print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
