"""Guide section 7: method x condition with subject/recording random intercepts."""

import argparse
import time
import warnings

import numpy as np
import pandas as pd
import statsmodels
import statsmodels.formula.api as smf
from threadpoolctl import threadpool_limits

from .common import CONDITIONS, OUTPUT, sha256, write_json


def mixed_model(folder, threads=2):
    started = time.perf_counter()
    source = folder / "per_unit.csv"
    data = pd.read_csv(source)
    if len(data) != 4 * 768 or data.subject.nunique() != 5:
        raise ValueError("The complete, four-method P1 table is required")
    data["roa_pct"] = data.roa * 100
    data["recording"] = data.subject + "/" + data.condition + "/" + data.snr_db.astype(str)
    formula = (
        "roa_pct ~ C(method, Treatment(reference='pareto')) * "
        "C(condition, Treatment(reference='staircase'))"
    )
    result = dict(
        formula=formula,
        groups="subject",
        random_intercepts="subject and recording nested within subject",
        reml=True,
        observations=len(data),
        subjects=5,
        recordings=data.recording.nunique(),
        statsmodels_version=statsmodels.__version__,
        per_unit_sha256=sha256(source),
        code_sha256=sha256(__file__),
        note=(
            "Gaussian approximation to bounded RoA; only five subject clusters. "
            "Repeated methods share units; the guide's random structure does not "
            "include a unit intercept. Subject-cluster bootstrap is reported separately."
        ),
    )
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=threads):
        warnings.simplefilter("always")
        model = smf.mixedlm(
            formula,
            data,
            groups=data.subject,
            re_formula="1",
            vc_formula={"recording": "0 + C(recording)"},
        )
        fit = model.fit(reml=True, method="lbfgs", maxiter=250, disp=False)
        if not fit.converged:
            fit = model.fit(reml=True, method="powell", maxiter=250, disp=False)
        fixed = fit.fe_params
        covariance = fit.cov_params().loc[fixed.index, fixed.index].to_numpy()
        contrasts = []
        for condition in CONDITIONS:
            for method in ("stage_a", "stage_a_no_coupling", "identity_sv_sd"):
                # Both cells occur in the observed full factorial table. Reuse
                # the model's actual design matrix, avoiding formula-engine APIs.
                treatment_row = np.flatnonzero(
                    (data.method == method) & (data.condition == condition)
                )[0]
                comparator_row = np.flatnonzero(
                    (data.method == "pareto") & (data.condition == condition)
                )[0]
                delta = model.exog[treatment_row] - model.exog[comparator_row]
                estimate = float(delta @ fixed.to_numpy())
                se = float(np.sqrt(max(0.0, delta @ covariance @ delta)))
                contrasts.append(
                    dict(
                        condition=condition,
                        method=method,
                        comparator="pareto",
                        delta_pp=estimate,
                        standard_error_pp=se,
                        normal_95_ci_pp=[estimate - 1.96 * se, estimate + 1.96 * se],
                    )
                )
        result.update(
            status="converged" if fit.converged else "did not converge",
            fixed_effects_pp={k: float(v) for k, v in fixed.items()},
            subject_variance=float(fit.cov_re.iloc[0, 0]),
            recording_variance=float(fit.vcomp[0]),
            residual_variance=float(fit.scale),
            log_likelihood=float(fit.llf),
            contrasts=contrasts,
        )
    result["warnings"] = list(dict.fromkeys(str(w.message) for w in caught))
    result["elapsed_seconds"] = time.perf_counter() - started
    write_json(folder / "mixed_model.json", result)
    print({k: result[k] for k in ("status", "observations", "warnings", "elapsed_seconds")})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=2)
    arguments = parser.parse_args()
    if arguments.threads < 1:
        parser.error("--threads must be positive")
    mixed_model(OUTPUT / "evaluations/p1_complete", arguments.threads)
