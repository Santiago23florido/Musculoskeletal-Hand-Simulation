"""Register, predict, seal, score and report one synthetic outer record end to end."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from .test_robust import _synthetic_record


def test_complete_pipeline_on_a_synthetic_outer_record(tmp_path, monkeypatch):
    import adapt_decomp
    import adapt_decomp.utils
    from adapt_decomp.spikes import rate_of_agreement_paired

    from . import evaluate, report

    record, truth = _synthetic_record(units=2, seconds=12)
    record["calibration"].gt_matched_indices = np.arange(2)
    output, data = tmp_path / "out", tmp_path / "data"
    methods = {k: v for k, v in evaluate.METHODS.items() if v["mode"] == "static"}
    for module in (evaluate, report):
        monkeypatch.setattr(module, "OUTPUT", output)
        monkeypatch.setattr(module, "METHODS", methods)
    monkeypatch.setattr(evaluate, "DATA", data)
    monkeypatch.setattr(evaluate, "STAGE_B_OUTPUT", tmp_path / "absent_stage_b")
    monkeypatch.setattr(evaluate, "RECORDS", [("sub-02", "staircase", 30)])
    monkeypatch.setattr(evaluate, "prepare", lambda *args: record)
    monkeypatch.setattr(evaluate, "pareto_path", lambda *args: tmp_path / "pareto.pkl")

    n = truth.shape[0]
    clean = data / "data" / "sub-02" / "clean"
    clean.mkdir(parents=True)
    angle = np.concatenate([np.zeros(7 * 2048), np.linspace(0, 10, n - 7 * 2048)])
    np.savez(clean / "sub-02_FDSI_staircase_angle.npz", angle=angle)
    roa, _, _ = rate_of_agreement_paired(truth, truth, fs=2048, tol_spike_ms=2)
    baseline = SimpleNamespace(spikes=torch.from_numpy(truth), sil=np.full(2, 0.9), roa=roa)
    monkeypatch.setattr(adapt_decomp.AdaptationResult, "load", staticmethod(lambda path: baseline))
    monkeypatch.setattr(adapt_decomp.utils, "load_gt", lambda path, n_samples: truth[:n_samples])

    evaluate.register()
    with pytest.raises(ValueError, match="already exists"):
        evaluate.register()
    evaluate.predict_subject("sub-02")
    evaluate.seal()
    evaluate.score()
    summary = report.report()

    assert summary["instances"] == 2
    assert set(summary["methods"]) == {"pareto", *methods}
    assert summary["methods"]["pareto"]["mean_roa_pct"] == pytest.approx(100)
    assert summary["methods"]["static_subtract"]["mean_roa_pct"] > 80
    assert summary["methods"]["static_subtract"]["by_phase"]["angle_moving_roa"] is not None
    assert (output / "report/per_unit.csv").exists()

    prediction = next((output / "predictions/static_subtract").glob("*.npz"))
    prediction.write_bytes(prediction.read_bytes() + b"tampered")
    for scored in (output / "scored").glob("*.json"):
        scored.unlink()
    with pytest.raises(ValueError, match="changed"):
        evaluate.score()


def test_phase_audit_reports_what_the_fixed_windows_contain(tmp_path, monkeypatch):
    from . import audits

    fs = 2048
    monkeypatch.setattr(audits, "OUTPUT", tmp_path / "out")
    monkeypatch.setattr(audits, "DATA", tmp_path)
    monkeypatch.setattr(audits, "SUBJECTS", ["sub-01", "sub-02"])
    monkeypatch.setattr(audits, "CONDITIONS", ["staircase", "triangular-ramp5s"])
    hold = np.zeros(5 * fs)
    traces = dict(
        staircase=np.concatenate([hold, np.linspace(0, 10, 10 * fs), np.full(5 * fs, 10.0)]),
        **{"triangular-ramp5s": np.concatenate([hold, np.linspace(0, 40, 5 * fs), hold])},
    )
    for subject in ("sub-01", "sub-02"):
        folder = tmp_path / "data" / subject / "clean"
        folder.mkdir(parents=True)
        for condition, angle in traces.items():
            np.savez(folder / f"{subject}_FDSI_{condition}_angle.npz", angle=angle)
    result = audits.phase_audit()
    ramp = result["sub-02/triangular-ramp5s"]
    assert ramp["max_abs_angle_first_10s_deg"] > 39
    assert ramp["moving_fraction_first_10s"] > 0.45
    assert result["sub-02/staircase"]["angle_at_5s_deg"] == 0
    assert (tmp_path / "out/audits/phase_audit.json").exists()


def test_target_reliability_separates_signal_from_split_half_noise(tmp_path, monkeypatch):
    import adapt_decomp.preprocessing

    from . import audits

    # The fixture's templates live in the preprocessed domain, as calibration.emg does
    # on real records, so the stored "clean" EMG is passed through unchanged.
    monkeypatch.setattr(adapt_decomp.preprocessing, "preprocess_emg", lambda x, *a, **k: x)

    record, truth = _synthetic_record(units=2, seconds=30)
    record["calibration"].gt_matched_indices = np.arange(2)
    n = truth.shape[0]
    folder = tmp_path / "data" / "sub-01" / "clean"
    folder.mkdir(parents=True)
    times = np.empty(2, dtype=object)
    times[:] = [np.flatnonzero(truth[:, u]) for u in range(2)]
    np.savez(folder / "sub-01_FDSI_staircase_spikes.npz", spikes=times)
    np.savez(folder / "sub-01_FDSI_staircase_emg.npz", emg=record["test_emg"])
    angle = np.concatenate([np.zeros(5 * 2048), np.linspace(0, 40, n - 5 * 2048)])
    np.savez(folder / "sub-01_FDSI_staircase_angle.npz", angle=angle)
    monkeypatch.setattr(audits, "prepare", lambda *args: record)
    monkeypatch.setattr(audits, "DATA", tmp_path)
    monkeypatch.setattr(audits, "OUTPUT", tmp_path / "out")
    monkeypatch.setattr(audits, "AUDIT_CONDITIONS", ("staircase",))
    summary = audits.target_reliability(minimum_events=10)
    # The synthetic MUAPs do not change with posture: the change is noise-limited.
    assert summary["raw_target_noise_nmse"] < 0.05
    assert summary["projected_change_nmse"] < 0.05
    assert 0.3 < summary["projected_signal_to_noise_ratio_moved_bins"] < 3
    assert (tmp_path / "out/audits/target_reliability.json").exists()
