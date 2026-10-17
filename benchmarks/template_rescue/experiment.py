"""Predeclare the analytic variant; select constants without spike truth."""

import json
import shutil
from datetime import UTC, datetime

import numpy as np
import torch
from torch import nn

from benchmarks.pareto_residual.evaluate import load_models
from benchmarks.pareto_residual.experiment import checked_protocol as check_previous
from benchmarks.pareto_residual.replay import replay as previous_replay

from .common import (
    OUTPUT,
    PREVIOUS,
    ROOT,
    SUBJECTS,
    TRAIN_CONDITIONS,
    sha256,
    source_hashes,
    split,
    write_json,
)
from .data import prepare

SPEC = dict(
    methods=["pareto", "previous", "constant", "protected", "physical", "rescue"],
    primary="rescue",
    constants=[0.0, -0.075, -0.15, 0.075, 0.15],
    constant_selection="minimum label-free validation objective, zero eligible; no test RoA",
    constant_minimum_improvement_fraction=0.005,
    constant_maximum_condition_proxy_regression=0.01,
    validation_conditions=TRAIN_CONDITIONS,
    validation_snr_db=25,
    spatial_covariance_ridge_fraction=0.05,
    template_gram_ridge=0.1,
    template_slope_ridge_pseudocount=20,
    template_relative_morph_bound=0.5,
    physical_residual_rms_bound=0.25,
    minimum_template_cosine=0.05,
    calibration_cosine_quantile=0.1,
    recovery_silence_seconds=0.25,
    reference_posture_tolerance_normalized=0.025,
    reference_speed_tolerance_normalized=0.1,
    envelope_ramp_normalized=0.1,
    post_calibration_true_spikes_used_for_fit=False,
    new_neural_training=False,
    seeds="deterministic analytic update; previous fixed two-seed ensemble reused",
    success=dict(
        ramp5s_delta_pp=5,
        ramp10s_delta_pp=5,
        staircase_delta_pp=0,
        final_phase_delta_pp=0,
        final_tracked_unit_delta=0,
        global_mean_delta_pp_strictly_greater_than=0,
        snr15_delta_pp=0,
    ),
)


class ConstantGate(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = float(value)

    def forward(self, features):
        return features.new_full(features.shape[:-1], self.value)


def register():
    check_previous()
    if (OUTPUT / "protocol.json").exists():
        raise ValueError("Existing protocol is immutable; use another output for a new experiment")
    write_json(
        OUTPUT / "protocol.json",
        dict(
            registered_at_utc=datetime.now(UTC).isoformat(),
            spec=SPEC,
            source_sha256=source_hashes(),
            previous_lock_sha256=sha256(PREVIOUS / "locked.json"),
            guide_sha256=(
                sha256(ROOT / "MUAP_dynamic_DL_research_report.pdf")
                if (ROOT / "MUAP_dynamic_DL_research_report.pdf").is_file()
                else None
            ),
            folds={s: split(s) for s in SUBJECTS[1:]},
            scope="80 records, 591 calibrated instances; same comparator and scoring",
            interpretation=(
                "Exploratory: all original P1 records and outcomes were previously examined"
            ),
            untested=(
                "new independent data, real EMG, other generators, blind kinematics, full MoCA"
            ),
            missing_from_constant_selection=["outer subject", "15dB", "ramp5s", "ramp20s"],
            physical_training=(
                "online ridge sufficient statistics from past trusted predictions only"
            ),
        ),
    )
    for relative in source_hashes():
        destination = OUTPUT / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)


def checked():
    protocol = json.loads((OUTPUT / "protocol.json").read_text())
    if protocol["spec"] != SPEC or protocol["source_sha256"] != source_hashes():
        raise ValueError("Code changed after registration; preserve and start a new experiment")
    if protocol["previous_lock_sha256"] != sha256(PREVIOUS / "locked.json"):
        raise ValueError("Previous models changed")
    return protocol


@torch.no_grad()
def select():
    checked()
    if (OUTPUT / "locked.json").exists():
        raise ValueError("Selection is frozen")
    previous = json.loads((PREVIOUS / "locked.json").read_text())
    selections = {}
    for subject in SUBJECTS[1:]:
        # Integrity-check BOTH fixed network seeds of every outer fold now.
        load_models(subject, previous)
        folder = OUTPUT / "selection" / subject
        if (folder / "selected.json").exists():
            selections[subject] = json.loads((folder / "selected.json").read_text())
            continue
        partition = split(subject)
        trials = {str(value): [] for value in SPEC["constants"]}
        for condition in TRAIN_CONDITIONS:
            record = prepare(partition["validation"][0], condition, 25, physical=False)
            for value in SPEC["constants"]:
                result = previous_replay(record, [ConstantGate(value)])
                trials[str(value)].append(result["objective"])
            print("validation", subject, condition, trials, flush=True)
        means = {value: float(np.mean(values)) for value, values in trials.items()}
        best = min(means, key=means.get)
        zero = trials["0.0"]
        accepted = (
            means[best] <= means["0.0"] * (1 - SPEC["constant_minimum_improvement_fraction"])
            and max(np.asarray(trials[best]) - zero)
            <= SPEC["constant_maximum_condition_proxy_regression"]
        )
        selected = dict(
            test_subject=subject,
            validation_subject=partition["validation"][0],
            value=float(best) if accepted else 0.0,
            best_unconstrained_value=float(best),
            proxy_means=means,
            proxy_by_condition=trials,
            accepted=bool(accepted),
            selected_at_utc=datetime.now(UTC).isoformat(),
            truth_used=False,
        )
        write_json(folder / "selected.json", selected)
        selections[subject] = selected
    write_json(
        OUTPUT / "locked.json",
        dict(
            frozen_at_utc=datetime.now(UTC).isoformat(),
            protocol_sha256=sha256(OUTPUT / "protocol.json"),
            previous_lock_sha256=sha256(PREVIOUS / "locked.json"),
            selections=selections,
            selection_files={
                s: sha256(OUTPUT / "selection" / s / "selected.json") for s in SUBJECTS[1:]
            },
        ),
    )


def frozen():
    checked()
    locked = json.loads((OUTPUT / "locked.json").read_text())
    if locked["protocol_sha256"] != sha256(OUTPUT / "protocol.json"):
        raise ValueError("Frozen protocol changed")
    for subject, digest in locked["selection_files"].items():
        if digest != sha256(OUTPUT / "selection" / subject / "selected.json"):
            raise ValueError("Frozen selection changed")
    return locked
