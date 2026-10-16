"""Freeze a candidate before any out-of-pool spike-truth evaluation."""

import shutil
from datetime import UTC, datetime

import torch

from .common import OUTPUT, ROOT, sha256, write_json
from .model import MotionCorrection
from .objectives import attach_constants
from .train import epoch


@torch.no_grad()
def select_model(names):
    from .data import prepare

    destination = OUTPUT / "selection.json"
    if destination.exists():
        raise ValueError(
            "Selection is already frozen; preserve it and do not tune against the test"
        )
    validation = attach_constants(prepare("sub-01", "triangular-ramp10s", 30, persist=True))
    results = []
    for name in names:
        path = OUTPUT / "runs" / name / "best.pt"
        saved = torch.load(path, map_location="cpu", weights_only=True)
        model = MotionCorrection(**saved["spec"])
        model.load_state_dict(saved["model"])
        score = epoch(
            model,
            [validation],
            samples=2000,
            loss_mode="squared",
            whitening_weight=0.0001,
            contrast_weight=0.1,
        )
        results.append(
            dict(
                run=name,
                epoch=saved["epoch"],
                checkpoint_sha256=sha256(path),
                common_validation=score,
                spec=saved["spec"],
            )
        )
        print(name, score, flush=True)
    best = min(results, key=lambda r: r["common_validation"]["loss"])
    shutil.copy2(OUTPUT / "runs" / best["run"] / "best.pt", OUTPUT / "selected.pt")
    write_json(
        destination,
        dict(
            frozen_at_utc=datetime.now(UTC).isoformat(),
            candidates=results,
            selected=best,
            checkpoint_sha256=sha256(OUTPUT / "selected.pt"),
            post_calibration_ground_truth_used=False,
            validation="sub-01, ramp10s, 30dB; within P1 adjustment pool",
            objective="0.0001 * normalized squared KL + 0.1 * squared contrast + smoothness",
            primary_inference="reference SV/SD with reference first-order whitening/SV coupling",
            ablations=["same network without coupling", "identity correction with reference SV/SD"],
            guide_sha256=(
                sha256(ROOT / "MUAP_dynamic_DL_research_report.pdf")
                if (ROOT / "MUAP_dynamic_DL_research_report.pdf").is_file()
                else None
            ),
        ),
    )
    print("Frozen candidate:", best["run"], flush=True)
