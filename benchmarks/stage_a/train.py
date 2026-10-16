"""Label-free sequence training on the P1 pool; choose epochs without spike truth."""

import json
import random
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from .common import OUTPUT, POOL, ROOT, sha256, write_json
from .data import decoder, prepare
from .model import MotionCorrection, apply_correction, contrast_loss, kl_change
from .objectives import attach_constants


def prepare_pool():
    return [prepare("sub-01", condition, 30, persist=True) for condition in POOL]


def trusted_mask(sources, state, config):
    from adapt_decomp.adaptation.ops import gate_spikes_by_iqr
    from adapt_decomp.spikes.detection import find_peaks_multisource

    with torch.no_grad():
        peaks, amplitude = find_peaks_multisource(
            sources.detach(), config.spike_min_dist, config.spike_det_exp, True
        )
        threshold = 0.5 * (state.spikes_centr_cal + state.base_centr_cal)
        spikes = peaks & (amplitude > threshold)
        return gate_spikes_by_iqr(
            sources.detach(),
            spikes,
            state.Q75_cal,
            state.IQR_cal,
            gate_factor=3.0,
            peak_power=config.spike_det_exp,
            use_abs_for_detection=True,
        ).float()


def sequence_loss(
    record,
    state,
    u,
    v,
    z_state,
    start_batch,
    *,
    samples,
    stride,
    contrast_weight,
    whitening_weight=1.0,
    loss_mode="delta",
):
    x0, batch = record["z"], record["config"].batch_size
    separators = state.sep_vectors
    values, whitening, contrast = [], [], []
    for local in range(0, len(u), stride):
        index = start_batch + local
        start, stop = index * batch, min((index + 1) * batch, len(x0))
        if start < 5 * record["config"].fs:
            continue
        fifo = x0[max(0, stop - 2000) : stop]
        if samples < len(fifo):
            selection = torch.linspace(0, len(fifo) - 1, samples).long()
            fifo = fifo[selection]
        change = kl_change(fifo, u[local], v[local], record["config"].shrinkage)
        if loss_mode == "squared":
            absolute_kl = record["kl_baseline"][index] + change
            wh = ((absolute_kl - record["kl_cal_mean"]) / record["kl_cal_std"]).square()
        else:
            wh = change / x0.shape[1]
        sources = apply_correction(x0[start:stop], u[local], v[local]) @ separators.T
        mask = trusted_mask(sources, state, record["config"])
        ctr = contrast_loss(sources, mask, state.contrast_calib_mean, state.contrast_calib_std)
        whitening.append(wh)
        contrast.append(ctr)
        values.append(whitening_weight * wh + contrast_weight * ctr)
    if not values:
        return None
    smooth = (z_state[1:] - z_state[:-1]).square().mean() if len(z_state) > 1 else u.sum() * 0
    return torch.stack(values).mean() + 0.001 * smooth, dict(
        whitening_loss=float(torch.stack(whitening).mean().detach()),
        contrast=float(torch.stack(contrast).mean().detach()),
    )


def epoch(
    model,
    records,
    *,
    optimizer=None,
    samples=2000,
    stride=4,
    contrast_weight=0.01,
    whitening_weight=1.0,
    loss_mode="delta",
):
    model.train(optimizer is not None)
    losses, whs, ctrs = [], [], []
    for record in records:
        hidden = None
        state = decoder(record["calibration"], record["config"])
        for start in range(0, len(record["features"]), 32):
            inputs = record["features"][start : start + 32]
            u, v, z, hidden = model(inputs[None], hidden)
            result = sequence_loss(
                record,
                state,
                u[0],
                v[0],
                z[0],
                start,
                samples=samples,
                stride=stride,
                contrast_weight=contrast_weight,
                whitening_weight=whitening_weight,
                loss_mode=loss_mode,
            )
            if result is not None:
                loss, components = result
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                if optimizer is not None:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                losses.append(float(loss.detach()))
                whs.append(components["whitening_loss"])
                ctrs.append(components["contrast"])
            if hidden is not None:
                hidden = hidden.detach()
    return dict(
        loss=float(np.mean(losses)),
        whitening_loss=float(np.mean(whs)),
        contrast=float(np.mean(ctrs)),
    )


def train_run(
    name,
    *,
    epochs=6,
    seed=0,
    samples=2000,
    learning_rate=0.001,
    contrast_weight=0.01,
    state_dim=1,
    encoder="gru",
    resume=None,
    loss_mode="squared",
    whitening_weight=1.0,
    validation_whitening_weight=0.0001,
    validation_contrast_weight=0.1,
):
    output = OUTPUT / "runs" / name
    if (output / "history.json").exists():
        raise ValueError("Run already exists; use a new run name to preserve the experiment")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    records = prepare_pool()
    if loss_mode == "squared":
        records = [attach_constants(record) for record in records]
    # This split is wholly INSIDE the original P1 adjustment pool. The other
    # subjects/SNRs/staircase/ramp20s remain unopened until final evaluation.
    training = [r for r in records if r["metadata"]["condition"] != "triangular-ramp10s"]
    validation = [r for r in records if r["metadata"]["condition"] == "triangular-ramp10s"]
    model = MotionCorrection(state_dim=state_dim, encoder=encoder)
    if resume:
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        model = MotionCorrection(**saved["spec"])
        model.load_state_dict(saved["model"])
    else:
        features = torch.cat([r["features"] for r in training])
        model.feature_mean.copy_(features.mean(0))
        model.feature_scale.copy_(features.std(0).clamp_min(0.05))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    output.mkdir(parents=True, exist_ok=True)
    snapshot = output / "source"
    snapshot.mkdir(exist_ok=True)
    for source in Path(__file__).parent.glob("*.py"):
        shutil.copy2(source, snapshot / source.name)
    protocol = dict(
        name=name,
        seed=seed,
        epochs=epochs,
        samples=samples,
        learning_rate=learning_rate,
        contrast_weight=contrast_weight,
        whitening_weight=whitening_weight,
        validation_objective=dict(
            whitening_weight=validation_whitening_weight,
            contrast_weight=validation_contrast_weight,
        ),
        threads=torch.get_num_threads(),
        loss_mode=loss_mode,
        spec=model.spec,
        training=[r["metadata"] for r in training],
        validation=[r["metadata"] for r in validation],
        labels_used=False,
        selection="minimum label-free loss on sub-01 ramp10s, 30 dB",
        parent_checkpoint=str(resume) if resume else None,
        parent_sha256=sha256(resume) if resume else None,
        code_sha256={p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
        guide_sha256=(
            sha256(ROOT / "MUAP_dynamic_DL_research_report.pdf")
            if (ROOT / "MUAP_dynamic_DL_research_report.pdf").is_file()
            else None
        ),
        approximations=[
            "KL ridge before correction, not after",
            "BPTT truncated at 32 100-ms batches; loss every fourth batch",
            "SV held at calibration during training; original updates retained at inference",
            "log-cosh trusted peaks from detached calibration-centroid detection",
            "FIFO evenly subsampled" if samples < 2000 else "full 2000-sample FIFO",
        ],
    )
    write_json(output / "protocol.json", protocol)
    history, best = [], float("inf")
    started = time.perf_counter()
    for number in range(epochs + 1):
        train_result = None
        if number:
            random.shuffle(training)
            train_result = epoch(
                model,
                training,
                optimizer=optimizer,
                samples=samples,
                contrast_weight=contrast_weight,
                whitening_weight=whitening_weight,
                loss_mode=loss_mode,
            )
        with torch.no_grad():
            val = epoch(
                model,
                validation,
                samples=2000,
                contrast_weight=validation_contrast_weight,
                whitening_weight=validation_whitening_weight,
                loss_mode=loss_mode,
            )
        row = dict(
            epoch=number,
            training=train_result,
            validation=val,
            elapsed_seconds=time.perf_counter() - started,
        )
        history.append(row)
        print(json.dumps(row), flush=True)
        if val["loss"] < best:
            best = val["loss"]
            torch.save(
                dict(
                    spec=model.spec,
                    model=model.state_dict(),
                    epoch=number,
                    selection_loss=best,
                    labels_used=False,
                ),
                output / "best.pt",
            )
        torch.save(
            dict(
                spec=model.spec,
                model=model.state_dict(),
                epoch=number,
                selection_loss=val["loss"],
                labels_used=False,
            ),
            output / "last.pt",
        )
        write_json(
            output / "history.json",
            dict(executed_at_utc=datetime.now(UTC).isoformat(), protocol=protocol, epochs=history),
        )
    return output / "best.pt"
