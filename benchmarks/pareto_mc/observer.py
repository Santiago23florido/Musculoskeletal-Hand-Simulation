"""C1 features from current/past EMG, with supervised targets loaded separately."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import torch

from .backbone import AdaptDecomp
from .common import OUTPUT, TRAIN, VALIDATION, guard, now, records, sha256, stub, write_new
from .data import load_angle, load_record
from .models import BlindState, gaussian_nll


class EMGFeatures:
    """The Stage B 100 RMS + 16 covariance + 4 previous-source features.

    A calibration-only normalizer and PCA frame are fixed before deployment.
    No full-record means, labels or future features are used.
    """

    @torch.no_grad()
    def __init__(self, calibration, config):
        from adapt_decomp.preprocessing import extend_data

        self.config = config
        w0 = torch.from_numpy(calibration.whitening).float()
        self.fixed = w0.T @ torch.from_numpy(calibration.sep_vectors).float()
        emg = torch.from_numpy(calibration.emg).float()
        x_cal = extend_data(emg, config.ext_fact, ext_mode=config.ext_mode)
        axes = torch.linalg.eigh(w0)[1][:, :16]
        projected = (x_cal - x_cal.mean(0)) @ axes
        self.projection = axes / projected.square().mean(0).sqrt().clamp_min(1e-6)
        self.channel_scale = emg.square().mean(0).sqrt().clamp_min(1e-6)
        self.source_scale = max(
            float(torch.from_numpy(calibration.sources).square().mean().sqrt()), 1e-6
        )
        self.previous = torch.zeros(4)
        values = []
        for start in range(0, len(x_cal), config.batch_size):
            block = x_cal[start : start + config.batch_size]
            values.append(self.raw(block - block.mean(0)))
        values = torch.stack(values)
        self.mean = values.mean(0)
        self.scale = values.std(0).clamp_min(0.1)
        self.previous = torch.zeros(4)

    def raw(self, block):
        rms = block[:, :100].square().mean(0).sqrt().clamp_min(1e-6)
        x = block @ self.projection
        spectrum = torch.linalg.eigvalsh(x.T @ x / len(x)).clamp_min(1e-6).log()
        result = torch.cat([(rms / self.channel_scale).log(), spectrum, self.previous])
        sources = (block @ self.fixed) / self.source_scale
        self.previous = torch.stack(
            [
                sources.abs().median(),
                sources.abs().quantile(0.95),
                sources.square().mean().sqrt(),
                (sources.abs() > 3).float().mean(),
            ]
        )
        if result.shape != (120,):
            raise ValueError("FDSI requires 100 raw channels and 120 observer features")
        return result

    def step(self, block):
        return ((self.raw(block) - self.mean) / self.scale).clamp(-6, 6)


def features_for_record(record):
    preprocessing = AdaptDecomp.__new__(AdaptDecomp)
    preprocessing.config = copy.deepcopy(record["config"])
    preprocessing.decomp = SimpleNamespace(zi=None, ext_fifo=None, ema_mean_online=None)
    extractor = EMGFeatures(record["calibration"], record["config"])
    result = []
    for start in range(0, len(record["raw"]), record["config"].batch_size):
        x, _ = preprocessing._preprocess_batch_raw(
            record["raw"][start : start + record["config"].batch_size]
        )
        result.append(extractor.step(x - x.mean(0)))
    return torch.stack(result)


def prepare(subject, condition, snr, *, phase):
    guard(subject, phase)
    folder = OUTPUT / "C1/features"
    name = stub(subject, condition, snr)
    path = folder / f"{name}.npz"
    if path.exists():
        with np.load(path) as archive:
            return torch.from_numpy(archive["features"]), torch.from_numpy(archive["target"])
    record = load_record(subject, condition, snr, phase=phase)
    features = features_for_record(record)
    # Angle appears only as a supervised target, never in features_for_record.
    angle = load_angle(record, phase=phase)
    b = record["config"].batch_size
    target = np.asarray(
        [angle[i : i + b].mean() / 40 for i in range(0, len(angle), b)], dtype=np.float32
    )
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, features=features.numpy(), target=target)
    write_new(
        folder / f"{name}.json",
        dict(
            prepared_at_utc=now(),
            raw_sha256=record["raw_sha256"],
            calibration_sha256=record["calibration_sha256"],
            cache_sha256=sha256(path),
            feature_inputs="EMG plus initial calibration only",
            angle_role="training target only",
        ),
    )
    return features, torch.from_numpy(target)


def train(seed=0, epochs=30):
    folder = OUTPUT / "C1" / f"seed{seed}"
    if (folder / "complete.json").exists():
        raise ValueError("Observer training already complete")
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("Use .venv-stage-b-gpu for observer training")
    torch.manual_seed(seed)
    training = [prepare(*case, phase="training") for case in records(TRAIN)]
    validation = [prepare(*case, phase="validation") for case in records(VALIDATION)]
    model = BlindState().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    rng = np.random.default_rng(seed)
    history, best = [], float("inf")
    folder.mkdir(parents=True, exist_ok=True)
    for epoch in range(epochs):
        model.train()
        losses = []
        for i in rng.permutation(len(training)):
            features, target = [v.to(device) for v in training[i]]
            mean, logvar, _ = model(features[None])
            loss = gaussian_nll(mean, logvar, target[None])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            values, rmse = [], []
            for features, target in validation:
                mean, logvar, _ = model(features[None].to(device))
                values.append(float(gaussian_nll(mean, logvar, target[None].to(device))))
                rmse.append(float((mean.cpu()[0] - target).square().mean()))
        score = float(np.mean(values))
        history.append(
            dict(
                epoch=epoch,
                train_nll=float(np.mean(losses)),
                val_nll=score,
                val_rmse_deg=float(np.sqrt(np.mean(rmse)) * 40),
            )
        )
        if score < best:
            best = score
            torch.save(
                dict(spec=model.spec, model=model.cpu().state_dict(), epoch=epoch),
                folder / "observer.pt",
            )
            model.to(device)
        print("C1", seed, history[-1], flush=True)
    write_new(folder / "history.json", history)
    write_new(
        folder / "complete.json",
        dict(
            completed_at_utc=now(),
            model_sha256=sha256(folder / "observer.pt"),
            validation_nll=best,
            training_subjects=TRAIN,
            validation_subjects=VALIDATION,
            seed=seed,
            gpu=torch.cuda.get_device_name(),
            epochs=epochs,
        ),
    )


def load_observer(seed=0):
    path = OUTPUT / "C1" / f"seed{seed}" / "observer.pt"
    meta = json.loads(path.with_name("complete.json").read_text(encoding="utf-8"))
    if sha256(path) != meta["model_sha256"]:
        raise ValueError("Observer checkpoint changed")
    saved = torch.load(path, weights_only=True, map_location="cpu")
    model = BlindState(**saved["spec"])
    model.load_state_dict(saved["model"])
    return model.eval()
