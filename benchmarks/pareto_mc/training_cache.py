"""Lossless, guarded training caches; no inference inputs or model decisions change."""

import hashlib
import json
from functools import cache
from types import SimpleNamespace

import numpy as np

from .common import OUTPUT, guard, now, sha256, write_new


@cache
def verified_archive(path, size, mtime_ns):
    return sha256(path)


def cached_labelled(path, case, phase, loader):
    guard(case[0], phase)
    relative = path.relative_to(OUTPUT).as_posix()
    key = hashlib.sha256(relative.encode()).hexdigest()[:24]
    folder = OUTPUT / "L1/training_cache" / key
    meta_path = folder / "metadata.json"
    stat = path.stat()
    digest = verified_archive(path, stat.st_size, stat.st_mtime_ns)
    if not meta_path.exists():
        events = loader(path, case, phase)
        folder.mkdir(parents=True, exist_ok=True)
        keys = [name for name in events if isinstance(events[name], np.ndarray)]
        arrays = {name: events[name] for name in keys}
        counts = np.array([len(ref) for ref in events["truth"]], dtype=np.int64)
        arrays["truth_times"] = np.concatenate(events["truth"])
        arrays["truth_edges"] = np.r_[0, np.cumsum(counts)]
        for name, values in arrays.items():
            np.save(folder / f"{name}.npy", values)
        record = events["record"]
        cfg = record["config"]
        write_new(
            meta_path,
            dict(
                created_at_utc=now(),
                case=case,
                phase=phase,
                source=relative,
                source_sha256=digest,
                arrays=list(arrays),
                arrays_sha256={name: sha256(folder / f"{name}.npy") for name in arrays},
                samples=len(record["raw"]),
                units=len(events["truth"]),
                fs=cfg.fs,
                batch_size=cfg.batch_size,
                distance=cfg.spike_min_dist,
                gt_indices=np.asarray(record["calibration"].gt_matched_indices).tolist(),
                lossless=True,
                label_origin="calibration lag; guarded train/validation subjects only",
            ),
        )
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if metadata["source_sha256"] != digest or tuple(metadata["case"]) != tuple(case):
        raise ValueError("Training cache source was changed")
    if metadata["phase"] != phase:
        raise ValueError("Cannot reuse labels in a different subject role")
    for name, expected in metadata["arrays_sha256"].items():
        array_path = folder / f"{name}.npy"
        array_stat = array_path.stat()
        if verified_archive(array_path, array_stat.st_size, array_stat.st_mtime_ns) != expected:
            raise ValueError("Training cache array changed")
    arrays = {name: np.load(folder / f"{name}.npy", mmap_mode="c") for name in metadata["arrays"]}
    edges, times = arrays.pop("truth_edges"), arrays.pop("truth_times")
    arrays["truth"] = [times[a:b] for a, b in zip(edges[:-1], edges[1:], strict=True)]
    arrays["record"] = dict(
        subject=case[0],
        condition=case[1],
        snr_db=case[2],
        raw=np.empty((metadata["samples"], 0)),
        config=SimpleNamespace(
            fs=metadata["fs"],
            batch_size=metadata["batch_size"],
            spike_min_dist=metadata["distance"],
        ),
        calibration=SimpleNamespace(
            spikes=np.empty((0, metadata["units"])),
            gt_matched_indices=np.asarray(metadata["gt_indices"]),
        ),
    )
    return arrays
