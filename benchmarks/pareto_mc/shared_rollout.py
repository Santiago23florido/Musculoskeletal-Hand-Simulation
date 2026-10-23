"""Reuse identical causal whitening/C1 work across simultaneous search trajectories.

SV, centroids, detector FIFOs, C4, and memories remain independent. Final gate
predictions and deployment timing always use the complete ordinary loop.
"""

import copy
import time
from types import MethodType

import numpy as np
import torch

from .backbone import build
from .memory_loop import make_memory_adapter
from .observer import EMGFeatures, load_observer
from .train_likelihood import load_model


class SharedFeatures:
    def __init__(self, context):
        self.context = context

    def step(self, x):
        return self.context["features"]


class SharedObserver:
    def __init__(self, context):
        self.context = context

    def __call__(self, features, hidden):
        return self.context["state"]


def cached_preprocess(adapter, raw):
    return adapter.shared_context["x"], adapter.shared_context["pad"]


def cached_whiten(adapter, x, index):
    context = adapter.shared_context
    # Assert the cache key at the point of use, including exact batch centering.
    if index != context["index"] or not torch.equal(x, context["centered"]):
        raise ValueError("Shared whitening requested for a different batch")
    adapter.decomp.whitening = context["whitening"]
    return context["z"], context["coupling"]


@torch.no_grad()
def shared_replay(raw, master, adapters, *, observer=None, extractor=None, collect_sources=False):
    context = {}
    for adapter in adapters:
        adapter.shared_context = context
        adapter._preprocess_batch_raw = MethodType(cached_preprocess, adapter)
        adapter._whiten = MethodType(cached_whiten, adapter)
        if observer is not None:
            adapter.extractor, adapter.observer = SharedFeatures(context), SharedObserver(context)
    hidden = None
    outputs = [dict(spikes=[], sources=[], centroid=[], trusted_count=[]) for _ in adapters]
    durations = []
    batch = master.config.batch_size
    for index, start in enumerate(range(0, len(raw), batch)):
        tick = time.perf_counter()
        block = raw[start : start + batch]
        x, pad = master._preprocess_batch_raw(block)
        centered = x - x.mean(0, keepdim=True)
        z, coupling = master._whiten(centered, index)
        context.update(
            index=index,
            x=x,
            pad=pad,
            centered=centered,
            z=z,
            coupling=coupling,
            whitening=master.decomp.whitening,
        )
        if observer is not None:
            features = extractor.step(centered)
            mean, logvar, hidden = observer(features[None, None], hidden)
            context.update(features=features, state=(mean, logvar, hidden))
        for adapter, output in zip(adapters, outputs, strict=True):
            spikes, sources = adapter.process_batch(block, index)
            output["spikes"].append(spikes.numpy().astype(np.uint8))
            if collect_sources:
                output["sources"].append(sources.numpy())
            output["centroid"].append(
                torch.stack([adapter.decomp.base_centr, adapter.decomp.spikes_centr]).numpy()
            )
            output["trusted_count"].append(adapter.last_trusted.numpy())
        durations.append((time.perf_counter() - tick) * 1000)
    for adapter, output in zip(adapters, outputs, strict=True):
        output["spikes"] = np.concatenate(output["spikes"])
        if collect_sources:
            output["sources"] = np.concatenate(output["sources"])
        else:
            output.pop("sources")
        for key in ("centroid", "trusted_count"):
            output[key] = np.asarray(output[key])
        output.update(
            state=np.asarray(adapter.state_trace),
            memory=np.asarray(adapter.memory_trace),
            group_batch_ms=np.asarray(durations),
        )
    return outputs


def prepare(record, runs):
    """One calibration, several heads/memories, no shared adapting detector state."""
    from .common import OUTPUT
    from .memory_loop import MemoryConfig

    config = copy.deepcopy(record["config"])
    config.max_sigma_batches = 1
    master = build(record["calibration"], config)
    adapters = []
    for selection, settings in runs:
        adapter = make_memory_adapter(
            record,
            load_model(OUTPUT / selection["checkpoint"]),
            memory_config=MemoryConfig(**settings),
            **selection["selection"]["parameters"],
        )
        adapter.collector = False
        adapters.append(adapter)
    return master, adapters, load_observer(), EMGFeatures(record["calibration"], config)
