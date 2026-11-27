"""Unit tests for benchmarks.pareto_mc.embc_cluster."""

from __future__ import annotations

import numpy as np
import torch

from benchmarks.pareto_mc.context_c4 import ContextLikelihood
from benchmarks.pareto_mc.embc_cluster import (
    DeviceContextHead,
    attach_component_timers,
    make_nomem_settings,
    probe_environment,
)
from benchmarks.pareto_mc.memory_loop import MemoryConfig


class _DummyOwner:
    def __init__(self, units: int):
        class _Cfg:
            batch_size = 204
            ext_fact = 10

        self.config = _Cfg()
        self.current_index = 0
        self.state_trace = [[0.1, -2.0, 0.0]]
        self.last_event = {
            "features": np.zeros((4, 22), dtype=np.float32),
            "time": np.array([10, 40, 80, 120], dtype=np.int64),
            "unit": np.array([0, 0, 1, 1], dtype=np.int64),
            "primary": np.array([True, False, True, False], dtype=bool),
        }


def test_probe_environment_returns_expected_keys():
    info = probe_environment()
    assert "platform" in info
    assert "torch" in info
    assert "tools" in info
    assert isinstance(info["neuromotion_container_runtime_available"], bool)


def test_make_nomem_settings_disables_beta_only():
    base = {
        "bins": 8,
        "h": 0.03,
        "threshold": 0.66,
        "kappa": 0.18,
        "beta_max": 0.61,
        "momentum": 0.95,
        "calibration_only": True,
    }
    nomem = make_nomem_settings(base)
    assert nomem["beta_max"] == 0.0
    assert base["beta_max"] == 0.61
    cfg = MemoryConfig(**nomem)
    assert cfg.beta_max == 0.0


def test_device_context_head_matches_cpu_reference():
    torch.manual_seed(0)
    model_ref = ContextLikelihood("setrec").eval()
    model_dev = ContextLikelihood("setrec").eval()
    model_dev.load_state_dict(model_ref.state_dict())

    owner = _DummyOwner(units=2)
    features = torch.from_numpy(owner.last_event["features"])
    primary = torch.from_numpy(owner.last_event["primary"])

    from benchmarks.pareto_mc.closed_context import ContextHead

    ref_head = ContextHead(model_ref, owner, units=2)
    cpu_head = DeviceContextHead(model_dev, owner, units=2, device="cpu")
    out_ref = ref_head(features, primary)
    out_cpu = cpu_head(features, primary)
    assert torch.allclose(out_ref, out_cpu, atol=1e-6)


def test_attach_component_timers_records_calls():
    class _FakeHead:
        def is_identity(self):
            return False

        def __call__(self, x):
            return x

    class _FakeAdapter:
        def __init__(self):
            self.head = _FakeHead()
            self.collector = None

        def _preprocess_batch_raw(self, x):
            return x

        def _whiten(self, x):
            return x

        def _detect_spikes(self, x):
            return x

        def decide(self, x):
            return self.head(x)

        def _update_spike_det(self, x):
            return x

        def _update_sep_vectors(self, x):
            return x

    adapter = _FakeAdapter()
    stages = attach_component_timers(adapter, sync_cuda=False)
    adapter._preprocess_batch_raw(1)
    adapter._whiten(1)
    adapter._detect_spikes(1)
    adapter.decide(1)
    adapter._update_spike_det(1)
    adapter._update_sep_vectors(1)
    for key in (
        "preprocess_ms",
        "whiten_ms",
        "detect_ms",
        "decide_ms",
        "head_ms",
        "update_sd_ms",
        "update_sv_ms",
    ):
        assert len(stages[key]) == 1
        assert stages[key][0] >= 0.0
