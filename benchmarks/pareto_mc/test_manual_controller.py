import pytest
import torch

from .manual_controller import ManualController


def test_five_parameter_baseline_starts_at_identity():
    controller = ManualController()
    wh, sv, beta = controller(torch.randn(8, 9))
    assert len(controller.spec) == 5
    assert wh == 1 and torch.all(sv == 1) and torch.all(beta == 0)


def test_manual_gains_increase_with_drift_and_mixing_with_memory_confidence():
    controller = ManualController(a_wh=100, a_sv=80, b=10, c=0.5, beta_max=0.8)
    x = torch.zeros(8, 9)
    x[:, 4] = torch.linspace(0, 2, 8)
    x[:, 6] = torch.linspace(0, 1, 8)
    wh, sv, beta = controller(x)
    assert 1 < wh <= 4
    assert torch.all(sv[1:] >= sv[:-1]) and torch.all(sv <= 4)
    assert torch.all(beta[1:] > beta[:-1]) and torch.all(beta <= 0.8)
    short = controller(x, dt=0.01)[1]
    assert torch.all(short <= sv)
    with pytest.raises(ValueError):
        ManualController(beta_max=-0.01)
