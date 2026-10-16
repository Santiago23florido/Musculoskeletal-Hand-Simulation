"""Mathematical restrictions, transferable representation, and split isolation."""

import numpy as np
import pytest
import torch

from .common import SUBJECTS, split
from .model import ResidualGate, correct, waveform_loss
from .replay import duplicate_loss

torch.set_num_threads(2)


def test_initial_head_is_exact_identity_with_nonzero_learning_signal():
    torch.manual_seed(7)
    model = ResidualGate()
    x = torch.randn(4, 10)
    base = torch.randn(50, 4)
    alternative = torch.randn_like(base)
    a = model(x)
    assert torch.equal(correct(base, alternative, a), base)
    loss = correct(base, alternative, a).square().mean()
    loss.backward()
    assert model.output.weight.grad.abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_shared_head_and_correction_are_unit_permutation_equivariant():
    torch.manual_seed(3)
    model = ResidualGate()
    with torch.no_grad():
        model.output.weight.normal_()
    features = torch.randn(9, 10)
    p, f = torch.randn(60, 9), torch.randn(60, 9)
    permutation = torch.randperm(9)
    expected = correct(p, f, model(features))[:, permutation]
    actual = correct(p[:, permutation], f[:, permutation], model(features[permutation]))
    torch.testing.assert_close(actual, expected)


def test_augmented_transform_has_certified_bound_and_inverse():
    torch.manual_seed(8)
    for units in (1, 3, 17):
        a = torch.linspace(-0.15, 0.15, units, dtype=torch.float64)
        eye = torch.eye(units, dtype=torch.float64)
        matrix = torch.cat(
            [
                torch.cat([eye - torch.diag(a), torch.diag(a)], 1),
                torch.cat([torch.zeros_like(eye), eye], 1),
            ],
            0,
        )
        assert (
            torch.linalg.matrix_norm(matrix - torch.eye(units * 2), ord=2) <= 2**0.5 * 0.15 + 1e-12
        )
        assert torch.linalg.svdvals(matrix).min() >= 1 - 2**0.5 * 0.15
        x = torch.randn(7, units * 2, dtype=torch.float64)
        recovered = torch.linalg.solve(matrix, (x @ matrix.T).T).T
        torch.testing.assert_close(recovered, x)


@pytest.mark.parametrize("test", SUBJECTS[1:])
def test_no_subject_overlap_and_pareto_tuning_subject_is_training_only(test):
    partition = split(test)
    a, b, c = map(set, [partition["training"], partition["validation"], partition["test"]])
    assert not (a & b or a & c or b & c)
    assert a | b | c == set(SUBJECTS)
    assert "sub-01" in a


def test_source_shape_loss_has_correct_zero_and_differentiable_mismatch():
    x = torch.zeros(71, 2, dtype=torch.float64)
    x[35] = torch.tensor([1.0, 2.0])
    template = torch.zeros(2, 21, dtype=x.dtype)
    template[:, 10] = 1
    trusted = torch.zeros_like(x, dtype=torch.bool)
    trusted[35] = True
    assert waveform_loss(x, trusted, template) == 0
    assert waveform_loss(-x, trusted, template) == 0
    changed = x.clone()
    changed[34] = 0.3
    changed.requires_grad_()
    loss = waveform_loss(changed, trusted, template)
    loss.backward()
    assert loss > 0 and torch.isfinite(changed.grad).all()


def test_duplicate_penalty_identifies_collapsed_sources():
    x = torch.linspace(-1, 1, 100)
    repeated = torch.stack([x, x], 1)
    assert duplicate_loss(repeated) > 0.99
    assert duplicate_loss(repeated[:, :1]) == 0


def test_head_capacity_does_not_depend_on_number_of_units():
    model = ResidualGate()
    assert sum(p.numel() for p in model.parameters()) == 97
    for units in (1, 5, 30):
        assert model(torch.zeros(units, 10)).shape == (units,)
    assert np.isclose(model.spec["bound"], 0.15)
