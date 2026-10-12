import pytest

from hand_sim import load_model


@pytest.fixture
def sim():
    with load_model(seed=0) as simulation:
        yield simulation
