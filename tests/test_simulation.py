import mujoco
import numpy as np
import pytest

from hand_sim import load_model


def test_model_matches_official_cpu_environment(sim):
    import myosuite

    env = myosuite.make_env("myoHandPoseFixed-v0", backend="cpu", normalize_act=False)
    try:
        upstream = env.unwrapped.model
        assert isinstance(sim.model, mujoco.MjModel)
        assert sim.model.nu == len(sim.identifiers.muscle_ids) == 39
        for field in (
            "body_mass",
            "body_inertia",
            "body_pos",
            "body_quat",
            "jnt_type",
            "jnt_range",
            "jnt_stiffness",
            "dof_damping",
            "tendon_stiffness",
            "tendon_damping",
            "actuator_gainprm",
            "actuator_biasprm",
            "actuator_dynprm",
            "actuator_ctrlrange",
            "geom_friction",
            "geom_contype",
            "geom_conaffinity",
            "eq_data",
        ):
            np.testing.assert_array_equal(getattr(sim.model, field), getattr(upstream, field))
        assert sim.model.opt.timestep == upstream.opt.timestep
        assert sim.provenance["biomechanical_modifications"] == []
    finally:
        env.close()


@pytest.mark.parametrize("excitation", [0.0, 0.02])
def test_reset_and_advance_at_least_200_steps(sim, excitation):
    initial = sim.reset()
    for _ in range(200):
        state = sim.step(np.full(sim.model.nu, excitation))
    assert state["time"] == pytest.approx(200 * sim.dt)
    for key, size in (
        ("qpos", sim.model.nq),
        ("qvel", sim.model.nv),
        ("activation", sim.model.na),
        ("actuator_force", sim.model.nu),
    ):
        assert state[key].shape == (size,)
        assert np.all(np.isfinite(state[key]))
    if excitation > 0:
        assert np.any(state["activation"] > 0)
        assert np.any(np.abs(state["actuator_force"]) > 0)
    reset = sim.reset()
    assert reset["time"] == 0
    for key in ("qpos", "qvel", "activation", "excitation"):
        np.testing.assert_array_equal(reset[key], initial[key])


def test_native_step_matches_direct_mujoco(sim):
    reference = mujoco.MjData(sim.model)
    mujoco.mj_copyData(reference, sim.model, sim.data)
    command = np.linspace(0.0, 0.1, sim.model.nu)
    reference.ctrl[:] = command
    for _ in range(20):
        sim.step(command)
        mujoco.mj_step(sim.model, reference)
        mujoco.mj_forward(sim.model, reference)
    np.testing.assert_array_equal(sim.data.qpos, reference.qpos)
    np.testing.assert_array_equal(sim.data.act, reference.act)


@pytest.mark.parametrize("command", [np.nan, np.inf, -0.1, 1.1, "wrong_shape"])
def test_invalid_control_does_not_advance_or_change_commands(sim, command):
    bad = np.ones(sim.model.nu - 1) if isinstance(command, str) else np.full(sim.model.nu, command)
    before = sim.get_state()
    with pytest.raises(ValueError):
        sim.step(bad)
    assert sim.data.time == before["time"]
    np.testing.assert_array_equal(sim.data.ctrl, before["excitation"])


def test_substeps_hold_previous_control_and_reset_is_reproducible(sim):
    command = np.full(sim.model.nu, 0.02)
    first = sim.step(command, substeps=5)
    assert first["time"] == pytest.approx(5 * sim.model.opt.timestep)
    sim.step()
    np.testing.assert_array_equal(sim.data.ctrl, command)
    sim.reset(seed=4)
    repeated = sim.step(command, substeps=5)
    np.testing.assert_array_equal(repeated["qpos"], first["qpos"])
    assert sim.provenance["seed"] == 4


@pytest.mark.parametrize("count", [0, -1, 1.5, True])
def test_invalid_substeps(sim, count):
    with pytest.raises(ValueError):
        sim.step(substeps=count)


def test_snapshot_owns_arrays(sim):
    snapshot = sim.get_state()
    qpos = snapshot["qpos"].copy()
    sim.step(np.full(sim.model.nu, 0.02))
    np.testing.assert_array_equal(snapshot["qpos"], qpos)
    snapshot["qpos"][:] = 999
    assert not np.any(sim.data.qpos == 999)


def test_context_manager_and_closed_state():
    with load_model() as sim:
        assert sim.get_state()["time"] == 0
    sim.close()
    for query in (sim.get_state, sim.step, sim.reset, sim.get_model_inventory):
        with pytest.raises(RuntimeError, match="closed"):
            query()


def test_load_yaml_configuration(tmp_path):
    config = tmp_path / "simulation.yaml"
    config.write_text("seed: 42\nsubsteps: 3\n", encoding="utf-8")
    with load_model(config) as sim:
        assert sim.seed == 42
        assert sim.dt == pytest.approx(3 * sim.model.opt.timestep)


@pytest.mark.parametrize(
    "contents", ["[]", "", "unknown_setting: 1", "substeps: 0", "env_id: invented_model"]
)
def test_reject_invalid_configuration(tmp_path, contents):
    config = tmp_path / "invalid.yaml"
    config.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError):
        load_model(config)
