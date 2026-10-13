import mujoco
import numpy as np
import pytest

from hand_sim.identifiers import ModelIdentifiers
from hand_sim.sensors import PhysicalSensors


def test_joint_state_and_limits_match_native_data(sim):
    sim.step(np.full(sim.model.nu, 0.02), substeps=20)
    for item in sim.get_model_inventory()["joints"]:
        joint = sim.get_joint_state(item["name"])
        np.testing.assert_array_equal(joint["position"], sim.data.qpos[item["qpos_indices"]])
        np.testing.assert_array_equal(joint["velocity"], sim.data.qvel[item["qvel_indices"]])
        np.testing.assert_array_equal(joint["anchor_world"], sim.data.xanchor[item["id"]])
        if item["limited"]:
            np.testing.assert_array_equal(joint["limits"], sim.model.jnt_range[item["id"]])


def test_muscle_activation_address_and_actuator_measurements(sim):
    sim.step(np.full(sim.model.nu, 0.02), substeps=20)
    for item in sim.get_model_inventory()["muscles"]:
        i = item["id"]
        state = sim.get_muscle_state(item["name"])
        np.testing.assert_array_equal(state["activation"], sim.data.act[state["activation_indices"]])
        assert state["excitation"] == sim.data.ctrl[i]
        for field in ("actuator_force", "actuator_length", "actuator_velocity"):
            assert state[field] == getattr(sim.data, field)[i]
            assert np.isfinite(state[field])


def test_tendon_lengths_velocities_and_qualified_force_contributions(sim):
    sim.step(np.full(sim.model.nu, 0.02), substeps=20)
    for item in sim.get_model_inventory()["tendons"]:
        state = sim.get_tendon_state(item["name"])
        assert state["length"] == sim.data.ten_length[item["id"]]
        assert state["velocity"] == sim.data.ten_velocity[item["id"]]
        assert state["total_tendon_force"] is None
        assert state["unavailable_reason"]
        assert state["actuator_force_contributions"]
        for contribution in state["actuator_force_contributions"]:
            a = contribution["actuator_id"]
            assert contribution["signed_force"] == pytest.approx(
                sim.data.actuator_force[a] * sim.model.actuator_gear[a, 0])


def test_all_anatomical_sites_and_body_orientations_are_queryable(sim):
    for site in sim.get_model_inventory()["sites"]:
        np.testing.assert_array_equal(sim.get_site_position(site["name"]),
                                      sim.data.site_xpos[site["id"]])
    for item in sim.get_model_inventory()["bodies"]:
        body = sim.get_body_state(item["name"])
        np.testing.assert_array_equal(body["position_world"], sim.data.xpos[item["id"]])
        assert np.linalg.norm(body["quaternion_world_wxyz"]) == pytest.approx(1)


def test_fingertip_velocity_matches_position_finite_difference(sim):
    sim.step(np.full(sim.model.nu, 0.02), substeps=30)
    fingertips = sim.sensors.fingertips()
    reference = mujoco.MjData(sim.model)
    mujoco.mj_copyData(reference, sim.model, sim.data)
    epsilon = 1e-7
    mujoco.mj_integratePos(sim.model, reference.qpos, reference.qvel, epsilon)
    mujoco.mj_forward(sim.model, reference)
    for name, tip in fingertips.items():
        difference = (reference.site_xpos[tip["id"]] - tip["position_world"]) / epsilon
        np.testing.assert_allclose(sim.get_site_velocity(name), difference, rtol=1e-4, atol=1e-6)


def test_original_hand_has_no_native_sensors(sim):
    assert sim.model.nsensor == sim.model.nsensordata == 0
    assert sim.get_state()["native_sensor_data"].shape == (0,)
    assert sim.sensors.capabilities()["native_sensor_count"] == 0
    with pytest.raises(KeyError):
        sim.get_sensor_data("invented_sensor")


def test_registered_sensor_slicing_on_separate_metadata_fixture():
    # Test actual MuJoCo sensor dimensions without adding sensors to the hand.
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body><joint name="hinge"/><geom type="sphere" size=".1"/>
        <site name="point" pos="0 0 .1"/></body></worldbody>
      <sensor><jointpos name="angle" joint="hinge"/>
        <framepos name="position" objtype="site" objname="point"/></sensor></mujoco>''')
    data = mujoco.MjData(model)
    data.qpos[:] = 0.2
    mujoco.mj_forward(model, data)
    sensors = PhysicalSensors(model, data, ModelIdentifiers(model))
    assert sensors.native_sensor("angle").shape == (1,)
    assert sensors.native_sensor("position").shape == (3,)
    np.testing.assert_allclose(sensors.native_sensor("angle"), [0.2])
    np.testing.assert_allclose(sensors.native_sensor("position"), data.site_xpos[0])


def test_contact_wrench_on_separate_collision_fixture():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom name="floor" type="plane" size="1 1 .1"/>
      <body pos="0 0 .04"><freejoint/>
        <geom name="sphere" type="sphere" size=".05"/></body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    sensors = PhysicalSensors(model, data, ModelIdentifiers(model))
    contacts = sensors.contacts()
    assert len(contacts) == data.ncon > 0
    for record in contacts:
        expected = np.empty(6)
        mujoco.mj_contactForce(model, data, record["id"], expected)
        assert record["constraint_active"]
        np.testing.assert_array_equal(record["wrench_contact_frame_force_then_torque"], expected)
        assert expected[0] > 0
        assert record["frame_world_to_contact"].shape == (3, 3)


def test_unknown_anatomical_names_raise(sim):
    for query in (sim.get_joint_state, sim.get_muscle_state, sim.get_tendon_state,
                  sim.get_site_position, sim.get_site_velocity, sim.get_body_state):
        with pytest.raises(KeyError):
            query("not_a_native_name")
