import json

import mujoco
import pytest

from hand_sim.identifiers import ModelIdentifiers


@pytest.mark.parametrize(
    "kind,plural",
    [
        ("body", "bodies"),
        ("joint", "joints"),
        ("actuator", "actuators"),
        ("tendon", "tendons"),
        ("site", "sites"),
        ("sensor", "sensors"),
    ],
)
def test_all_named_identifiers_roundtrip(sim, kind, plural):
    inventory = sim.get_model_inventory()
    for entry in inventory[plural]:
        assert sim.identifiers.name(kind, entry["id"]) == entry["name"]
        if entry["name"] is not None:
            assert sim.identifiers.resolve(kind, entry["name"]) == entry["id"]
    with pytest.raises(KeyError):
        sim.identifiers.resolve(kind, "does_not_exist")


def test_hand_joint_indices_cover_generalized_coordinates(sim):
    inventory = sim.get_model_inventory()
    assert [q for joint in inventory["joints"] for q in joint["qpos_indices"]] == list(
        range(sim.model.nq)
    )
    assert [v for joint in inventory["joints"] for v in joint["qvel_indices"]] == list(
        range(sim.model.nv)
    )
    for joint in inventory["joints"]:
        assert joint["dofs"] == len(joint["qvel_indices"])
        assert joint["type"] == "mjJNT_HINGE"


def test_free_ball_and_hinge_use_different_coordinate_widths():
    # A separate metadata fixture: it is never substituted for the official hand.
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <body><freejoint name="free"/><geom type="sphere" size=".1"/></body>
      <body pos="1 0 0"><joint name="ball" type="ball"/>
        <geom type="sphere" size=".1"/>
        <body pos="0 0 1"><joint name="hinge"/><geom type="sphere" size=".1"/></body>
      </body></worldbody></mujoco>""")
    ids = ModelIdentifiers(model)
    entries = ids.inventory()["joints"]
    assert entries[0]["qpos_indices"] == list(range(7))
    assert entries[0]["qvel_indices"] == list(range(6))
    assert entries[1]["qpos_indices"] == list(range(7, 11))
    assert entries[1]["qvel_indices"] == list(range(6, 9))
    assert entries[2]["qpos_indices"] == [11]
    assert entries[2]["qvel_indices"] == [9]


def test_export_inventory_and_upstream_fingertips(sim, tmp_path):
    path = sim.identifiers.export(tmp_path / "anatomy" / "model.json", metadata=sim.provenance)
    inventory = json.loads(path.read_text(encoding="utf-8"))
    assert inventory == sim.get_model_inventory()
    assert {site["name"] for site in inventory["fingertips"]} == {
        "THtip_r",
        "IFtip_r",
        "MFtip_r",
        "RFtip_r",
        "LFtip_r",
    }
    assert len(inventory["muscles"]) == sim.model.nu
    assert len(inventory["tendons"]) == sim.model.ntendon
    assert len(inventory["sites"]) == sim.model.nsite
    assert inventory["sensors"] == []


def test_invalid_ids_are_rejected(sim):
    for value in (-1, sim.model.njnt):
        with pytest.raises(IndexError):
            sim.identifiers.joint_slices(value)
    with pytest.raises(ValueError):
        ModelIdentifiers(sim.model, (-1,))
