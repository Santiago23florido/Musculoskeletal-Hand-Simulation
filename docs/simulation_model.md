# Physical model and observation contract

## Source and load path

`hand_sim.load_model()` calls the installed `myosuite.make_env` with
`myoHandPoseFixed-v0`, `backend="cpu"`, and `normalize_act=False`.
MyoSuite 3.0.0 builds its official `hand_pose` recipe using myo-sim 0.2.3.
The wrapper uses the returned `env.unwrapped.model`, `data` and `tip_sids` directly.
No local XML copy, model extension or replacement actuator is supplied.

This upstream environment supplies existing pose-target visual sites and fixed
scaffolding as part of its recipe. Those sites are included honestly in the
inventory; `*_target` sites are display references, not anatomical fingertips.
The package never uses task rewards, success criteria or Gym episode termination.
The environment's native reset is used, while stepping calls MuJoCo directly.
Its presence is a model-loading adapter, not a new research task implementation.

Upstream masses, inertias, joint ranges, damping, tendon routing, force laws,
activation time constants, contact settings, solver settings and integrator remain
unchanged. Tests compare important compiled arrays with a fresh upstream load.
Upstream composition and conversion choices remain the responsibility of the
model authors; this repository makes no independent physiological validation claim.

## Coordinates and units

All positions are in the loaded model's world frame, in meters. Body quaternions
use MuJoCo's `w, x, y, z` convention. Joint `anchor_world` is `MjData.xanchor`;
joint generalized position is `qpos`, a distinct quantity. All current joints are
hinges, so their coordinates are radians and velocities radians per second.
Lengths are meters, translational velocities meters per second, forces newtons,
torques newton-meters and time seconds. Excitation and muscle activation are
dimensionless native quantities. Actuator length/velocity use transmission
coordinates and need not equal an independently measured anatomical fiber length.

Joint widths are selected from `jnt_type`: free joints have 7 position coordinates
and 6 velocity DOFs, ball joints 4 and 3, and hinge/slide joints 1 and 1.
Addresses come from `jnt_qposadr` and `jnt_dofadr`. Muscle activation comes from
`actuator_actadr` and `actuator_actnum`, independently of the actuator ID.

## Available observations

| Quantity | Source | Interpretation |
| --- | --- | --- |
| Joint positions/velocities | `data.qpos`, `data.qvel` | Slices selected using joint metadata |
| Joint limits | `model.jnt_limited`, `model.jnt_range` | `None` for an unlimited joint |
| Joint spatial location | `data.xanchor` | World-frame joint anchor |
| Excitation commands | `data.ctrl` | Native commands, not activation states |
| Activation states | `data.act` | State slice; `None` if an actuator has none |
| Actuator forces | `data.actuator_force` | Native signed scalar transmission force |
| Actuator length/velocity | `data.actuator_length/velocity` | Transmission-space state |
| Tendon length/velocity | `data.ten_length/velocity` | Native tendon path state |
| Tendon actuator contribution | `gear[0] * actuator_force` | Derived signed contribution for each tendon transmission |
| Total per-tendon tension | Unavailable | `None`; contributions are not total tension |
| Body positions/orientations | `data.xpos`, `xquat`, `xmat` | World frame, body origin rather than COM |
| Site/landmark positions | `data.site_xpos` | All upstream named sites are queryable |
| Fingertip velocity | `mj_jacSite` multiplied by `qvel` | Derived world-frame translational velocity |
| Contacts | `data.contact[:ncon]` | Current collisions; may be empty |
| Contact wrench | `mj_contactForce` | Six values: force then torque, contact frame |
| Registered sensors | `sensor_adr/dim` slices of `sensordata` | This loaded environment has none |

`mj_forward` runs after stepping so kinematics, forces and returned coordinates
refer to the same final state. It does not advance time. All public observation
arrays are copied. If you edit native MuJoCo data directly, call `mj_forward`
yourself to refresh dependent fields before queries.

Contact frames are stored as a 3-by-3 world-to-contact rotation with basis vectors
as rows. The first contact axis is the normal. A non-constraint contact has no
solved wrench; its wrench is `None`. Actuator forces may include the model's
passive muscle force contribution and do not separately identify active/passive
fiber forces. A tendon may have multiple actuator contributions; no unexposed
passive, constraint or total force is invented or replaced with zero.

## Reproducibility and limitations

The default seed is zero. The configuration can set `seed` and the positive integer
`substeps`; unsupported settings and alternative environments are rejected.
The official fixed-pose initialization is preserved, not the target pose from a
published experiment. Low constant excitation is only a mechanical smoke check.

`get_state()` includes owned arrays of coordinates, velocities, accelerations,
commands, activations, forces, tendon state, body/site poses, joint anchors and
native sensor data. `save_states` stacks those observations and serializes metadata
as JSON in a `.npz` archive without pickle. These are debugging snapshots; they
omit solver warm starts and other fields needed for a full exact restart.

Pinned versions and a fixed seed support repeatability; floating-point bitwise
identity across operating systems or hardware is not promised. Native Windows is
the primary validated runtime. Visualization requires a working desktop/OpenGL
context and is tested separately from headless physics.

No additional sensors, anatomical sites, artificial neural measurements, learned
estimators or scientific algorithms are introduced. The independent code observes
the original physics and leaves the future research methodology unspecified.
