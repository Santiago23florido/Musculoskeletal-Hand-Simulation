# Musculoskeletal Hand Simulation

![Official MyoHand bones and muscle-tendon paths rendered in MuJoCo, with 23 joint DOFs, 39 muscle actuators and five fingertips](docs/images/myohand-overview.png)

[![Validate physical simulation](https://github.com/Santiago23florido/Musculoskeletal-Hand-Simulation/actions/workflows/validate.yml/badge.svg)](https://github.com/Santiago23florido/Musculoskeletal-Hand-Simulation/actions/workflows/validate.yml)
![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB)
![MuJoCo 3.16.0](https://img.shields.io/badge/MuJoCo-3.16.0-55d9cf)
![MyoSuite 3.0.0](https://img.shields.io/badge/MyoSuite-3.0.0-ff7b80)

A small, independent Python foundation for loading and observing the **official
MyoSuite MyoHand model** in MuJoCo. Windows native is the primary development and
validated runtime; Ubuntu under WSL2 is also supported.

The original model supplies all bodies, joint constraints, tendon paths, muscle
force laws, activation dynamics and contact parameters. This repository adds
anatomical identification, physical-state access, NumPy debugging archives and a
standard MuJoCo viewer. It contains no research controller, training algorithm,
EMG synthesis or experimental task definition.

## Research direction

The long-term aim is **robust identification of motor unit action potentials
(MUAPs) from EMG during hand movement**, including movement-related variability.
This repository establishes the reproducible musculoskeletal simulation and
anatomical observations needed to support that independent research direction.
The MUAP-identification methodology will be defined separately; the current
implementation provides physical state and geometry without generating EMG,
identifying MUAPs or claiming robustness results.

![Physical measurements from a 200-step native simulation: excitation, activation, actuator forces, joint coordinates and fingertip displacement](docs/images/physical-observations.png)

Both figures come from the actual pinned model and a constant-excitation smoke
check. The geometry, muscle-tendon paths and plotted quantities are simulation
outputs. Regenerate the images on a desktop with OpenGL using
`uv run --locked python scripts/render_readme_assets.py`. The script changes only
rendering options (lighting and offscreen resolution); it preserves biomechanics.

## Install on Windows (PowerShell)

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), for example:

```powershell
winget install --id=astral-sh.uv -e
# Reopen PowerShell after installing uv, then enter the repository.
$env:UV_PROJECT_ENVIRONMENT = '.venv-windows'
uv sync --locked --python 3.12
```

Python 3.12 is downloaded by uv if needed. No system Python changes, NVIDIA GPU,
CUDA toolkit or separate MuJoCo download are required. Keep the environment
variable set in each PowerShell session so Windows uses `.venv-windows` and WSL
can use its separate `.venv`. The scripts below run from the repository root.
After installing uv, `powershell -ExecutionPolicy Bypass -File scripts/setup_windows.ps1 -Validate`
also creates and verifies the native environment; the policy change applies only
to that PowerShell process. The script locates uv on `PATH` or in `.local/bin`.

```powershell
uv run --locked python scripts/run_simulation.py --steps 200 --output outputs/smoke.npz
uv run --locked python scripts/inspect_model.py --output outputs/model_inventory.json
uv run --locked python scripts/inspect_model.py --joint mcp2_flexion_r --muscle FDS2_r --site IFtip_r
uv run --locked python scripts/visualize_hand.py
```

For the already installed native environment, these commands also work without
activating a virtual environment or putting uv on `PATH`:

```powershell
.\.venv-windows\Scripts\python.exe scripts/run_simulation.py --steps 200
.\.venv-windows\Scripts\python.exe scripts/visualize_hand.py
```

The headless demo holds a constant native excitation of `0.02` on all muscle
actuators to check that physics advances. It does not implement a movement task.
The viewer uses the same simulation and holds its reset commands by default.
Drag to rotate, scroll to zoom and close the window to stop. View options expose
native geometry groups and tendons. Attachment-site markers are initially hidden
to keep hundreds of sites from obscuring the hand. A desktop with an OpenGL-capable
driver is needed for visualization; headless execution does not open a window.

## Python API

```python
import numpy as np
from hand_sim import load_model
from hand_sim.state import save_states

with load_model("configs/simulation.yaml") as sim:
    initial = sim.reset()
    current = sim.step(np.full(sim.model.nu, 0.02))
    joint = sim.get_joint_state("mcp2_flexion_r")
    muscle = sim.get_muscle_state("FDS2_r")
    fingertip = sim.get_site_position("IFtip_r")
    velocity = sim.get_site_velocity("IFtip_r")
    inventory = sim.get_model_inventory()
    save_states("outputs/query.npz", [initial, current], metadata=sim.provenance)
```

`load_model()` returns a context manager with `reset()`, `step()`, `get_state()`,
`get_joint_state(name)`, `get_muscle_state(name)`, `get_tendon_state(name)`,
`get_site_position(name)`, `get_site_velocity(name)`, `get_body_state(name)`,
`get_contacts()`, `get_sensor_data(name)`, `get_model_inventory()` and `close()`.
Queries return owned copies. Invalid names raise `KeyError`. The native MuJoCo
objects are available as `sim.model` and `sim.data`; call `mujoco.mj_forward` after
direct state changes before querying dependent spatial or force quantities.

`step()` accepts a finite vector of length `model.nu`, within the loaded native
control limits (`[0, 1]` for this model). Passing no vector holds the previous
commands. It advances one native 0.002-second step by default; `substeps` changes
the number of native steps, preserving the upstream integrator and timestep.
The standard Gym environment has a different control interval and normalized
actions; this API steps `mujoco.mj_step` directly and does not compute rewards or
episode termination. `reset()` uses the official environment's reset configuration.

## Anatomy and measurements

The pinned environment contains 38 bodies (including world and fixed upstream
scaffolding), 23 hinge joints, 39 muscle actuators, 39 tendons, 496 sites and **zero
registered MuJoCo sensors**. `nq = nv = 23`, `na = nu = 39`. The upstream `tip_sids`
identify `THtip_r`, `IFtip_r`, `MFtip_r`, `RFtip_r`, and `LFtip_r`.

```powershell
uv run --locked python scripts/inspect_model.py --list joints
uv run --locked python scripts/inspect_model.py --list muscles
uv run --locked python scripts/inspect_model.py --list tendons
uv run --locked python scripts/inspect_model.py --list sites
uv run --locked python scripts/inspect_model.py --list sensors
```

The inventory includes IDs, joint types, independent `qpos`/`qvel` indices,
activation-state addresses, actuator transmission IDs, limits and sensor dimensions.
No assumption equates a joint ID with a generalized coordinate index.
The complete checked-in [name list](docs/anatomical_names.md) and
[JSON inventory](docs/model_inventory.json) were generated from this pinned model.

Joint coordinates, commands, activation states, actuator forces, actuator/tendon
lengths and velocities, and body/site poses are read from native model/data fields.
Site velocity is derived with MuJoCo's site Jacobian. Contact wrenches come from
`mj_contactForce` in the contact frame. `get_sensor_data(name)` reads only actual
registered sensor channels; on this hand every sensor name raises `KeyError`.

Total tendon tension and separate muscle-fiber force are not exposed as measurement
channels. The API returns available **signed actuator force contributions** for
tendon transmissions and `None` with a reason for total tendon force. Empty native
sensor arrays and an empty contact list mean no channels or current contacts;
they are not synthetic zero measurements. No sites or sensors were added.
See [the physical model and measurement details](docs/simulation_model.md).

Saved `.npz` arrays have a leading sample dimension and can be loaded using
`np.load("outputs/smoke.npz", allow_pickle=False)`. `metadata_json` is a scalar JSON
string. Archives contain observations, not complete restart checkpoints.

## Validation

```powershell
uv run --locked pytest -q
uv run --locked ruff check src scripts tests
uv run --locked ruff format --check src scripts tests
$env:HAND_SIM_TEST_GRAPHICS = '1'
uv run --locked pytest -q -m graphics
Remove-Item Env:HAND_SIM_TEST_GRAPHICS
```

The suite compares biomechanical arrays with a separately loaded official
environment, advances at least 200 steps, checks reset and input validation,
validates every named object, tests sensor slicing, verifies fingertip velocities
against finite differences and round-trips JSON/NumPy exports. Separate small
MuJoCo fixtures exercise free/ball joint metadata, nonempty sensors and contacts;
they never replace the physical hand. The graphics test is skipped unless explicitly
enabled. Tests keep temporary files under `.pytest_cache/tmp` to avoid system-temp
permission conflicts on Windows.
The [local validation record](docs/validation.md) documents 47 passing tests on
Windows with graphics enabled and the separately checked headless WSL2 path.

## Ubuntu under WSL2 (alternative)

Run these in Ubuntu's Bash terminal, with uv installed there:

```bash
sudo apt-get update
sudo apt-get install -y libgl1 libglfw3 libegl1 libosmesa6
uv sync --locked --python 3.12
uv run --locked python scripts/run_simulation.py --steps 200
uv run --locked pytest -q
uv run --locked python scripts/visualize_hand.py
```

WSLg on a compatible Windows desktop supplies the graphical display. Otherwise an
X server and a working `DISPLAY` are needed. Do not set `MUJOCO_GL=egl` or `osmesa`
for the interactive GLFW viewer. Headless physics needs no display or GPU. Keep
Windows and Linux virtual environments separate; for faster WSL installs, clone
on the Linux filesystem and use `uv sync --link-mode copy` when needed.

## Sources, licensing and scope

The model is imported from [MyoSuite](https://github.com/MyoHub/myosuite) 3.0.0 and
[myo-sim](https://github.com/MyoHub/myo_sim) 0.2.3, with MuJoCo 3.16.0. Python 3.12
and all resolved dependencies are captured in `.python-version` and `uv.lock`.
The implementation was validated on native Windows with Python 3.12.14.

Cite [Caggiano et al. (2022)](https://arxiv.org/abs/2205.13600) for MyoSuite.
[Hodossy and Farina (2026)](https://arxiv.org/abs/2603.07364) is acknowledged as a
related application, with no replication of its methodology. See
[references](docs/references.md), [CITATION.cff](CITATION.cff) and
[third-party notices](THIRD_PARTY_NOTICES.md). The existing MIT license covers this
repository's independent code; upstream components retain their own licenses.

The initial Git commit is preserved. At the repository owner's request, the new
implementation commits use retrospective author/committer dates starting on
1 October 2026. Implementation and validation were performed on 15 October 2026;
the commit dates do not establish historical execution or validation dates.
