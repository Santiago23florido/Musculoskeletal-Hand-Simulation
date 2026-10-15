# Local validation record

Validation was performed on **15 October 2026**. The retrospective Git dates
requested by the repository owner are independent of this execution date.

## Native Windows (primary)

| Component | Validated version |
| --- | --- |
| OS | Windows 11 Home, 64-bit, 10.0.26200 |
| Python | CPython 3.12.14 |
| uv | 0.12.21 |
| MyoSuite | 3.0.0 |
| myo-sim | 0.2.3 |
| MuJoCo | 3.16.0 |
| NumPy | 2.5.4 |
| Environment | `myoHandPoseFixed-v0`, CPU |

The full suite with `HAND_SIM_TEST_GRAPHICS=1` completed with **47 passed**.
This includes an actual standard MuJoCo passive viewer launch, physics steps,
window synchronization and clean shutdown. The separate two-second interactive
smoke run also completed successfully. Without the opt-in graphics variable,
the suite reports **46 passed, 1 skipped**.

`ruff check src scripts tests` passed and `ruff format --check src scripts tests`
reported all 16 Python files formatted. `git diff --check` passed.
`cffconvert --validate -i CITATION.cff` validated the citation against CFF 1.2.0.

The headless 200-step demonstration used constant excitation `0.02`, advanced
0.4000000000000003 simulated seconds at 0.002 seconds per step, returned 23 position
coordinates and 39 activation states, and produced finite actuator forces.
The optional NumPy archive was written and tested with pickle disabled.

The complete inventory and name lists were exported into
[model_inventory.json](model_inventory.json) and
[anatomical_names.md](anatomical_names.md). The loaded dimensions were:

| Quantity | Count |
| --- | --- |
| Bodies, including world and fixed scaffolding | 38 |
| Joints / position coordinates / velocity DOFs | 23 / 23 / 23 |
| Muscle actuators / activation states | 39 / 39 |
| Tendons | 39 |
| Sites, including upstream target and attachment sites | 496 |
| Fingertip sites from upstream `tip_sids` | 5 |
| Native sensor channels / data dimension | 0 / 0 |

The tests exercise every named object in those categories. Additional small
MuJoCo fixtures verify free/ball joint widths, activation indices that differ
from actuator IDs, native sensor slices and solved contact wrenches. They are
test fixtures and are never used as substitute hand models.

The two README images were rendered locally with MuJoCo and Matplotlib. Geometry
comes from the official model; plotted values come from its actual 200-step
smoke run. These plots are physical diagnostics, not EMG or MUAP results.

## WSL2 alternative

Ubuntu under WSL2 with CPython 3.12.3 and the same dependency lock completed
**46 passed, 1 skipped** with `DISPLAY` and `WAYLAND_DISPLAY` unset. The skipped
test was the deliberately opt-in viewer test. This confirms the physics API can
run without a display. The final graphical suite was validated on native Windows.

## Repeat the primary checks

From the repository root in PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_windows.ps1 -Validate
$env:HAND_SIM_TEST_GRAPHICS = '1'
.\.venv-windows\Scripts\python.exe -m pytest -q
Remove-Item Env:HAND_SIM_TEST_GRAPHICS
.\.venv-windows\Scripts\ruff.exe check src scripts tests
.\.venv-windows\Scripts\ruff.exe format --check src scripts tests
```

The GitHub Actions workflow repeats headless checks on Windows and Ubuntu after
pushes and pull requests. Its remotely reported status is separate from the local
results recorded here.

Test temporary files use the top-level ignored `.pytest_tmp` directory, which
pytest creates on a clean checkout. It does not depend on a pre-existing cache
directory or on writable system temporary directories.
