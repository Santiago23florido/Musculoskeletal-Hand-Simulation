# Third-party attribution

The original code in `src/hand_sim`, `scripts` and `tests` is covered by the
repository's existing [MIT license](LICENSE), Copyright (c) 2026 Santiago Florido Gomez.
That license does not relicense the physical model or any dependency.

The MyoHand musculoskeletal model is provided by [MyoHub/myo_sim](https://github.com/MyoHub/myo_sim)
and loaded through [MyoHub/myosuite](https://github.com/MyoHub/myosuite).
MyoSuite's notice is: Copyright (c) MyoSuite Authors. All rights reserved.
Both installed distributions carry Apache License 2.0; a verbatim copy of that
license is retained in [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt).
The originals remain installed in each distribution's `.dist-info/licenses/LICENSE`.

No upstream XML, meshes, model-building code, or experimental code is vendored here.
Installation uses the published distributions unchanged, retaining their notices.
If redistributing dependencies or model assets, retain their licenses and any
asset-specific notices in addition to this repository's license.

[MuJoCo](https://github.com/google-deepmind/mujoco) is an external physics-engine
dependency with its own Apache-2.0 license and bundled third-party notices.
The dependency lock also contains transitive packages under their own licenses;
their installed license files remain authoritative.

See [docs/references.md](docs/references.md) and [CITATION.cff](CITATION.cff) for
scientific citations and the distinction between the upstream model and this code.

The README figures in `docs/images` are generated from the installed upstream
model and physical observations using `scripts/render_readme_assets.py`. The
anatomical assets retain the upstream model attribution; the rendering script is
independently developed here. These figures do not depict measured human EMG or
validated MUAP-identification results.

## External EMG comparison baselines

The optional workflow in `benchmarks/farina/` downloads `imendezguerra/adapt_decomp`
(MIT license) into the Git-ignored `external/` directory and fetches the authors'
NeuroMotion example from Zenodo (DOI: 10.5281/zenodo.22880910). The upstream code
retains its license and attribution. It is used only as a comparison baseline,
is not redistributed here, and is not imported by `hand_sim`. The tracked wrapper,
independent spike scoring, and generated result summaries are this repository's
code and measurements. See `benchmarks/farina/sources.json` and its README for
paper citations, revisions, provenance, and limits on reproduction.

The FDSI cached-reproduction workflow additionally downloads the authors' full
signals (DOI: 10.5281/zenodo.22882346) and cached outputs
(DOI: 10.5281/zenodo.22882323). Both archives and their extracted contents remain
Git-ignored. The generated comparison figure and summary derive from the authors'
predictions and simulated ground truth; they do not represent newly fitted models
or measurements on people. See `docs/benchmarks/fdsi_weeks_1_2.md` for the protocol.

The optional weeks 3–5 workflow in `benchmarks/stage_a/` independently implements
a low-rank learned whitening correction. It reuses
the original MIT-licensed `adapt_decomp` detector, calibration structures, SV/SD
updates and scorer from the clean pinned external checkout recorded in its
README. Its neural model, training wrapper and reporting code are independently
developed under this repository's MIT license. The resulting learned predictions
use the same simulated FDSI signals and supervised initial calibrations; they are
distinct from the authors' cached predictions and from the physical MyoHand
model. Experiments, limitations and outcomes are recorded in
`benchmarks/stage_a/README.md` and `docs/benchmarks/stage_a_p1_summary.json`.

The subsequent `benchmarks/pareto_residual/` experiment reads the same published
Pareto and fixed-filter source predictions and calibrations. It retains the
MIT-licensed detector and centroid updates from the pinned external checkout;
the bounded residual gate, training protocol and reporting are independent code.
It neither imports GPL generator code nor changes the provenance of the simulated
signals. Its two source views are predictions, not measured human EMG or new
ground-truth MUAP trajectories.

The follow-up `benchmarks/template_rescue/` experiment additionally estimates
multichannel templates from the same initial calibration and from past predicted
discharges. It reuses the pinned MIT `adapt_decomp` preprocessing, time extension,
detector and scorer. Spatial covariance shrinkage, analytic template synthesis,
online ridge morph statistics, recovery and fallback orchestration are independent
code. No true post-calibration MUAP trajectories, GPL generators or new human
recordings are included. The earlier neural models remain frozen.

The `benchmarks/stage_b/` workflow independently implements shared FiLM MUAP
morphing and an EMG-only GRU observer, with closed-form covariance-based filters.
It reuses the pinned MIT `adapt_decomp` preprocessing, calibration structures,
SD detector and scorer. Its supervision consists of estimated clean-EMG
spike-triggered averages on training/validation subjects, not original generator
MUAP trajectories. No NeuroMotion, BioMime or MUniverse generator code is copied
or imported. CUDA training uses a separate PyTorch environment; inference remains
in the original CPU benchmark environment. FDSI signals retain their existing
provenance and are distinct from the physical MyoHand model.

The later `stage_b_separation` and `stage_b_geometry` pilots reuse those
training/validation trajectories and implement independent covariance-aware
losses, MVDR normalization and geometric template warping. They are retained
as unpromoted validation experiments. The `benchmarks/spike_likelihood/`
repair uses the original cached Pareto and fixed sources and initial calibration
to estimate event likelihoods. Its candidate features, bounded shared classifier,
training and reporting are independent code. Only permitted training/validation
firings supervise that classifier; outer scoring retains the pinned MIT scorer.
No new generator code or original generator MUAP trajectories are included.
