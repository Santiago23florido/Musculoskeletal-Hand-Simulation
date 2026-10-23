# MUAP geometry warping pilot

**This pilot stopped during training and validation, before outer-test prediction.**
After validation ruled out unrestricted CNN refinement, the next experiment
warped the initial template using 240 shared parameters. The output preserves
identity at q=0, maps zero input to zero output, and scales with input amplitude.
It uses no shared additive pattern or parameters tied to unit identifiers.

The operator shifts and stretches time, then shifts, scales, and rotates the
spatial grid. Its eight parameters are bounded functions of q, q², q³, and ten
energy moments of the calibration template. It uses deterministic trilinear
interpolation and bounds the correction relative to the template's own norm.
Tests check identity, amplitude homogeneity, unit permutation, and finite CUDA
gradients.

When projected loss dominates, selection by source risk retains the identity.
A separate pilot with source risk as the dominant objective reduces validation
risk from 0.7151 to 0.6498 in six epochs, while waveform and projected errors
worsen. Those weights were discarded. No outer-test RoA was measured, so the
lower risk does not establish that discharges were recovered.

Protocols, source snapshots, histories, and pilot outputs are retained in
`outputs/stage_b_geometry/`. The subsequent experiment evaluates discharge
decisions directly in [spike_likelihood](../spike_likelihood/README.md), using
event labels and threshold selection by validation Jaccard. Its results are in
the [published summary](../../docs/benchmarks/spike_likelihood_summary.json).
