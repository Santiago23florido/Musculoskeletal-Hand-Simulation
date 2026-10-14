# Scientific sources and attribution

This is an independent simulation infrastructure project. MyoSuite and myo-sim
provide the musculoskeletal model and its biomechanics. This repository provides
the Python observation API, metadata inventory, debugging archives, viewer wrapper,
documentation and validation tests.

The authoritative upstream sources are the [MyoSuite repository](https://github.com/MyoHub/myosuite),
[official documentation](https://myosuite.readthedocs.io/en/latest/), and
[myo-sim model repository](https://github.com/MyoHub/myo_sim). The installed versions
are pinned; the latest online documentation may describe later releases.

## Primary model citation

Caggiano, V., Wang, H., Durandau, G., Sartori, M., and Kumar, V. (2022).
*MyoSuite -- A contact-rich simulation suite for musculoskeletal motor control*.
Learning for Dynamics and Control, PMLR 168:492–507.
[arXiv:2205.13600](https://arxiv.org/abs/2205.13600).

```bibtex
@inproceedings{Caggiano2022MyoSuite,
  title = {MyoSuite -- A contact-rich simulation suite for musculoskeletal motor control},
  author = {Caggiano, Vittorio and Wang, Huawei and Durandau, Guillaume and Sartori, Massimo and Kumar, Vikash},
  booktitle = {Learning for Dynamics and Control},
  year = {2022},
  pages = {492--507},
  url = {https://arxiv.org/abs/2205.13600}
}
```

Also acknowledge the installed software versions: MyoSuite 3.0.0 and myo-sim 0.2.3.
The upstream [hand model notes](https://github.com/MyoHub/myo_sim/blob/main/myo_sim/models/arm/README_hand.md)
describe its MoBL and 2nd-Hand ancestry and conversion adjustments. Those are
upstream choices; this project does not retune their parameters.

## Related application, cited for context only

Hodossy, B. K. and Farina, D. (2026). *Neural Control and Learning of Simulated
Hand Movements With an EMG-Based Closed-Loop Interface*.
[arXiv:2603.07364](https://arxiv.org/abs/2603.07364).

This reference acknowledges a related application of hand simulation. This
repository does not reproduce that publication's methodology, EMG synthesis,
reinforcement learning, experiments, results or scientific objectives. No such
algorithms are implemented or inferred for this independent project.

Machine-readable citations are in [CITATION.cff](../CITATION.cff). Licensing is
documented in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
