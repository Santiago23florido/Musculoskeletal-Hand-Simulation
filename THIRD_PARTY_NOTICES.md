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
