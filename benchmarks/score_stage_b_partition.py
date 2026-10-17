"""Schedule one subject through the frozen scorer without changing its mathematics.

All 320 predictions must already be sealed. Four disjoint subject processes can
then evaluate their records concurrently; the original report still requires all
80 records and the same 591 instances per method. This changes only the outer
iteration list, never models, predictions, metric implementation or thresholds.
"""

import argparse
from datetime import UTC, datetime

import torch

from benchmarks.stage_b.common import OUTPUT, SUBJECTS, reference, sha256, write_json
from benchmarks.stage_b.experiment import checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subject", choices=SUBJECTS[1:])
    args = parser.parse_args()
    torch.set_num_threads(2)
    checked()
    reference()
    if not (OUTPUT / "scoring_started.json").exists():
        raise ValueError("Start the serial scorer once to validate the seal and create its marker")
    from benchmarks.stage_b import evaluate

    # score() iterates SUBJECTS[1:]. Restrict only that outer iteration; it still
    # validates the entire sealed manifest and invokes the frozen original scorer.
    evaluate.SUBJECTS = (SUBJECTS[0], args.subject)
    evaluate.score()
    paths = sorted((OUTPUT / "scored").glob(f"{args.subject}_*.json"))
    if len(paths) != 20:
        raise ValueError("Subject scoring is incomplete")
    write_json(
        OUTPUT / f"scoring_partition_{args.subject}.json",
        dict(
            completed_at_utc=datetime.now(UTC).isoformat(),
            subject=args.subject,
            records=20,
            runner_sha256=sha256(__file__),
            frozen_scorer_sha256=sha256(evaluate.__file__),
            prediction_manifest_sha256=sha256(OUTPUT / "predictions_complete.json"),
            score_files={str(p.relative_to(OUTPUT)): sha256(p) for p in paths},
            scope="disjoint outer loop only; frozen metric/model/thresholds unchanged",
        ),
    )


if __name__ == "__main__":
    main()
