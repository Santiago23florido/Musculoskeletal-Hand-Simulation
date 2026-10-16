"""Combine disjoint subject partitions of the SAME frozen evaluation."""

import ast
import hashlib
import json
import time

from .common import OUTPUT, ROOT, sha256, write_json
from .evaluate import finalize_results, write_csv


def evaluation_core_sha256(path):
    """Permit a summary-only repair; require identical prediction and scoring code."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [
        node
        for node in tree.body
        if not isinstance(node, ast.FunctionDef) or node.name != "summarize"
    ]
    return hashlib.sha256(ast.dump(tree).encode()).hexdigest()


def merge_evaluations(names, destination="p1_complete"):
    started = time.perf_counter()
    plans, rows, timings, sources, cores = [], [], [], [], []
    for name in names:
        folder = OUTPUT / "evaluations" / name
        plan = json.loads((folder / "plan.json").read_text())
        plans.append(plan)
        evaluator = folder / "source/evaluate.py"
        if sha256(evaluator) != plan["evaluator_code_sha256"]:
            raise ValueError("Evaluation source snapshot differs from its frozen plan")
        cores.append(evaluation_core_sha256(evaluator))
        paths = sorted((folder / "records").glob("*.json"))
        expected = len(plan["subjects"]) * len(plan["conditions"]) * len(plan["snrs"])
        if len(paths) != expected:
            raise ValueError("Partition is incomplete")
        for path in paths:
            result = json.loads(path.read_text())
            rows.extend(result["rows"])
            timings.extend(result["timings"])
        sources.append(
            dict(
                name=name,
                plan_sha256=sha256(folder / "plan.json"),
                evaluator_code_sha256=plan["evaluator_code_sha256"],
                record_sha256={path.name: sha256(path) for path in paths},
            )
        )
    if len(set(cores)) != 1:
        raise ValueError("Partitions used different prediction or scoring code")
    execution_sources = {}
    for filename in ("data.py", "model.py", "inference.py"):
        hashes = {sha256(OUTPUT / "evaluations" / name / "source" / filename) for name in names}
        if len(hashes) != 1:
            raise ValueError(f"Partitions used different {filename}")
        execution_sources[filename] = hashes.pop()
    if execution_sources["inference.py"] != plans[0]["inference_code_sha256"]:
        raise ValueError("Inference snapshot differs from its frozen plan")
    for key in (
        "checkpoint_sha256",
        "methods",
        "conditions",
        "snrs",
        "inference_code_sha256",
        "threads",
    ):
        if any(p[key] != plans[0][key] for p in plans):
            raise ValueError(f"Partitions used different {key}")
    subjects = [subject for p in plans for subject in p["subjects"]]
    if sorted(subjects) != [f"sub-0{i}" for i in range(1, 6)]:
        raise ValueError("Five disjoint subject partitions are required")
    selection = json.loads((OUTPUT / "selection.json").read_text())
    if selection["checkpoint_sha256"] != plans[0]["checkpoint_sha256"]:
        raise ValueError("Evaluation does not use the preselected checkpoint")
    plan = {
        **plans[0],
        "subjects": subjects,
        "partitions": sources,
        "selection_sha256": sha256(OUTPUT / "selection.json"),
        "evaluation_core_sha256": cores[0],
        "execution_source_sha256": execution_sources,
        "reporter_code_sha256": sha256(ROOT / "benchmarks/stage_a/evaluate.py"),
    }
    output = OUTPUT / "evaluations" / destination
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "plan.json", plan)
    write_csv(output / "per_unit.csv", rows)
    return finalize_results(rows, timings, plan, output, time.perf_counter() - started)
