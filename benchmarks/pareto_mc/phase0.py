"""Record a concrete, bounded attempt to use the official NeuroMotion backend."""

import shutil
import subprocess

from .common import OUTPUT, ROOT, now, sha256, write_new


def run():
    checkout = ROOT / "external/pareto_mc_muniverse"
    probes = {}
    commands = [
        ["docker", "info", "--format", "{{.ServerVersion}}"],
        ["docker", "run", "--rm", "--pull", "never", "pranavm19/muniverse:neuromotion", "true"],
        [
            "wsl",
            "-d",
            "Ubuntu",
            "--",
            "sh",
            "-lc",
            "command -v singularity; command -v apptainer; command -v docker",
        ],
    ]
    for command in commands:
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=30)
            probes[" ".join(command)] = dict(
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            probes[" ".join(command)] = dict(error=str(error))
    report = dict(
        attempted_at_utc=now(),
        muniverse_commit=subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
        ).strip(),
        official_backend="pranavm19/muniverse:neuromotion",
        dockerfile_sha256=sha256(checkout / "docker/neuromotion/Dockerfile"),
        launcher_sha256=sha256(checkout / "src/muniverse/datasets/_generate_recording.sh"),
        probes=probes,
        singularity_on_windows=shutil.which("singularity"),
        generated_new_fdsi_records=False,
        conclusion="No running Docker Linux engine; no Singularity/Apptainer in Ubuntu. "
        "Official backend cannot start in the available environment. "
        "No substitute line-source data are called FDSI confirmation.",
        result_scope="exploratory only",
        new_subject_seeds_reserved=[210601, 210602, 210603],
    )
    write_new(OUTPUT / "phase0.json", report)
    print(report["conclusion"], flush=True)


if __name__ == "__main__":
    run()
