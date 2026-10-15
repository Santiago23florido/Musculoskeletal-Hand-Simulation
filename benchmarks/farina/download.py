"""Fetch the authors' Zenodo archives; verify before extracting. No model dependencies."""

import argparse
import hashlib
import http.client
import json
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RECORDS = {"tutorial": "22880910", "fdsi-data": "22882346", "fdsi-outputs": "22882323"}
CHUNK = 32 * 1024 * 1024


def digest(path, algorithm):
    result = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def download(name, destination, workers=6):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(f"https://zenodo.org/api/records/{RECORDS[name]}", timeout=60) as r:
        record = json.load(r)
    provenance = {"record_id": record["id"], "doi": record.get("doi"), "files": []}
    for entry in record["files"]:
        filename = entry["key"]
        if Path(filename).name != filename:
            raise ValueError("Archive name must be a filename")
        archive = destination / filename
        algorithm, expected = entry["checksum"].split(":", 1)
        verified = (
            archive.exists()
            and archive.stat().st_size == entry["size"]
            and digest(archive, algorithm) == expected
        )
        if not verified:
            parts = destination / (filename + ".parts")
            parts.mkdir(exist_ok=True)

            def fetch(start, entry=entry, parts=parts):
                end = min(entry["size"], start + CHUNK) - 1
                part = parts / str(start)
                if part.exists() and part.stat().st_size == end - start + 1:
                    return part
                for attempt in range(4):
                    try:
                        offset = part.stat().st_size if part.exists() else 0
                        if offset > end - start + 1:
                            raise ValueError(f"Oversized partial download: {part}")
                        if offset == end - start + 1:
                            return part
                        resume_start = start + offset
                        request = urllib.request.Request(
                            entry["links"]["self"], headers={"Range": f"bytes={resume_start}-{end}"}
                        )
                        with urllib.request.urlopen(request, timeout=90) as response:
                            if response.status != 206:
                                raise ValueError("Server must support HTTP byte ranges")
                            if response.headers["Content-Range"] != (
                                f"bytes {resume_start}-{end}/{entry['size']}"
                            ):
                                raise ValueError("Unexpected byte range")
                            with part.open("ab") as handle:
                                for block in iter(lambda: response.read(1024 * 1024), b""):
                                    handle.write(block)
                        if part.stat().st_size != end - start + 1:
                            raise ValueError("Truncated byte range")
                        return part
                    except (OSError, ValueError, http.client.HTTPException):
                        if attempt == 3:
                            raise
                        time.sleep(2 ** (attempt + 1))

            starts = list(range(0, entry["size"], CHUNK))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(fetch, start) for start in starts]
                for count, future in enumerate(as_completed(futures), 1):
                    future.result()
                    print(f"{name}: {count}/{len(starts)} chunks", flush=True)
            with archive.open("wb") as handle:
                for start in starts:
                    with (parts / str(start)).open("rb") as part:
                        for block in iter(lambda: part.read(4 * 1024 * 1024), b""):
                            handle.write(block)
            if digest(archive, algorithm) != expected:
                raise ValueError(f"Checksum mismatch: {archive}; remove its .parts and retry")
            for start in starts:
                (parts / str(start)).unlink()
            parts.rmdir()
        print(f"{name}: checksum verified, extracting", flush=True)
        with zipfile.ZipFile(archive) as zipped:
            for member in zipped.infolist():
                target = (destination / member.filename).resolve()
                if not target.is_relative_to(destination):
                    raise ValueError("Unsafe archive path")
            zipped.extractall(destination)
        provenance["files"].append(
            {"filename": filename, "size": entry["size"], "checksum": entry["checksum"]}
        )
    (destination / f"{name}_download.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    return provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", choices=RECORDS)
    parser.add_argument("--dest", type=Path, default=ROOT / "data/benchmarks/farina")
    parser.add_argument("--workers", type=int, default=6, choices=range(1, 33))
    args = parser.parse_args()
    download(args.archive, args.dest, args.workers)
