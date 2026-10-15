"""Exercise resumption and checksum enforcement without network or large files."""

import hashlib
import io
import json
import zipfile

import download as downloader
import pytest


def archive_bytes():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("dataset/example.txt", "verified example")
    return buffer.getvalue()


class Response(io.BytesIO):
    status = 206

    def __init__(self, payload, headers=None):
        super().__init__(payload)
        self.headers = headers or {}


def fake_server(monkeypatch, payload, *, disconnect=False, checksum=None):
    calls = []
    record = {
        "id": 22882346,
        "doi": "10.5281/zenodo.22882346",
        "files": [
            {
                "key": "example.zip",
                "size": len(payload),
                "checksum": checksum or "md5:" + hashlib.md5(payload).hexdigest(),
                "links": {"self": "https://example.invalid/archive"},
            }
        ],
    }

    class Interrupted(Response):
        def read(self, size=-1):
            if self.tell():
                raise OSError("Connection interrupted")
            return super().read(20)

    def open_url(request, timeout):
        if isinstance(request, str):
            return Response(json.dumps(record).encode())
        start, end = map(int, request.headers["Range"].removeprefix("bytes=").split("-"))
        calls.append((start, end))
        response_class = Interrupted if disconnect and len(calls) == 1 else Response
        return response_class(
            payload[start : end + 1], {"Content-Range": f"bytes {start}-{end}/{len(payload)}"}
        )

    monkeypatch.setattr(downloader, "CHUNK", len(payload))
    monkeypatch.setattr(downloader.urllib.request, "urlopen", open_url)
    monkeypatch.setattr(downloader.time, "sleep", lambda _: None)
    return calls


def test_resume_existing_partial_range_and_verify_before_extraction(tmp_path, monkeypatch):
    payload = archive_bytes()
    calls = fake_server(monkeypatch, payload)
    parts = tmp_path / "example.zip.parts"
    parts.mkdir()
    (parts / "0").write_bytes(payload[:20])
    downloader.download("fdsi-data", tmp_path, workers=1)
    assert calls == [(20, len(payload) - 1)]
    assert (tmp_path / "example.zip").read_bytes() == payload
    assert (tmp_path / "dataset/example.txt").read_text() == "verified example"
    assert not parts.exists()


def test_retry_keeps_bytes_received_before_disconnect(tmp_path, monkeypatch):
    payload = archive_bytes()
    calls = fake_server(monkeypatch, payload, disconnect=True)
    downloader.download("fdsi-data", tmp_path, workers=1)
    assert calls == [(0, len(payload) - 1), (20, len(payload) - 1)]
    assert (tmp_path / "example.zip").read_bytes() == payload


def test_checksum_failure_prevents_extraction(tmp_path, monkeypatch):
    payload = archive_bytes()
    fake_server(monkeypatch, payload, checksum="md5:" + "0" * 32)
    with pytest.raises(ValueError, match="Checksum mismatch"):
        downloader.download("fdsi-data", tmp_path, workers=1)
    assert not (tmp_path / "dataset").exists()
