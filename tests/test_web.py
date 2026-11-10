"""Tests for the web app; the real audit is replaced, so no model is needed."""
import sys
import time
from pathlib import Path

import pymupdf
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import auditor  # noqa: E402
import auditor_web  # noqa: E402

FILES = [("files", ("a.txt", b"Termin 7 dni.")), ("files", ("b.txt", b"Termin 3 dni."))]
HOSTILE = "<script>alert(1)</script>"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(auditor_web, "AUDITS_DIR", tmp_path)
    return TestClient(auditor_web.app)


def fake_result(findings=None, warnings=None):
    return {
        "documents": [{"name": "a.txt", "pages": 1}, {"name": "b.txt", "pages": 1}],
        "findings": findings or [],
        "warnings": warnings or [],
        "pairs_checked": 1,
        "rejected_unverified": 0,
        "model": "test-model",
    }


def finish(client, audit_id):
    for _ in range(200):
        audit = next(a for a in client.get("/api/audits").json() if a["id"] == audit_id)
        if audit["status"] in ("done", "failed"):
            return audit
        time.sleep(0.02)
    raise AssertionError("the audit didn't finish")


def test_needs_two_files(client):
    assert client.post("/api/audits", files=FILES[:1]).status_code == 400


def test_rejects_unsupported_format(client):
    files = [FILES[0], ("files", ("virus.exe", b"x"))]
    response = client.post("/api/audits", files=files)
    assert response.status_code == 400
    assert "virus.exe" in response.json()["detail"]


def test_rejects_duplicate_names(client):
    assert client.post("/api/audits", files=[FILES[0], FILES[0]]).status_code == 400


def test_path_in_filename_cannot_escape_upload_folder(client, monkeypatch):
    seen = {}

    def run(paths, progress):
        seen["parents"] = {p.parent.name for p in paths}
        seen["names"] = {p.name for p in paths}
        return fake_result()

    monkeypatch.setattr(auditor, "run_audit", run)
    files = [("files", ("..\\..\\bad.txt", b"x")), ("files", ("../../other.txt", b"y"))]
    audit_id = client.post("/api/audits", files=files).json()["id"]
    finish(client, audit_id)
    assert seen["names"] == {"bad.txt", "other.txt"}
    assert len(seen["parents"]) == 1


def test_uploads_are_deleted_after_audit(client, monkeypatch):
    seen = {}

    def run(paths, progress):
        seen["paths"] = paths
        assert all(p.exists() for p in paths)
        return fake_result()

    monkeypatch.setattr(auditor, "run_audit", run)
    finish(client, client.post("/api/audits", files=FILES).json()["id"])
    assert not any(p.exists() for p in seen["paths"])


def test_report_escapes_document_content(client, monkeypatch):
    finding = {
        "topic": HOSTILE, "severity": "high", "explanation": HOSTILE, "newer": None, "confirmed": True,
        "a": {"source": HOSTILE + ".txt", "page": None, "quote": HOSTILE},
        "b": {"source": "b.txt", "page": 2, "quote": "quote"},
    }
    monkeypatch.setattr(auditor, "run_audit", lambda *_: fake_result([finding], [HOSTILE]))
    audit_id = client.post("/api/audits", files=FILES).json()["id"]
    assert finish(client, audit_id)["findings"] == 1
    page = client.get(f"/audits/{audit_id}/report").text
    assert HOSTILE not in page
    assert "&lt;script&gt;" in page
    assert "b.txt, p. 2" in page
    assert "What wasn&#x27;t checked" in page or "What wasn't checked" in page
    assert "{{" not in page


def test_uncertain_findings_are_kept_in_a_separate_section(client, monkeypatch):
    def finding(topic, confirmed):
        return {
            "topic": topic, "severity": "high", "explanation": "x", "newer": None, "confirmed": confirmed,
            "a": {"source": "a.txt", "page": None, "quote": "q1"}, "b": {"source": "b.txt", "page": None, "quote": "q2"},
        }

    result = fake_result([finding("Certain", True), finding("Doubtful", False)])
    monkeypatch.setattr(auditor, "run_audit", lambda *_: result)
    audit_id = client.post("/api/audits", files=FILES).json()["id"]
    audit = finish(client, audit_id)
    assert (audit["findings"], audit["uncertain"]) == (1, 1)
    page = client.get(f"/audits/{audit_id}/report").text
    assert page.index("Certain") < page.index("<h2>To review") < page.index("Doubtful")
    assert "Found 1 contradiction" in page


def test_only_uncertain_findings_are_not_reported_as_clean(client, monkeypatch):
    uncertain = {
        "topic": "Doubtful", "severity": "low", "explanation": "x", "newer": None, "confirmed": False,
        "a": {"source": "a.txt", "page": None, "quote": "q1"}, "b": {"source": "b.txt", "page": None, "quote": "q2"},
    }
    monkeypatch.setattr(auditor, "run_audit", lambda *_: fake_result([uncertain]))
    audit_id = client.post("/api/audits", files=FILES).json()["id"]
    finish(client, audit_id)
    page = client.get(f"/audits/{audit_id}/report").text
    assert "No confirmed contradictions were found" in page
    assert "Doubtful" in page


def test_reports_saved_with_polish_severity_still_render(client, monkeypatch):
    old = {
        "topic": "Termin", "severity": "wysoka", "explanation": "Rozbieżność: 7 dni wobec: 3 dni.", "newer": None,
        "confirmed": True,
        "a": {"source": "a.txt", "page": None, "quote": "q1"}, "b": {"source": "b.txt", "page": None, "quote": "q2"},
    }
    monkeypatch.setattr(auditor, "run_audit", lambda *_: fake_result([old]))
    audit_id = client.post("/api/audits", files=FILES).json()["id"]
    finish(client, audit_id)
    page = client.get(f"/audits/{audit_id}/report").text
    assert 'class="severity high">High severity' in page


def test_failed_audit_shows_reason_and_has_no_report(client, monkeypatch):
    def run(*_):
        raise auditor.AuditError("No model.")

    monkeypatch.setattr(auditor, "run_audit", run)
    audit_id = client.post("/api/audits", files=FILES).json()["id"]
    audit = finish(client, audit_id)
    assert (audit["status"], audit["error"]) == ("failed", "No model.")
    assert client.get(f"/audits/{audit_id}/report").status_code == 409


@pytest.mark.skipif(auditor_web.find_browser() is None, reason="no Edge/Chrome")
def test_pdf_download_is_real_pdf_with_polish_text(client, monkeypatch):
    finding = {
        "topic": "Zażółć gęślą jaźń", "severity": "high", "explanation": "Different deadlines.", "newer": None,
        "confirmed": True,
        "a": {"source": "a.txt", "page": None, "quote": "Termin wynosi 7 dni."},
        "b": {"source": "b.pdf", "page": 2, "quote": "Termin wynosi 3 dni."},
    }
    monkeypatch.setattr(auditor, "run_audit", lambda *_: fake_result([finding]))
    audit_id = client.post("/api/audits", files=FILES).json()["id"]
    finish(client, audit_id)
    response = client.get(f"/audits/{audit_id}/report.pdf")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert f"audit-report-{audit_id[:8]}.pdf" in response.headers["content-disposition"]
    with pymupdf.open(stream=response.content, filetype="pdf") as pdf:
        text = "".join(page.get_text() for page in pdf)
    assert "Zażółć gęślą jaźń" in text
    assert "b.pdf, p. 2" in text
    assert "Page 1 of" in text


def test_pdf_without_browser_gives_clear_error(client, monkeypatch):
    monkeypatch.setattr(auditor_web, "find_browser", lambda: None)
    monkeypatch.setattr(auditor, "run_audit", lambda *_: fake_result())
    audit_id = client.post("/api/audits", files=FILES).json()["id"]
    finish(client, audit_id)
    response = client.get(f"/audits/{audit_id}/report.pdf")
    assert response.status_code == 503
    assert "Edge" in response.json()["detail"]


def test_pdf_of_unknown_audit_is_not_found(client):
    assert client.get("/audits/missing/report.pdf").status_code == 404


def test_delete_removes_report(client, monkeypatch):
    monkeypatch.setattr(auditor, "run_audit", lambda *_: fake_result())
    audit_id = client.post("/api/audits", files=FILES).json()["id"]
    finish(client, audit_id)
    assert client.delete(f"/api/audits/{audit_id}").status_code == 200
    assert client.get(f"/audits/{audit_id}/report").status_code == 404


@pytest.mark.parametrize("bad_id", ["missing", "..%2F..%2Fsecret"])
def test_invalid_id_is_not_found(client, bad_id):
    assert client.get(f"/audits/{bad_id}/report").status_code == 404


def test_foreign_host_is_rejected(client):
    assert client.get("/api/audits", headers={"host": "evil.example"}).status_code == 400
