"""
Web app for the Document Auditor. It only runs on this computer (127.0.0.1).

Run:
    python auditor_web.py
Address: http://127.0.0.1:8000

Uploaded files are deleted as soon as the audit ends. Only the report stays, in the audits/ folder
(it contains quotes from the documents), and it can be deleted with a button in the app.
"""
import html
import json
import os
import shutil
import subprocess
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from starlette.middleware.trustedhost import TrustedHostMiddleware

APP_DIR = Path(__file__).resolve().parent
AUDITS_DIR = APP_DIR / "audits"
# The same values as in auditor.py and mini_rag.py. Not imported, because that pulls in torch
# and the server should start quickly.
SUPPORTED = {".pdf", ".docx", ".txt"}
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:1234/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen/qwen3.5-9b")
MAX_FILE_BYTES = 50 * 1024 * 1024
MAX_FILES = 50
PDF_TIMEOUT_SECONDS = 60
BROWSERS = [
    ("ProgramFiles(x86)", "Microsoft/Edge/Application/msedge.exe"),
    ("ProgramFiles", "Microsoft/Edge/Application/msedge.exe"),
    ("ProgramFiles", "Google/Chrome/Application/chrome.exe"),
    ("ProgramFiles(x86)", "Google/Chrome/Application/chrome.exe"),
    ("LOCALAPPDATA", "Google/Chrome/Application/chrome.exe"),
]
SEVERITY_LABELS = {"high": "High severity", "medium": "Medium severity", "low": "Low severity"}
# Reports saved before the app was translated use the Polish names.
SEVERITY_ALIASES = {"wysoka": "high", "średnia": "medium", "niska": "low"}

app = FastAPI(title="Document Auditor")
# Blocks web pages that would try to call this server through DNS rebinding.
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])

_running: dict[str, dict] = {}  # audits in progress; finished ones are saved as files
_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=1)  # the model handles one audit at a time


# --- storage --------------------------------------------------------------
def audit_path(audit_id: str) -> Path:
    try:
        return AUDITS_DIR / f"{uuid.UUID(hex=audit_id).hex}.json"
    except ValueError:
        raise HTTPException(404, "No such audit.") from None


def load_audit(audit_id: str) -> dict:
    path = audit_path(audit_id)
    with _lock:
        if audit_id in _running:
            return dict(_running[audit_id])
    if not path.exists():
        raise HTTPException(404, "No such audit.")
    return json.loads(path.read_text(encoding="utf-8"))


def execute(audit_id: str, folder: Path, files: list[Path]) -> None:
    def progress(stage: str, fraction: float) -> None:
        with _lock:
            _running[audit_id].update(status="running", stage=stage, progress=fraction)

    with _lock:
        audit = _running[audit_id]
    try:
        import auditor  # heavy import (torch), only when it's needed
        try:
            audit.update(auditor.run_audit(files, progress), status="done", progress=1.0)
        except auditor.AuditError as error:
            audit.update(status="failed", error=str(error))
    except Exception as error:
        audit.update(status="failed", error=f"Unexpected error: {error}")
    finally:
        shutil.rmtree(folder, ignore_errors=True)  # the client's documents don't stay on disk
    AUDITS_DIR.mkdir(exist_ok=True)
    audit_path(audit_id).write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    with _lock:
        del _running[audit_id]


# --- API ------------------------------------------------------------------
@app.get("/")
def index() -> FileResponse:
    return FileResponse(APP_DIR / "auditor_ui.html")


@app.get("/api/health")
def health() -> dict:
    try:
        with urlopen(f"{LLM_BASE_URL}/models", timeout=3) as response:
            models = [m["id"] for m in json.load(response)["data"]]
    except (URLError, OSError, ValueError, KeyError):
        return {"ok": False, "message": f"No model available at {LLM_BASE_URL}. Start the server in LM Studio."}
    if LLM_MODEL not in models:
        return {"ok": False, "message": f"The server is running, but model {LLM_MODEL} isn't there. Load it in LM Studio."}
    return {"ok": True, "message": f"Model ready: {LLM_MODEL}. Documents never leave this computer."}


@app.post("/api/audits")
def create_audit(files: list[UploadFile] = File(...)) -> dict:
    if len(files) < 2:
        raise HTTPException(400, "Choose at least two documents.")
    if len(files) > MAX_FILES:
        raise HTTPException(400, f"One audit can include at most {MAX_FILES} files.")

    names = [Path((f.filename or "").replace("\\", "/")).name for f in files]
    bad = [n for n in names if Path(n).suffix.lower() not in SUPPORTED]
    if bad:
        raise HTTPException(400, f"Unsupported format: {', '.join(bad)}. Allowed: PDF, DOCX, TXT.")
    if len(set(names)) != len(names):
        raise HTTPException(400, "Two files have the same name. Rename one of them so the report is unambiguous.")

    folder = Path(tempfile.mkdtemp(prefix="auditor_"))
    saved = []
    try:
        for upload, name in zip(files, names):
            if (upload.size or 0) > MAX_FILE_BYTES:
                raise HTTPException(400, f"{name} is larger than 50 MB.")
            target = folder / name
            with target.open("wb") as out:
                shutil.copyfileobj(upload.file, out)
            saved.append(target)
    except Exception:
        shutil.rmtree(folder, ignore_errors=True)
        raise

    audit_id = uuid.uuid4().hex
    with _lock:
        _running[audit_id] = {
            "id": audit_id,
            "created": datetime.now().isoformat(timespec="seconds"),
            "file_names": names,
            "status": "queued",
            "stage": "Queued",
            "progress": 0.0,
        }
    _executor.submit(execute, audit_id, folder, saved)
    return {"id": audit_id}


@app.get("/api/audits")
def list_audits() -> list[dict]:
    audits = []
    if AUDITS_DIR.exists():
        audits = [json.loads(p.read_text(encoding="utf-8")) for p in AUDITS_DIR.glob("*.json")]
    with _lock:
        audits += [dict(a) for a in _running.values()]
    audits.sort(key=lambda a: a["created"], reverse=True)
    return [
        {
            "id": a["id"], "created": a["created"], "status": a["status"], "stage": a.get("stage"),
            "progress": a.get("progress", 0), "error": a.get("error"), "file_names": a["file_names"],
            "findings": sum(f["confirmed"] for f in a.get("findings", [])),
            "uncertain": sum(not f["confirmed"] for f in a.get("findings", [])),
        }
        for a in audits
    ]


@app.delete("/api/audits/{audit_id}")
def delete_audit(audit_id: str) -> dict:
    path = audit_path(audit_id)
    with _lock:
        if audit_id in _running:
            raise HTTPException(409, "The audit is still running.")
    if not path.exists():
        raise HTTPException(404, "No such audit.")
    path.unlink()
    return {"deleted": True}


# --- report ---------------------------------------------------------------
def citation(side: dict) -> str:
    return side["source"] if side["page"] is None else f'{side["source"]}, p. {side["page"]}'


def severity_of(finding: dict) -> str:
    severity = SEVERITY_ALIASES.get(finding["severity"], finding["severity"])
    return severity if severity in SEVERITY_LABELS else "medium"


def render_finding(number: int, finding: dict) -> str:
    esc = html.escape
    def side(key: str) -> str:
        data = finding[key]
        newer = ' <span class="newer">newer document</span>' if finding.get("newer") == data["source"] else ""
        return (
            f'<figure><figcaption>{esc(citation(data))}{newer}</figcaption>'
            f'<blockquote>{esc(data["quote"])}</blockquote></figure>'
        )
    severity = severity_of(finding)
    explanation = f'<p>{esc(finding["explanation"])}</p>' if finding["explanation"] else ""
    return (
        f'<article class="finding"><header><h3>{number}. {esc(finding["topic"])}</h3>'
        f'<span class="severity {severity}">{SEVERITY_LABELS[severity]}</span></header>'
        f'<div class="quotes">{side("a")}{side("b")}</div>'
        f'{explanation}</article>'
    )


def render_report(audit: dict) -> str:
    esc = html.escape
    findings = [f for f in audit["findings"] if f["confirmed"]]
    uncertain = [f for f in audit["findings"] if not f["confirmed"]]
    created = datetime.fromisoformat(audit["created"])
    created_text = f"{created.day} {created:%B %Y, %H:%M}"
    if findings:
        counts = {s: sum(severity_of(f) == s for f in findings) for s in SEVERITY_LABELS}
        detail = ", ".join(f"{s}: {counts[s]}" for s in SEVERITY_LABELS)
        noun = "contradiction" if len(findings) == 1 else "contradictions"
        summary = f"Found {len(findings)} {noun} ({detail})."
    else:
        summary = "No confirmed contradictions were found between the compared fragments."
    if uncertain:
        summary += f" To review: {len(uncertain)}."
    documents = "".join(
        f'<li>{esc(d["name"])}' + (f' <span class="muted">({d["pages"]} pages)</span>' if d["pages"] > 1 else "") + "</li>"
        for d in audit["documents"]
    )
    warnings = ""
    if audit["warnings"]:
        items = "".join(f"<li>{esc(w)}</li>" for w in audit["warnings"])
        warnings = f'<section class="warnings"><h2>What wasn\'t checked</h2><ul>{items}</ul></section>'
    body = "".join(render_finding(i, f) for i, f in enumerate(findings, start=1))
    if not findings:
        body = '<p class="muted">No confirmed contradictions.</p>'
    review = ""
    if uncertain:
        items = "".join(render_finding(i, f) for i, f in enumerate(uncertain, start=len(findings) + 1))
        review = (
            "<h2>To review</h2><p>For these findings, no differing values could be found in the quotes, "
            f"or both sides contain the same numbers. They're more often false alarms, but not always.</p>{items}"
        )
    template = (APP_DIR / "auditor_report.html").read_text(encoding="utf-8")
    return (
        template.replace("{{created}}", esc(created_text))
        .replace("{{summary}}", esc(summary))
        .replace("{{warnings}}", warnings)
        .replace("{{findings}}", body)
        .replace("{{uncertain}}", review)
        .replace("{{documents}}", documents)
        .replace("{{pairs}}", str(audit["pairs_checked"]))
        .replace("{{model}}", esc(audit["model"]))
    )


def find_browser() -> str | None:
    for base, relative in BROWSERS:
        path = Path(os.environ.get(base, "")) / relative
        if os.environ.get(base) and path.exists():
            return str(path)
    return None


def render_pdf(page: str) -> bytes:
    """Prints the report to PDF with the browser installed on the system, with networking blocked."""
    browser = find_browser()
    if browser is None:
        raise HTTPException(503, "Creating a PDF needs Microsoft Edge or Google Chrome.")
    with tempfile.TemporaryDirectory(prefix="auditor_pdf_", ignore_cleanup_errors=True) as folder:
        source, target = Path(folder) / "report.html", Path(folder) / "report.pdf"
        source.write_text(page, encoding="utf-8")
        command = [
            browser, "--headless=new", "--disable-gpu", "--disable-extensions", "--no-first-run",
            "--disable-background-networking", "--host-resolver-rules=MAP * ~NOTFOUND",
            "--no-pdf-header-footer", f"--user-data-dir={Path(folder) / 'profile'}",
            f"--print-to-pdf={target}", source.as_uri(),
        ]
        try:
            subprocess.run(command, capture_output=True, timeout=PDF_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            raise HTTPException(503, "Creating the PDF took too long. Please try again.") from None
        if not target.exists():
            raise HTTPException(503, "The PDF file couldn't be created.")
        return target.read_bytes()


def finished_audit(audit_id: str) -> dict:
    audit = load_audit(audit_id)
    if audit["status"] != "done":
        raise HTTPException(409, "The report isn't ready yet.")
    return audit


@app.get("/audits/{audit_id}/report", response_class=HTMLResponse)
def report(audit_id: str) -> HTMLResponse:
    return HTMLResponse(render_report(finished_audit(audit_id)))


@app.get("/audits/{audit_id}/report.pdf")
def report_pdf(audit_id: str) -> Response:
    pdf = render_pdf(render_report(finished_audit(audit_id)))
    headers = {"Content-Disposition": f'attachment; filename="audit-report-{audit_id[:8]}.pdf"'}
    return Response(pdf, media_type="application/pdf", headers=headers)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
