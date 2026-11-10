"""
Measurement on the public documents of three real shops (folder tests/real_shops, not in git).

    python tests/eval_shops.py                # new audit (needs the model), results in tests/real_shops/latest;
                                              # shops already audited with the current auditor.py are skipped
    python tests/eval_shops.py baseline       # scores the saved results in tests/real_shops/baseline

The labels are inconsistencies checked by hand in November 2026. A finding that matches no label counts
as a false alarm, so new real findings have to be reviewed and added.
"""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "tests" / "real_shops"
sys.path.insert(0, str(ROOT))

# Labels live next to the shop pages in tests/real_shops/labels.json, outside git, so the published
# code names no real company. Each label: [shop, name, [part of file name, word in quote], [same for B]].
LABELS = DATA / "labels.json"
REAL = [(s, n, tuple(a), tuple(b)) for s, n, a, b in json.loads(LABELS.read_text(encoding="utf-8"))] \
    if LABELS.exists() else []


def matches(finding: dict, a: tuple, b: tuple) -> bool:
    sides = [(finding["a"]["source"].lower(), finding["a"]["quote"].lower()),
             (finding["b"]["source"].lower(), finding["b"]["quote"].lower())]
    for first, second in (sides, sides[::-1]):
        if a[0] in first[0] and a[1] in first[1] and b[0] in second[0] and b[1] in second[1]:
            return True
    return False


def score(results: dict[str, dict]) -> None:
    total = {"found": 0, "confirmed": 0, "findings": 0, "false": 0, "false_confirmed": 0, "seconds": 0}
    for shop in sorted(results):
        result = results[shop]
        labels = [r for r in REAL if r[0] == shop]
        findings = result["findings"]
        is_real = [any(matches(f, a, b) for _, _, a, b in labels) for f in findings]
        print(f"\n=== {shop}: {len(findings)} findings, {result.get('seconds', '?')} s")
        for _, name, a, b in labels:
            hits = [f for f in findings if matches(f, a, b)]
            mark = "main  " if any(f["confirmed"] for f in hits) else "review" if hits else "MISSED"
            print(f"  [{mark}] {name}")
            total["found"] += bool(hits)
            total["confirmed"] += any(f["confirmed"] for f in hits)
        false = [f for f, real in zip(findings, is_real) if not real]
        total["findings"] += len(findings)
        total["false"] += len(false)
        total["false_confirmed"] += sum(f["confirmed"] for f in false)
        total["seconds"] += result.get("seconds", 0)
        for f in false:
            print(f"    false{' (main)' if f['confirmed'] else ''}: {f['topic']} | {f['explanation'] or '-'}")
    print(f"\nTOTAL: found {total['found']}/{len(REAL)}, in the main section {total['confirmed']}/{len(REAL)}, "
          f"findings {total['findings']}, false {total['false']} (in the main section {total['false_confirmed']}), "
          f"time {total['seconds']} s")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    if sys.argv[1:] == ["baseline"]:
        score({p.stem: json.loads(p.read_text(encoding="utf-8")) for p in (DATA / "baseline").glob("*.json")})
        return
    os.environ.setdefault("AUDIT_CACHE_DIR", str(ROOT / "tests" / "llm_cache"))
    import auditor

    out = DATA / "latest"
    out.mkdir(exist_ok=True)
    code_changed = (ROOT / "auditor.py").stat().st_mtime
    results = {}
    for shop in sorted({r[0] for r in REAL}):
        saved = out / f"{shop}.json"
        if saved.exists() and saved.stat().st_mtime > code_changed:  # interrupted run: don't start over
            results[shop] = json.loads(saved.read_text(encoding="utf-8"))
            print(f"{shop}: result from the previous run", flush=True)
            continue
        start = time.time()
        results[shop] = auditor.run_audit(sorted((DATA / shop).glob("*.txt")))
        results[shop]["seconds"] = round(time.time() - start)
        (out / f"{shop}.json").write_text(json.dumps(results[shop], ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{shop}: done in {results[shop]['seconds']} s", flush=True)
    score(results)


if __name__ == "__main__":
    main()
