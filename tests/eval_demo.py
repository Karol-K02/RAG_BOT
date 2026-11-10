"""
Quality check on the demo sets (needs a running model):
    python tests/eval_demo.py        # demo_docs: Polish documents
    python tests/eval_demo.py en     # demo_docs_en: English documents, a PDF with a rule split across
                                     # two pages, and a German FAQ (contradictions across languages)

Shows which planted contradictions were found and which decoys were wrongly reported.
"""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import auditor  # noqa: E402

# (name, parts of both file names, numbers/words that must appear in the quotes)
EXPECTED = [
    ("leave request 7 days / 3 days", ("regulamin", "faq"), ("7 dni", "3 dni")),
    ("remote work 3 days / 2 days", ("polityka", "faq"), ("3 dni", "2 dni")),
    ("password change 90 / 180 days", ("polityka", "faq"), ("90 dni", "180 dni")),
    ("laptop return last day / 14 days", ("regulamin", "polityka"), ("ostatnim dniu", "14 dni")),
    ("travel expenses 30 / 14 days", ("regulamin", "zasady"), ("30 dni", "14 dni")),
    ("receipts originals / scans", ("regulamin", "zasady"), ("oryginały", "skany")),
]
# Decoys: similar topics that are NOT contradictions.
DECOYS = [
    ("employee leave / no leave for B2B", ("regulamin", "zasady"), ("urlop", "urlop")),
    ("working hours / office hours", ("regulamin", "faq"), ("8:00", "7:00")),
]

EXPECTED_EN = [
    ("password change 120 / 60 days", ("handbook", "security"), ("120 days", "60 days")),
    ("remote work 2 days / 1 day", ("handbook", "security"), ("2 days", "1 day")),
    ("lost device 24 h / 4 h (rule split across PDF pages)", ("handbook", "security"), ("24 hours", "4 hours")),
    ("leave request 14 days / 5 Arbeitstage (English vs German)", ("handbook", "faq"), ("14 days", "5 arbeitstage")),
    ("laptop return last day / 7 Tagen (English vs German)", ("security", "faq"), ("last working day", "7 tagen")),
]
DECOYS_EN = [
    ("office hours / helpdesk hours", ("handbook", "security"), ("9:00", "8:00")),
    ("employee paid leave / no paid leave for B2B", ("handbook", "faq"), ("paid leave", "bezahlten urlaub")),
]
SETS = {"pl": ("demo_docs", EXPECTED, DECOYS), "en": ("demo_docs_en", EXPECTED_EN, DECOYS_EN)}


def matches(finding: dict, files: tuple, words: tuple) -> bool:
    sources = (finding["a"]["source"], finding["b"]["source"])
    quotes = (finding["a"]["quote"].lower(), finding["b"]["quote"].lower())
    for order in (sources, sources[::-1]):
        if files[0] in order[0] and files[1] in order[1]:
            first, second = quotes if order == sources else quotes[::-1]
            return words[0] in first and words[1] in second
    return False


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    os.environ.setdefault("AUDIT_CACHE_DIR", str(ROOT / "tests" / "llm_cache"))
    folder, EXPECTED, DECOYS = SETS[sys.argv[1] if len(sys.argv) > 1 else "pl"]  # noqa: N806
    files = sorted(p for p in (ROOT / folder).iterdir() if p.suffix.lower() in auditor.SUPPORTED)
    start = time.time()
    result = auditor.run_audit(files)
    findings = result["findings"]

    found = [name for name, f, w in EXPECTED if any(matches(x, f, w) for x in findings)]
    confirmed = [x for x in findings if x["confirmed"]]
    found_confirmed = [name for name, f, w in EXPECTED if any(matches(x, f, w) for x in confirmed)]
    false_decoys = [name for name, f, w in DECOYS if any(matches(x, f, w) for x in confirmed)]
    review_decoys = [name for name, f, w in DECOYS if any(matches(x, f, w) for x in findings if not x["confirmed"])]
    known = [x for x in findings if any(matches(x, f, w) for _, f, w in EXPECTED)]
    other = [x for x in findings if x not in known]

    print(f"Time: {time.time() - start:.0f} s, pairs checked: {result['pairs_checked']}")
    print(f"Planted contradictions found: {len(found)}/{len(EXPECTED)}")
    for name, _, _ in EXPECTED:
        print(f"  [{'x' if name in found else ' '}] {name}")
    print(f"Of those in the main section: {len(found_confirmed)}/{len(EXPECTED)}")
    print(f"Decoys in the main section: {false_decoys or 'none'}")
    print(f"Decoys in the review section: {review_decoys or 'none'}")
    print(f"Rejected by the quote check: {result['rejected_unverified']}")
    print(f"Warnings: {result['warnings'] or 'none'}")
    print("All findings:")
    for x in findings:
        pages = f"p. {x['a']['page']} / {x['b']['page']}" if x["a"]["page"] or x["b"]["page"] else ""
        print(f"  [{'main  ' if x['confirmed'] else 'review'}] {x['topic']}: {x['explanation'] or '(no difference line)'} {pages}")
    print(f"Other findings ({len(other)}):")
    print(json.dumps(other, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
