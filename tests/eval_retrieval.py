"""
Do real contradictions even reach the pairs the model judges? Doesn't call the model.

    python tests/eval_retrieval.py [embedding_model ...]

For each known contradiction, finds the fragments containing both quotes and checks where their pair
ranks in the queue. A pair beyond the MAX_PAIRS limit never reaches the model.
"""
import sys
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import auditor  # noqa: E402
import eval_demo  # noqa: E402
import eval_shops  # noqa: E402


def label_sets() -> dict[str, tuple[list[Path], list[tuple]]]:
    """{set: (files, [(name, (file, word), (file, word))])}"""
    sets = {}
    for shop in sorted({r[0] for r in eval_shops.REAL}):
        files = sorted((eval_shops.DATA / shop).glob("*.txt"))
        if files:
            sets[shop] = (files, [(n, a, b) for s, n, a, b in eval_shops.REAL if s == shop])
    demo = [(n, (f[0], w[0]), (f[1], w[1])) for n, f, w in eval_demo.EXPECTED]
    sets["demo_pl"] = (sorted((ROOT / "demo_docs").glob("*.txt")), demo)
    english = getattr(eval_demo, "EXPECTED_EN", None)
    if english and (ROOT / "demo_docs_en").exists():
        sets["demo_en"] = (sorted((ROOT / "demo_docs_en").glob("*")),
                           [(n, (f[0], w[0]), (f[1], w[1])) for n, f, w in english])
    return sets


def side(chunks: list[dict], spec: tuple) -> set[int]:
    file, word = spec
    return {i for i, c in enumerate(chunks) if file in c["source"].lower() and word.lower() in c["text"].lower()}


def evaluate(model_name: str, sets: dict) -> None:
    model = SentenceTransformer(model_name, device="cpu")
    print(f"\n##### {model_name} (token limit: {model.max_seq_length})")
    total_hit = total = 0
    for name, (files, labels) in sets.items():
        chunks = auditor.load_documents(files)["chunks"]
        start = time.time()
        texts = [auditor.embedding_text(c) if hasattr(auditor, "embedding_text") else c["text"] for c in chunks]
        vectors = model.encode(texts, normalize_embeddings=True, batch_size=16)
        seconds = time.time() - start
        lengths = np.array([len(model.tokenizer(t)["input_ids"]) for t in texts])
        pairs = auditor.candidate_pairs(chunks, vectors)
        rank = {pair: r for r, pair in enumerate(pairs)}
        print(f"=== {name}: {len(chunks)} fragments, {len(pairs)} pairs, embeddings {seconds:.0f} s, "
              f"truncated fragments: {(lengths > model.max_seq_length).mean():.0%}")
        for label, a, b in labels:
            left, right = side(chunks, a), side(chunks, b)
            if not left or not right:
                print(f"  [ ? ] {label}: no fragment with the quote was found")
                continue
            ranks = [rank[(min(i, j), max(i, j))] for i in left for j in right if (min(i, j), max(i, j)) in rank]
            best = min(ranks) if ranks else None
            hit = best is not None and best < auditor.MAX_PAIRS
            total += 1
            total_hit += hit
            where = "no pair" if best is None else f"rank {best + 1}"
            print(f"  [{'x' if hit else ' '}] {label}: {where}")
    print(f"TOTAL {model_name}: {total_hit}/{total} contradictions in the pairs the model judges")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    for model_name in sys.argv[1:] or [auditor.EMBED_MODEL]:
        evaluate(model_name, label_sets())
