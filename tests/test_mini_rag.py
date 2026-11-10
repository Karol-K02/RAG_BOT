"""Tests for DocChat's retrieval in mini_rag.py; the embedding model is replaced by simple stand-ins."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mini_rag  # noqa: E402


def words(text: str) -> int:
    """Stand-in tokenizer: one token per word."""
    return len(text.split())


def test_every_piece_fits_the_limit_and_no_text_is_lost():
    text = "One two three. Four five six seven. Eight nine. Ten eleven twelve thirteen fourteen."
    pieces = mini_rag.split_windows(text, words, limit=6)
    assert all(words(p) <= 6 for p in pieces)
    assert " ".join(pieces) == text


def test_pieces_keep_whole_sentences_when_they_fit():
    pieces = mini_rag.split_windows("Alpha beta gamma. Delta epsilon zeta.", words, limit=4)
    assert pieces == ["Alpha beta gamma.", "Delta epsilon zeta."]


def test_a_sentence_longer_than_the_limit_is_split_by_words():
    sentence = " ".join(f"w{n}" for n in range(10)) + "."
    pieces = mini_rag.split_windows(sentence, words, limit=4)
    assert [words(p) for p in pieces] == [4, 4, 2]


def test_empty_text_gives_no_pieces():
    assert mini_rag.split_windows("   ", words, limit=4) == []


def test_search_returns_each_chunk_once_best_piece_first():
    index = object.__new__(mini_rag.Index)
    index.chunks = [{"source": "a.txt", "page": None, "text": "A"},
                    {"source": "b.txt", "page": None, "text": "B"},
                    {"source": "c.txt", "page": None, "text": "C"}]
    index.parents = np.array([0, 0, 1, 2])  # chunk A has two pieces
    index.vectors = np.array([[0.2, 0.0], [0.9, 0.0], [0.5, 0.0], [0.1, 0.0]])
    index.embedder = SimpleNamespace(encode=lambda texts, normalize_embeddings: np.array([[1.0, 0.0]]))
    results = index.search("question", k=2)
    assert [r["text"] for r in results] == ["A", "B"]
    assert results[0]["score"] == 0.9
