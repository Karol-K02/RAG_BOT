"""Tests for the parts that don't need the language model. Document samples are partly Polish on
purpose: the auditor must keep handling Polish documents."""
import io
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from openai import APIConnectionError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import auditor  # noqa: E402

A = {"source": "a.txt", "page": None, "text": "Hasła należy zmieniać co 90 dni. Hasło ma 12 znaków."}
B = {"source": "b.pdf", "page": 3, "text": "Hasło zmieniasz co 180 dni. System przypomni o tym."}
VERDICT = {
    "topic": "Hasła", "severity": "medium", "values_a": ["co 90 dni"], "values_b": ["co 180 dni"],
    "quote_a": "Hasła należy zmieniać co 90 dni.", "quote_b": "Hasło zmieniasz co 180 dni.", "newer": "A",
}


def test_finding_keeps_sources_and_page():
    finding = auditor.build_finding(VERDICT, A, B)
    assert finding["a"]["source"] == "a.txt"
    assert finding["b"]["page"] == 3
    assert finding["newer"] == "a.txt"


def test_explanation_is_built_from_values_in_the_quotes():
    assert auditor.build_finding(VERDICT, A, B)["explanation"] == "Difference: co 90 dni vs. co 180 dni."


def test_model_prose_never_reaches_the_report():
    finding = auditor.build_finding({**VERDICT, "explanation": "Invented legal risk."}, A, B)
    assert "Invented" not in finding["explanation"]


def test_value_not_in_quote_is_not_shown():
    verdict = {**VERDICT, "values_a": ["co 90 dni", "co 7 dni"]}
    assert auditor.build_finding(verdict, A, B)["explanation"] == "Difference: co 90 dni vs. co 180 dni."


@pytest.mark.parametrize("values", [None, [], ["co 7 dni"], 5, [1, None], "  "])
def test_finding_survives_without_usable_values(values):
    finding = auditor.build_finding({**VERDICT, "values_b": values}, A, B)
    assert finding is not None
    assert finding["explanation"] == ""


def test_swapped_sides_swap_values_too():
    swapped = {**VERDICT, "quote_a": VERDICT["quote_b"], "quote_b": VERDICT["quote_a"],
               "values_a": VERDICT["values_b"], "values_b": VERDICT["values_a"]}
    assert auditor.build_finding(swapped, A, B)["explanation"] == "Difference: co 90 dni vs. co 180 dni."


def test_invented_quote_is_rejected():
    assert auditor.build_finding({**VERDICT, "quote_b": "Hasło zmieniasz co 30 dni."}, A, B) is None


def test_swapped_quotes_are_corrected_along_with_newer():
    swapped = {**VERDICT, "quote_a": VERDICT["quote_b"], "quote_b": VERDICT["quote_a"]}
    finding = auditor.build_finding(swapped, A, B)
    assert "90" in finding["a"]["quote"] and "180" in finding["b"]["quote"]
    assert finding["newer"] == "b.pdf"


def test_quote_matching_ignores_whitespace_and_typographic_quotes():
    assert auditor.quote_in("wniosek „urlopowy”  składasz", 'Wniosek "urlopowy" składasz w systemie.')


def test_unknown_severity_falls_back_to_medium():
    assert auditor.build_finding({**VERDICT, "severity": "critical"}, A, B)["severity"] == "medium"


def test_polish_severity_names_are_understood():
    assert auditor.build_finding({**VERDICT, "severity": "wysoka"}, A, B)["severity"] == "high"


def test_duplicates_are_merged_even_when_sides_are_swapped():
    first = auditor.build_finding(VERDICT, A, B)
    second = {**first, "a": first["b"], "b": first["a"]}
    assert len(auditor.deduplicate([first, second])) == 1


def test_different_conflicts_between_same_documents_stay_separate():
    first = auditor.build_finding(VERDICT, A, B)
    other = {**first, "a": {**first["a"], "quote": "Urlop trwa zawsze dokładnie dwadzieścia dni."}}
    assert len(auditor.deduplicate([first, other])) == 2


def test_shorter_quote_contained_in_longer_is_a_duplicate_and_longer_wins():
    full = auditor.build_finding(VERDICT, A, B)
    partial = {**full, "b": {**full["b"], "quote": "zmieniasz co 180"}}
    kept = auditor.deduplicate([partial, full])
    assert len(kept) == 1
    assert kept[0]["b"]["quote"] == VERDICT["quote_b"]


def test_topic_starts_with_capital_letter():
    assert auditor.build_finding({**VERDICT, "topic": "termin"}, A, B)["topic"] == "Termin"


def test_finding_with_values_in_quotes_is_confirmed():
    assert auditor.build_finding(VERDICT, A, B)["confirmed"] is True


def test_finding_without_verified_values_goes_to_review():
    finding = auditor.build_finding({**VERDICT, "values_a": ["co 7 dni"]}, A, B)
    assert finding["confirmed"] is False


@pytest.mark.parametrize("a, b, expected", [
    (["14 dni"], ["w ciągu 14 dni"], True),
    (["8:00 do 16:00"], ["8:00-16:00"], True),
    (["7 zł, 12 zł"], ["12 zł", "7 zł"], True),
    (["30 dni"], ["14 dni kalendarzowych"], False),
    (["oryginały rachunków"], ["skany rachunków"], False),  # no numbers: stays confirmed
])
def test_same_numbers(a, b, expected):
    assert auditor.same_numbers(a, b) is expected


def test_same_numbers_on_both_sides_go_to_review():
    chunk_a = {"source": "a.txt", "page": None, "text": "Zwrot pieniędzy w ciągu 14 dni od otrzymania towaru."}
    chunk_b = {"source": "b.txt", "page": None, "text": "Pieniądze oddamy w terminie 14 dni od zgłoszenia."}
    verdict = {"topic": "Zwrot", "quote_a": chunk_a["text"], "quote_b": chunk_b["text"],
               "values_a": ["14 dni od otrzymania towaru"], "values_b": ["14 dni od zgłoszenia"]}
    finding = auditor.build_finding(verdict, chunk_a, chunk_b)
    assert finding["explanation"] and finding["confirmed"] is False


def test_translated_value_is_recovered_from_the_number_in_the_quote():
    assert auditor.grounded_values(["co 60 dni"], "Passwords must be changed every 60 days.") == ["every 60 days"]


def test_number_in_parentheses_keeps_its_words():
    quote = "defects which become apparent within eighteen (18) months from ship date"
    assert auditor.grounded_values(["18 miesięcy"], quote) == ["eighteen (18) months"]


def test_number_inside_a_longer_number_is_not_matched():
    assert auditor.grounded_values(["co 6 dni"], "Passwords must be changed every 60 days.") == []
    assert auditor.grounded_values(["1 day"], "Remote work is limited to 1.5 days.") == []


def test_parse_verdict_handles_thinking_and_garbage():
    assert auditor.parse_verdict('<think>{"conflicts": [1]}</think>\n{"conflicts": []}') == []
    assert auditor.parse_verdict('{"conflicts": [{"topic": "a"}, {"topic": "b"}]}') == [{"topic": "a"}, {"topic": "b"}]
    assert auditor.parse_verdict("I don't know") is None
    assert auditor.parse_verdict('{"other": 1}') is None
    assert auditor.parse_verdict('{"conflicts": ["text"]}') is None


class FlakyClient:
    """A model client that fails a given number of times, then answers."""

    def __init__(self, failures):
        self.failures, self.calls = failures, 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **_):
        self.calls += 1
        if self.calls <= self.failures:
            raise APIConnectionError(request=httpx.Request("POST", "http://local"))
        message = SimpleNamespace(content='{"conflict": false}')
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_model_crash_is_retried(monkeypatch):
    monkeypatch.setattr(auditor, "LLM_RETRY_SECONDS", 0)
    client = FlakyClient(failures=2)
    assert auditor.complete(client, []) == '{"conflict": false}'
    assert client.calls == 3


def test_permanent_model_failure_stops_the_audit(monkeypatch):
    monkeypatch.setattr(auditor, "LLM_RETRY_SECONDS", 0)
    client = FlakyClient(failures=99)
    with pytest.raises(auditor.AuditError):
        auditor.complete(client, [])
    assert client.calls == auditor.LLM_ATTEMPTS


def test_cache_reuses_answer_only_for_identical_request(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIT_CACHE_DIR", str(tmp_path))
    client = FlakyClient(failures=0)
    messages = [{"role": "user", "content": "question"}]
    assert auditor.complete(client, messages) == auditor.complete(client, messages)
    assert client.calls == 1
    auditor.complete(client, [{"role": "user", "content": "another question"}])
    assert client.calls == 2


def test_cache_is_off_by_default(monkeypatch):
    monkeypatch.delenv("AUDIT_CACHE_DIR", raising=False)
    client = FlakyClient(failures=0)
    auditor.complete(client, [])
    auditor.complete(client, [])
    assert client.calls == 2


def test_pairs_only_cross_documents():
    chunks = [{"source": "a"}, {"source": "a"}, {"source": "b"}]
    vectors = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    assert set(auditor.candidate_pairs(chunks, vectors)) == {(0, 2), (1, 2)}


def test_dissimilar_chunks_are_not_paired():
    chunks = [{"source": "a"}, {"source": "b"}]
    assert auditor.candidate_pairs(chunks, np.array([[1.0, 0.0], [0.0, 1.0]])) == []


def test_chunks_respect_size_and_keep_page():
    text = " ".join(f"To jest zdanie numer {n} o jakiejś zasadzie." for n in range(100))
    chunks = auditor.chunk_pages([{"source": "x.pdf", "page": 7, "text": text}])
    assert len(chunks) > 1
    assert all(c["page"] == 7 and len(c["text"]) <= auditor.CHUNK_SIZE + 60 for c in chunks)


def test_sentence_continues_across_page_break_and_cites_its_first_page():
    pages = [
        {"source": "it.pdf", "page": 1, "text": "4. Lost devices\nA lost device must be reported to the IT"},
        {"source": "it.pdf", "page": 2, "text": "helpdesk within 4 hours.\nThe helpdesk works 8:00-16:00."},
    ]
    chunks = auditor.chunk_pages(pages)
    chunk = next(c for c in chunks if "within 4 hours" in c["text"])
    assert "reported to the IT helpdesk within 4 hours." in chunk["text"]
    assert chunk["section"] == "4. Lost devices"
    assert auditor.quote_page(chunk, "A lost device must be reported to the IT helpdesk within 4 hours.") == 1
    assert auditor.quote_page(chunk, "The helpdesk works 8:00-16:00.") == 2


def test_finding_cites_page_where_quote_starts():
    chunk = {"source": "it.pdf", "page": 1, "text": "Pierwsze zdanie tutaj. Hasła należy zmieniać co 90 dni.",
             "sentences": [["Pierwsze zdanie tutaj.", 1], ["Hasła należy zmieniać co 90 dni.", 2]]}
    finding = auditor.build_finding(VERDICT, chunk, B)
    assert finding["a"]["page"] == 2


@pytest.mark.parametrize("line", [
    "§ 4. Podróże służbowe", "Rozdział 3. Faktury", "Section 3. Remote work", "Article IV", "Kapitel 2 Urlaub",
    "3. Sprzęt po zakończeniu współpracy", "REKLAMACJE", "TERMS AND CONDITIONS", "Jak długo czekam na zwrot?",
])
def test_headings_are_recognised(line):
    assert auditor.is_heading(line)


@pytest.mark.parametrize("line", [
    "8.4.2 dla pozostałych umów – od dnia zawarcia umowy.",
    "Wynagrodzenie jest wypłacane do 10. dnia następnego miesiąca.",
    "Section introduces the rules", "Partner companies may apply",
    "3 dni robocze", "A" * 130,
    "6.4. The Vendor shall not be liable to Client in any manner if delivery becomes impossible due to",
])
def test_body_lines_are_not_headings(line):
    assert not auditor.is_heading(line)


def test_chunks_stay_within_one_section():
    text = "§ 1. Urlop\nUrlop trwa 20 dni.\n§ 2. Sprzęt\nLaptop zwracasz w ostatnim dniu pracy."
    chunks = auditor.chunk_pages([{"source": "r.txt", "page": None, "text": text}])
    assert [(c["section"], c["text"]) for c in chunks] == [
        ("§ 1. Urlop", "Urlop trwa 20 dni."), ("§ 2. Sprzęt", "Laptop zwracasz w ostatnim dniu pracy.")]


def test_heading_number_on_its_own_line_joins_the_title():
    text = "8.\nLimited Warranty and Defects\n8.1 The product is warranted for one year.\n9.\nThe Vendor may assign this contract."
    chunks = auditor.chunk_pages([{"source": "t.pdf", "page": 1, "text": text}])
    assert chunks[0]["section"] == "8. Limited Warranty and Defects"
    assert chunks[0]["text"] == "8.1 The product is warranted for one year. 9. The Vendor may assign this contract."


@pytest.mark.parametrize("quote, text", [
    ("the Vendor's liability is limited", "the Vendor’s liability is limited to repair."),
    ("valid 2020-2024", "valid 2020–2024 only"),
    ("the ‘Vendor’ shall", "the 'Vendor' shall deliver"),
    ("financial terms apply", "ﬁnancial terms apply here"),
])
def test_quotes_match_despite_typography(quote, text):
    assert auditor.quote_in(quote, text)


def test_line_cleanup_joins_hyphenated_words_and_drops_page_numbers():
    text = "Termin zakoń-\nczenia umowy wynosi 30 dni.\n12\nStrona 3 z 9\nSecu\xadrity\xa0pol\xad\nicy applies to all."
    chunk = auditor.chunk_pages([{"source": "u.pdf", "page": 1, "text": text}])[0]
    assert chunk["text"] == "Termin zakończenia umowy wynosi 30 dni. Security policy applies to all."


def test_section_is_given_to_the_judge():
    seen = []

    class Recorder(FlakyClient):
        def create(self, **kwargs):
            seen.append(kwargs["messages"][1]["content"])
            return super().create(**kwargs)

    a = {**A, "section": "§ 9. Reklamacje"}
    auditor.judge_pair(Recorder(failures=0), a, B, {"a.txt": "Regulamin", "b.pdf": "FAQ"})
    assert "Section of fragment A: § 9. Reklamacje" in seen[0]
    assert "Section of fragment B: none" in seen[0]


def test_cp1250_text_file_keeps_polish_letters(tmp_path):
    path = tmp_path / "stary.txt"
    path.write_bytes("Zażółć gęślą jaźń".encode("cp1250"))
    pages, _ = auditor.extract_pages(path)
    assert pages[0]["text"] == "Zażółć gęślą jaźń"


def test_docx_is_read_without_page_numbers(tmp_path):
    body = '<w:p><w:r><w:t>Termin wynosi 14 dni.</w:t></w:r></w:p>'
    xml = ('<?xml version="1.0" encoding="UTF-8"?><w:document xmlns:w='
           '"http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
           f'<w:body>{body}</w:body></w:document>')
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("[Content_Types].xml", (
            '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/word/document.xml" ContentType='
            '"application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>'))
        archive.writestr("_rels/.rels", (
            '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="r1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/'
            'officeDocument" Target="word/document.xml"/></Relationships>'))
        archive.writestr("word/_rels/document.xml.rels", (
            '<?xml version="1.0"?><Relationships '
            'xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>'))
        archive.writestr("word/document.xml", xml)
    path = tmp_path / "umowa.docx"
    path.write_bytes(buffer.getvalue())
    pages, _ = auditor.extract_pages(path)
    assert pages[0]["page"] is None
    assert "14 dni" in pages[0]["text"]


def test_scanned_pdf_pages_are_counted_as_empty(tmp_path):
    import pymupdf
    path = tmp_path / "skan.pdf"
    doc = pymupdf.open()
    doc.new_page()
    doc.new_page().insert_text((72, 72), "Termin wynosi 14 dni.")
    doc.save(path)
    doc.close()
    pages, empty = auditor.extract_pages(path)
    assert empty == 1
    assert pages[0]["page"] == 2
