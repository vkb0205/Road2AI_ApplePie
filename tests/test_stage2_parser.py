from data.stage2_parse_html import parse_document


def _metadata(title="Quyết định kiểm tra"):
    return {
        "law_id": "01/2026/QD-TEST",
        "ten_van_ban": title,
        "loai_van_ban": "Quyết định",
        "ngay_ban_hanh": "01/01/2026",
        "nganh": "Doanh nghiệp",
        "linh_vuc": "Doanh nghiệp",
        "title": title,
    }


def test_parse_document_drops_non_law_like_document_without_dieu():
    html = "<html><body><p>Nội dung văn bản này đủ dài nhưng không có điều khoản.</p></body></html>"

    records, failure = parse_document("doc-1", html, _metadata())

    assert records == []
    assert failure["reason"] == "zero_dieu_no_fallback"
    assert failure["num_dieu"] == 0


def test_parse_document_drops_law_like_document_without_dieu():
    html = "<html><body><p>Nội dung luật này đủ dài nhưng không có marker điều.</p></body></html>"

    records, failure = parse_document("doc-2", html, _metadata(title="Luật kiểm tra"))

    assert records == []
    assert failure["reason"] == "zero_dieu_law_like_doc"
    assert failure["num_dieu"] == 0


def test_parse_document_keeps_articles_when_dieu_exists():
    html = """
    <html><body>
      <p>Điều 1. Phạm vi điều chỉnh</p>
      <p>Nội dung điều khoản này đủ dài để được giữ lại trong parser.</p>
    </body></html>
    """

    records, failure = parse_document("doc-3", html, _metadata(title="Luật kiểm tra"))

    assert failure["reason"] == "single_dieu_law_like_doc"
    assert len(records) == 1
    assert records[0]["dieu_so"] == "Điều 1"
    assert records[0]["noi_dung"] == "Nội dung điều khoản này đủ dài để được giữ lại trong parser."
