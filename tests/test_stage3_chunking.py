import pandas as pd

from data.stage3_chunking import write_chunks_batched


class WhitespaceTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": self.encode(text, add_special_tokens=add_special_tokens)}

    def encode(self, text, add_special_tokens=False):
        return str(text).split()

    def decode(self, token_ids, clean_up_tokenization_spaces=True, skip_special_tokens=True):
        return " ".join(token_ids)


def _article_row(doc_id, dieu_so, noi_dung):
    law_id = f"{doc_id}/TEST"
    ten_van_ban = f"Luật kiểm tra {doc_id}"
    return {
        "doc_id": doc_id,
        "law_id": law_id,
        "ten_van_ban": ten_van_ban,
        "loai_van_ban": "Luật",
        "ngay_ban_hanh": "01/01/2026",
        "nganh": "Doanh nghiệp",
        "linh_vuc": "Doanh nghiệp",
        "phan": "",
        "chuong": "",
        "muc": "",
        "dieu_so": dieu_so,
        "dieu_ten": "Phạm vi điều chỉnh",
        "noi_dung": noi_dung,
        "start_char": 0,
        "end_char": len(noi_dung),
        "doc_uid": f"{law_id}|{ten_van_ban}|{dieu_so}",
    }


def test_write_chunks_batched_writes_multiple_parquet_batches(tmp_path):
    stage2_df = pd.DataFrame(
        [
            _article_row("1", "Điều 1", "1. Nội dung khoản một đủ dài để tạo chunk."),
            _article_row("2", "Điều 2", "1. Nội dung khoản hai đủ dài để tạo chunk."),
            _article_row("3", "Điều 3", "1. Nội dung khoản ba đủ dài để tạo chunk."),
        ]
    )
    output_path = tmp_path / "stage3_chunks.parquet"

    total_chunks, unique_documents = write_chunks_batched(
        stage2_df=stage2_df,
        tokenizer=WhitespaceTokenizer(),
        max_tokens=1024,
        output_path=output_path,
        batch_size=1,
    )

    chunks = pd.read_parquet(output_path)

    assert total_chunks == 3
    assert unique_documents == 3
    assert len(chunks) == 3
    assert chunks["chunk_id"].is_unique
    assert {"breadcrumb", "chunk_id", "part_idx", "chunk_text"}.issubset(chunks.columns)
