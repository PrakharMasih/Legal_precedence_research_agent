from src.models.query import RankedChunk
from src.retrieval.hybrid import rrf_fuse


def make_chunk(chunk_id: str, *, score: float = 0.0) -> RankedChunk:
    return RankedChunk(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        file_name=f"{chunk_id}.pdf",
        content=f"content {chunk_id}",
        char_start=0,
        char_end=10,
        rrf_score=score,
    )


def test_rrf_fuse_merges_overlap_and_sorts_descending() -> None:
    dense = [make_chunk("a"), make_chunk("b"), make_chunk("c")]
    sparse = [make_chunk("b"), make_chunk("c"), make_chunk("d")]

    fused = rrf_fuse(dense, sparse, k=60)

    assert [chunk.chunk_id for chunk in fused[:4]] == ["b", "c", "a", "d"]
    assert fused[0].rrf_score > fused[1].rrf_score > fused[2].rrf_score


def test_rrf_fuse_handles_single_source_results() -> None:
    dense = [make_chunk("x")]

    fused = rrf_fuse(dense, [], k=60)

    assert len(fused) == 1
    assert fused[0].chunk_id == "x"
    assert fused[0].rrf_score == 1 / 61


def test_rrf_fuse_returns_empty_for_no_results() -> None:
    assert rrf_fuse([], [], k=60) == []
