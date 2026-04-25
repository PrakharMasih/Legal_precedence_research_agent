from __future__ import annotations

import pytest

from src.ingestion.chunker import Chunker


@pytest.mark.asyncio
async def test_chunker_returns_empty_for_blank_text() -> None:
    chunker = Chunker()

    chunks = await chunker.chunk_text("   ")

    assert chunks == []


@pytest.mark.asyncio
async def test_chunker_returns_parent_and_child_for_short_text() -> None:
    chunker = Chunker()

    chunks = await chunker.chunk_text("This is a short legal sentence.")

    parents = [c for c in chunks if c.chunk_type == "parent"]
    children = [c for c in chunks if c.chunk_type == "child"]
    assert len(parents) >= 1
    assert len(children) >= 1
    assert parents[0].content == "This is a short legal sentence."
    assert children[0].content == "This is a short legal sentence."


@pytest.mark.asyncio
async def test_chunker_parents_come_before_children() -> None:
    """Parents must occupy the front of the returned list."""
    chunker = Chunker()
    text = "Short text for ordering test."

    chunks = await chunker.chunk_text(text)

    seen_child = False
    for chunk in chunks:
        if chunk.chunk_type == "child":
            seen_child = True
        if chunk.chunk_type == "parent":
            assert not seen_child, "Parent chunk found after a child chunk"


@pytest.mark.asyncio
async def test_child_parent_index_resolves_correctly() -> None:
    """Every child's parent_index must reference a valid parent in the parent sublist."""
    chunker = Chunker()
    text = "\n\n".join(
        [
            "The insurer denied liability because the truck driver had no valid licence.",
            "The deceased was 42 years old and supported two minor children.",
            "The transport company owned the commercial vehicle involved in the accident.",
            "The claimant relies on pay and recover principles under the Motor Vehicles Act.",
        ]
    )

    chunks = await chunker.chunk_text(text)

    parents = [c for c in chunks if c.chunk_type == "parent"]
    children = [c for c in chunks if c.chunk_type == "child"]
    assert children, "Expected at least one child chunk"
    for child in children:
        assert child.parent_index is not None
        assert 0 <= child.parent_index < len(parents)


@pytest.mark.asyncio
async def test_section_labels_detected_for_legal_headers() -> None:
    """Section-header detection must label chunks with the correct section name."""
    chunker = Chunker()
    text = (
        "FACTS\n"
        "The vehicle collided at the intersection on 12 March 2022.\n\n"
        "JUDGMENT\n"
        "The respondent is held liable and ordered to pay compensation.\n"
    )

    chunks = await chunker.chunk_text(text)

    sections = {c.section for c in chunks}
    assert "facts" in sections
    assert "judgment" in sections


@pytest.mark.asyncio
async def test_child_chunks_respect_max_size() -> None:
    """Child chunks must not exceed the configured child_size."""
    child_size = 200
    chunker = Chunker(
        parent_size=800,
        parent_overlap=160,
        child_size=child_size,
        child_overlap=40,
    )
    # Build text long enough to force multiple child chunks
    text = " ".join(["The claimant sustained grievous injuries in the road accident."] * 20)

    chunks = await chunker.chunk_text(text)

    children = [c for c in chunks if c.chunk_type == "child"]
    assert children
    for child in children:
        assert len(child.content) <= child_size + 50  # small tolerance for hard-split boundary


@pytest.mark.asyncio
async def test_chunker_handles_document_with_no_section_headers() -> None:
    """Documents without recognised headers must still produce parent+child chunks."""
    chunker = Chunker()
    text = (
        "This is a plain paragraph without any section headers.\n\n"
        "Another paragraph that continues the argument about liability."
    )

    chunks = await chunker.chunk_text(text)

    parents = [c for c in chunks if c.chunk_type == "parent"]
    children = [c for c in chunks if c.chunk_type == "child"]
    assert parents
    assert children
    assert all(c.section == "other" for c in chunks)
