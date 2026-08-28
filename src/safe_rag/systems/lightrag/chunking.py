"""Split corpus documents into overlapping chunks for indexed progress."""


def chunk_text(text: str, chunk_chars: int, overlap_chars: int) -> list[str]:
    cleaned = (text or "").strip()
    if not cleaned:
        return []
    if chunk_chars <= 0:
        return [cleaned]
    overlap_chars = max(0, min(overlap_chars, chunk_chars - 1))
    step = max(chunk_chars - overlap_chars, 1)
    chunks: list[str] = []
    start = 0
    length = len(cleaned)
    while start < length:
        end = min(start + chunk_chars, length)
        piece = cleaned[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= length:
            break
        start += step
    return chunks


def chunk_documents(
    documents: list[str],
    *,
    chunk_token_size: int,
    overlap_token_size: int,
    chars_per_token: int = 4,
) -> list[str]:
    chunk_chars = max(int(chunk_token_size) * int(chars_per_token), 1)
    overlap_chars = max(int(overlap_token_size) * int(chars_per_token), 0)
    pieces: list[str] = []
    for document in documents:
        pieces.extend(chunk_text(document, chunk_chars, overlap_chars))
    return pieces
