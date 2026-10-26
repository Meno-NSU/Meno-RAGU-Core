def combine_relevant_chunks(
    chunk_list_1: list[tuple[int, float]] | None = None,
    chunk_list_2: list[tuple[int, float]] | None = None,
) -> list[tuple[int, float]]:
    united_chunks: dict[int, float] = {}
    if chunk_list_1 is not None:
        for idx, score in chunk_list_1:
            if idx in united_chunks:
                if score > united_chunks[idx]:
                    united_chunks[idx] = score
            else:
                united_chunks[idx] = score
    if chunk_list_2 is not None:
        for idx, score in chunk_list_2:
            if idx in united_chunks:
                if score > united_chunks[idx]:
                    united_chunks[idx] = score
            else:
                united_chunks[idx] = score
    return sorted([(idx, united_chunks[idx]) for idx in united_chunks], key=lambda it: (-it[1], it[0]))
