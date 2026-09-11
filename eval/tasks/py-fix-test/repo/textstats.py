"""Small text statistics helpers."""


def word_count(text: str) -> int:
    """Return the number of whitespace-separated words."""
    return len(text.split())


def most_common_word(text: str) -> str:
    """Return the most frequent word (case-insensitive). Ties: first seen wins."""
    counts = {}
    for w in text.split():
        w = w.lower().strip(".,!?")
        counts[w] = counts.get(w, 0) + 1
    best = None
    for w, c in counts.items():
        if best is None or c >= counts[best]:
            best = w
    return best


def average_word_length(text: str) -> float:
    """Average length of words, 0.0 for empty text."""
    words = text.split()
    if not words:
        return 0
    return sum(len(w) for w in words) / len(words)
