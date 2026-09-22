"""OpenCoder-inspired deterministic quality signals for source documents."""

from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache
from typing import Any

from tree_sitter_language_pack import get_parser


LANGUAGE_PARSERS = {
    "c": "c",
    "c#": "csharp",
    "c++": "cpp",
    "dart": "dart",
    "go": "go",
    "java": "java",
    "javascript": "javascript",
    "kotlin": "kotlin",
    "php": "php",
    "python": "python",
    "ruby": "ruby",
    "rust": "rust",
    "scala": "scala",
    "swift": "swift",
    "typescript": "typescript",
}

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
WORD_RE = re.compile(r"\w+", re.UNICODE)
ENCODED_RE = re.compile(r"(?:[A-Za-z0-9+/]{100,}={0,2}|[0-9A-Fa-f]{100,})")


@lru_cache(maxsize=None)
def _parser(language: str):
    return get_parser(LANGUAGE_PARSERS[language])


def _walk(root):
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(reversed(node.children))


def _top_ngram_fraction(tokens: list[str], size: int) -> float:
    if len(tokens) < size:
        return 0.0
    counts = Counter(tuple(tokens[index:index + size]) for index in range(len(tokens) - size + 1))
    return max(counts.values()) * size / len(tokens)


def measure_quality(content: str, language: str) -> dict[str, Any]:
    """Return JSON-safe, per-document signals adapted from OpenCoder rules."""
    characters = len(content)
    encoded = content.encode("utf-8")
    words = WORD_RE.findall(content)
    tokens = TOKEN_RE.findall(content)
    lines = content.splitlines()
    nonblank_lines = [line.strip() for line in lines if line.strip()]
    duplicate_lines = sum(count - 1 for count in Counter(nonblank_lines).values() if count > 1)

    alpha = sum(character.isalpha() for character in content)
    digits = sum(character.isdigit() for character in content)
    whitespace = sum(character.isspace() for character in content)
    replacement = content.count("\ufffd")
    encoded_characters = sum(match.end() - match.start() for match in ENCODED_RE.finditer(content))

    signals: dict[str, Any] = {
        "num_characters": characters,
        "num_bytes": len(encoded),
        "num_words": len(words),
        "num_lines": len(lines),
        "mean_word_length": sum(map(len, words)) / max(1, len(words)),
        "max_line_length": max(map(len, lines), default=0),
        "mean_line_length": sum(map(len, lines)) / max(1, len(lines)),
        "replacement_character_fraction": replacement / max(1, characters),
        "duplicate_line_fraction": duplicate_lines / max(1, len(nonblank_lines)),
        "top_2gram_fraction": _top_ngram_fraction(tokens, 2),
        "top_3gram_fraction": _top_ngram_fraction(tokens, 3),
        "top_4gram_fraction": _top_ngram_fraction(tokens, 4),
        "alphabetic_fraction": alpha / max(1, characters),
        "numeric_fraction": digits / max(1, characters),
        "whitespace_fraction": whitespace / max(1, characters),
        "encoded_data_fraction": encoded_characters / max(1, characters),
        "parse_supported": language.lower() in LANGUAGE_PARSERS,
        "parse_error": False,
        "comment_fraction": 0.0,
        "long_string_fraction": 0.0,
    }

    normalized_language = language.lower()
    if normalized_language in LANGUAGE_PARSERS:
        tree = _parser(normalized_language).parse(encoded)
        signals["parse_error"] = tree.root_node.has_error
        comment_bytes = 0
        long_string_bytes = 0
        for node in _walk(tree.root_node):
            node_size = node.end_byte - node.start_byte
            node_type = node.type.lower()
            if "comment" in node_type:
                comment_bytes += node_size
            if (
                "string" in node_type
                and node_size >= 100
                and (node.parent is None or "string" not in node.parent.type.lower())
            ):
                long_string_bytes += node_size
        signals["comment_fraction"] = comment_bytes / max(1, len(encoded))
        signals["long_string_fraction"] = long_string_bytes / max(1, len(encoded))

    return signals


# signal, comparison, threshold, rejection reason
QUALITY_RULES = (
    ("num_characters", "min", 50, "too_few_characters"),
    ("num_words", "min", 30, "too_few_words"),
    ("num_lines", "min", 10, "too_few_lines"),
    ("num_lines", "max", 100_000, "too_many_lines"),
    ("num_bytes", "max", 3_000_000, "file_too_large"),
    ("mean_word_length", "min", 2.0, "mean_word_too_short"),
    ("mean_word_length", "max", 10.0, "mean_word_too_long"),
    ("max_line_length", "max", 1_000, "line_too_long"),
    ("mean_line_length", "min", 5.0, "mean_line_too_short"),
    ("mean_line_length", "max", 100.0, "mean_line_too_long"),
    ("replacement_character_fraction", "max", 0.01, "too_many_replacement_characters"),
    ("duplicate_line_fraction", "max", 0.70, "too_many_duplicate_lines"),
    ("top_2gram_fraction", "max", 0.20, "repetitive_2grams"),
    ("top_3gram_fraction", "max", 0.18, "repetitive_3grams"),
    ("top_4gram_fraction", "max", 0.16, "repetitive_4grams"),
    ("alphabetic_fraction", "min", 0.25, "too_little_alphabetic_content"),
    ("numeric_fraction", "max", 0.20, "too_much_numeric_content"),
    ("whitespace_fraction", "max", 0.50, "too_much_whitespace"),
    ("comment_fraction", "max", 0.80, "too_many_comments"),
    ("encoded_data_fraction", "max", 0.40, "too_much_encoded_data"),
    ("long_string_fraction", "max", 0.40, "too_much_long_string_data"),
    ("parse_error", "false", False, "parse_error"),
)


def first_rejection(signals: dict[str, Any]) -> tuple[str, str, Any, Any] | None:
    for signal, comparison, threshold, reason in QUALITY_RULES:
        value = signals[signal]
        failed = (
            (comparison == "min" and value < threshold)
            or (comparison == "max" and value > threshold)
            or (comparison == "false" and value is not False)
        )
        if failed:
            return reason, signal, value, threshold
    return None
