"""
Minimal language identification via function-word overlap.

Why this exists: proper language ID (langdetect, fasttext's lid.176,
langid.py) cannot be installed in this environment -- there is no network
access for `pip install` here. Rather than silently mislabel non-English
tweets as English (which the earlier non-ASCII-ratio heuristic did for any
Latin-script language -- French, German, Spanish, Italian, and Portuguese
tweets have very few non-ASCII characters and slipped through), this module
does simple, transparent, auditable function-word matching: count how many
of a message's tokens are in each language's most common ~40 function
words, and pick the language with the most hits.

This is a real limitation, not a hidden one: function-word overlap can
misfire on very short messages, code-switched text, or languages not in
this word list. It's adequate for filtering a clustering/taxonomy input
set, not a claim of general-purpose language detection. Documented in
docs/decision-log.md.
"""
from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-zA-ZÀ-ÿ]+")

# Compact, hand-picked function-word lists (articles, pronouns, common
# prepositions/conjunctions) -- the words most likely to appear in *any*
# short message regardless of topic, which is what makes them useful
# signal for language ID even on 5-10 word tweets.
_LANG_WORDS: dict[str, set[str]] = {
    "en": {
        "the", "a", "an", "is", "are", "was", "were", "i", "you", "my", "your",
        "to", "of", "and", "in", "on", "for", "with", "this", "that", "it",
        "have", "has", "not", "do", "does", "did", "please", "can", "will",
        "me", "we", "be", "at", "but", "so", "still", "get", "got", "no",
    },
    "fr": {
        "le", "la", "les", "un", "une", "des", "et", "est", "je", "tu", "il",
        "elle", "vous", "nous", "pas", "que", "qui", "pour", "avec", "sur",
        "dans", "au", "aux", "ce", "cette", "mon", "ma", "mes", "votre",
        "merci", "bonjour", "colis", "pourquoi",
    },
    "de": {
        "der", "die", "das", "ein", "eine", "und", "ist", "ich", "du", "sie",
        "wir", "nicht", "mit", "auf", "für", "bei", "von", "es", "kann",
        "wird", "haben", "hat", "warum", "danke", "wie", "auch", "noch",
        "aber", "sind", "war", "wenn",
    },
    "es": {
        "el", "la", "los", "las", "un", "una", "y", "es", "yo", "tu", "su",
        "no", "que", "por", "para", "con", "en", "de", "mi", "gracias",
        "pero", "muy", "esta", "este", "cuando", "como", "si", "ya",
    },
    "it": {
        "il", "lo", "la", "gli", "le", "un", "una", "e", "è", "io", "tu",
        "non", "che", "per", "con", "su", "di", "da", "mio", "grazie",
        "ma", "molto", "questo", "questa", "quando", "come", "se",
    },
    "pt": {
        "o", "a", "os", "as", "um", "uma", "e", "é", "eu", "tu", "não",
        "que", "por", "para", "com", "em", "de", "meu", "obrigado",
        "obrigada", "mas", "muito", "este", "esta", "quando", "como", "se",
        "já", "vocês", "não",
    },
}


def detect_language(text: str) -> tuple[str, float]:
    """Return (best_lang_code, confidence) where confidence is the fraction
    of recognized tokens that matched the winning language's word list.

    Falls back to ("unknown", 0.0) if too few tokens are recognized to
    make a confident call.
    """
    tokens = [t.lower() for t in _TOKEN_RE.findall(text or "")]
    if len(tokens) < 2:
        return "unknown", 0.0

    scores = {lang: 0 for lang in _LANG_WORDS}
    matched_tokens = 0
    for t in tokens:
        for lang, wordset in _LANG_WORDS.items():
            if t in wordset:
                scores[lang] += 1
                matched_tokens += 1

    if matched_tokens == 0:
        return "unknown", 0.0

    best_lang = max(scores, key=scores.get)
    if scores[best_lang] == 0:
        return "unknown", 0.0

    confidence = scores[best_lang] / max(len(tokens), 1)
    return best_lang, round(confidence, 3)


def is_english(text: str, min_confidence: float = 0.12) -> bool:
    """Conservative English filter: require the message to be identified as
    English with at least `min_confidence` (fraction of tokens matching
    English function words), AND not have a *stronger* match against
    another language. `detect_language` already picks the max, so this is
    just a confidence floor to avoid false positives on very short/emoji-
    heavy/URL-heavy messages where matched_tokens is tiny.
    """
    lang, conf = detect_language(text)
    return lang == "en" and conf >= min_confidence
