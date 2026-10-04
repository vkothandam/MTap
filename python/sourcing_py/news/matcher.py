"""Verified article -> symbol attribution.

Feed tagging is noisy: a Google News search for ticker ``MANE`` returns a mix of articles
about *MANE* and about *Bagmane* (a substring match on the search term). So the feed's own
tag is treated as a CANDIDATE, not truth — every candidate symbol must be corroborated in
the article text before it is stored, and uncorroborated candidates are discarded.

Corroboration combines two signals:
  • ticker signal  — a ``$MANE`` cashtag, or a standalone ``MANE`` token matched on WORD
    BOUNDARIES (so *Bagmane* does not match) and required to be an uppercase run.
  • company-name signal — the company's normalized name appears as a whole phrase.

Decision (favouring precision, per the requirement to discard incorrect matches):
  • name present                         -> accept ('name')            — names are unambiguous
  • ticker present AND name present       -> accept ('ticker+name')     — strongest
  • explicit ``$TICKER`` cashtag present  -> accept ('cashtag')         — a deliberate signal
  • bare ticker token, no name, no cashtag-> REJECT                     — this is the noise case

Only feed-tag / cashtag / bare-ticker-token symbols are considered candidates (no full
universe name scan per article); each candidate carries a known company name to verify
against. Symbols with no company_name can only be attributed via an explicit cashtag.
"""

from __future__ import annotations

import re

# Company-name tokens we strip before matching, so "Apple Inc." matches "Apple".
_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "limited",
    "plc", "llc", "lp", "sa", "ag", "nv", "holdings", "holding", "group", "the",
    "class", "common", "stock", "ordinary", "shares",
}
# Uppercase tokens that are valid tickers but also common English words — never treat a
# bare occurrence of these as a ticker mention (they still match via cashtag or name).
_STOPWORD_TICKERS = {
    "A", "I", "AN", "AS", "AT", "BE", "BY", "DO", "GO", "IF", "IN", "IS", "IT", "ON", "OR",
    "SO", "TO", "UP", "US", "WE", "ALL", "AND", "ANY", "ARE", "BIG", "CEO", "FOR", "GET",
    "HAS", "NEW", "NOW", "ONE", "OUT", "SEE", "THE", "WHO", "YOU", "CAN", "USA", "PM", "AM",
}

_CASHTAG_RE = re.compile(r"\$([A-Za-z]{1,5})(?:\.[A-Za-z]+)?\b")
# Standalone uppercase run (a plausible bare ticker) — bounded so 'Bagmane' can't yield MANE.
_UPPER_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9])([A-Z]{1,5})(?![A-Za-z0-9])")


def _normalize_name(name: str) -> str:
    """Lowercased company name with punctuation and corporate suffixes stripped."""
    cleaned = re.sub(r"[^a-z0-9 ]+", " ", name.lower())
    tokens = [t for t in cleaned.split() if t and t not in _SUFFIXES]
    return " ".join(tokens)


class Matcher:
    """Built once from the symbol universe; reused across articles."""

    def __init__(self, universe: list[dict]) -> None:
        # symbol -> normalized company name (may be "" if unknown)
        self._name_by_symbol: dict[str, str] = {}
        self._valid_tickers: set[str] = set()
        # normalized-name -> symbol, plus a compiled whole-phrase regex, for candidates
        self._name_regex: dict[str, re.Pattern] = {}
        for row in universe:
            symbol = (row.get("symbol") or "").strip()
            if not symbol:
                continue
            self._valid_tickers.add(symbol.upper())
            norm = _normalize_name(row.get("company_name") or "")
            self._name_by_symbol[symbol] = norm
            # Skip ultra-short/ambiguous names (<=2 chars) to hold precision.
            if len(norm) > 2:
                self._name_regex[symbol] = re.compile(
                    r"(?<![a-z0-9])" + re.escape(norm) + r"(?![a-z0-9])"
                )

    def _ticker_present(self, symbol: str, upper_tokens: set[str]) -> bool:
        u = symbol.upper()
        return u in upper_tokens and u not in _STOPWORD_TICKERS

    def _name_present(self, symbol: str, lowered: str) -> bool:
        rx = self._name_regex.get(symbol)
        return bool(rx and rx.search(lowered))

    def match(self, text: str, *, feed_symbol: str | None = None) -> list[dict]:
        """Return verified [{symbol, match_method}] for `text`. Empty if nothing corroborates."""
        if not text:
            text = ""
        lowered = text.lower()
        upper_tokens = {t for t in _UPPER_TOKEN_RE.findall(text)}
        # Candidate symbols: the feed's tag, explicit cashtags, and bare uppercase tokens
        # that are valid tickers. Each still has to pass verification below.
        cashtags = {c.upper() for c in _CASHTAG_RE.findall(text)}
        candidates: set[str] = set()
        if feed_symbol:
            candidates.add(feed_symbol)
        for c in cashtags:
            if c in self._valid_tickers:
                candidates.add(c)
        for t in upper_tokens:
            if t in self._valid_tickers:
                candidates.add(t)

        results: dict[str, str] = {}
        for symbol in candidates:
            has_cashtag = symbol.upper() in cashtags
            has_ticker = self._ticker_present(symbol, upper_tokens)
            has_name = self._name_present(symbol, lowered)
            if has_ticker and has_name:
                method = "ticker+name"
            elif has_name:
                method = "name"
            elif has_cashtag:
                method = "cashtag"
            else:
                # bare ticker token with no name corroboration -> noise (e.g. MANE in "Bagmane"
                # never reaches here because word-boundary excludes it; a genuine stray token
                # is dropped for lack of support). Discard.
                continue
            results[symbol] = method
        return [{"symbol": s, "match_method": m} for s, m in sorted(results.items())]
