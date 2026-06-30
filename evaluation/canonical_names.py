"""Canonical agent / model names.

Single source of truth for collapsing the historical naming drift in run
metadata (``_meta.json`` / ``_score.json``) into stable identities. Repeated runs
of the same agent should share one canonical ``agent_name`` and one canonical
``model`` id; the dashboard distinguishes individual runs automatically by run
ordinal (see ``server.api_leaderboard``), so names must no longer be hand-mangled
with suffixes like ``2`` or a date.

Unknown values pass through unchanged so new agents/models don't silently break.
"""

from __future__ import annotations

# Raw agent_name -> canonical product label.
AGENT_ALIASES = {
    "Claude Code": "Claude Code",
    "Claude Code 2": "Claude Code",
    "ClaudeCode2": "Claude Code",
    "ChatGPT": "ChatGPT",
    "ChatGPT 2": "ChatGPT",
    "Qwestor": "Qwestor",
    "Qwestor 2": "Qwestor",
    "Qwestor28jun26": "Qwestor",
    "Qwestor28jun26_2": "Qwestor",
    "Codex CLI": "Codex CLI",
}

# Raw model id -> canonical model id. Separator drift is normalized; variants the
# user considers the same model are collapsed (gpt-5.5 == gpt-5.5-thinking).
MODEL_ALIASES = {
    "claude-opus-4-8": "claude-opus-4-8",
    "claude-sonnet-4.6": "claude-sonnet-4.6",
    "claude-4.6-sonnet": "claude-sonnet-4.6",
    "gpt-5.5": "gpt-5.5",
    "gpt-5.5-thinking": "gpt-5.5",
    "gpt-5.5_thinking": "gpt-5.5",
}


def canonical_agent(name: str | None) -> str:
    """Return the canonical agent label for a raw ``agent_name``."""
    if not name:
        return name or ""
    return AGENT_ALIASES.get(name.strip(), name.strip())


def canonical_model(name: str | None) -> str:
    """Return the canonical model id for a raw ``model`` value."""
    if not name:
        return name or ""
    return MODEL_ALIASES.get(name.strip(), name.strip())
