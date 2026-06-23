# Two-axis scoring: scientific capability vs. paper fidelity

The scorer (`evaluation/score.py`) rates each run on **two independent axes**, anchored
so that **50 = on par with the target paper** on each:

- **Paper fidelity** — did the agent reproduce the target paper's *specific* results?
  Judged **per checklist item** (the items are paper-derived demonstrations), from the
  report + images.
- **Scientific capability** — did the agent actually do good *research*? Judged
  **holistically, once per run**, across research-process stages, against the agent's
  real work product (report, code, outputs, run metadata, execution trajectory).

```
paper_fidelity_score        = weighted avg of per-item fidelity scores
scientific_capability_score = weighted avg of research-stage scores
total_score                 = 0.7 * scientific_capability_score + 0.3 * paper_fidelity_score
```

## Why the two axes have different granularity
The checklist items are *demonstrations* ("reproduce this figure / output"). Reproduction
has a real per-item axis (fidelity), but it has **no per-item "research process" axis** —
process quality (framing, design, evidence handling, adaptation, synthesis) is a property
of the *whole effort*, not of one demo item. Grading "research" per demo item just made it
collapse onto fidelity. So fidelity stays per-item and the research axis is judged once,
holistically, against what the agent actually did.

## Research-process stages (the scientific axis)
Each stage is scored 0-100 (50 = on par with the paper) and contributes by `weight`
(`RESEARCH_DIMENSIONS` in `score.py`):

| Stage | Weight | Best observed in |
|---|---|---|
| Problem Framing | 1.0 | report |
| Process Design | 1.0 | code + trajectory |
| Experiment / Implementation Design | 0.5 | code |
| Evidence Acquisition | 0.5 | outputs + trajectory |
| Claim Handling | 1.0 | report vs. outputs |
| Adaptation / Pivoting | 0.75 | trajectory |
| Final Output / Synthesis | 1.0 | report |

`adaptation` is *only* visible in the trajectory (clean reports hide dead-ends); the rubric
tells the judge **not** to penalize a clean run that genuinely needed no pivots.

## Evidence the research axis reads (static — no execution)
The judge **reads** the agent's artifacts; it never runs anything. Gathered per run,
each capped to keep the prompt bounded (~20-25K tokens total on Information_000):

- `report/report.md` — the writeup (cap 40k chars)
- `code/**` + top-level `*.py` — scripts the agent wrote (cap 16k)
- `outputs/**` — logs / results the agent's code produced (cap 12k)
- `_meta.json` — status, exit code, model, duration
- `_agent_output.jsonl` — the Claude Code stream-json trajectory, **distilled** to a
  compact `THINK` / `ACTION` / `RESULT` / `FINAL` log (cap 30k), dropping session and
  token-accounting noise

**Critical evidence rule (in the rubric):** trust what the agent *demonstrably did*
(code/outputs/trajectory) over what the report *claims*. A claimed result with no code or
output that produced it is fabrication → the relevant stages (Claim Handling, Evidence
Acquisition) are capped in the 21-40 band, no matter how polished the prose.

## What each run produces (`_score.json`)
```jsonc
{
  "run_id": "...", "task_id": "...", "agent_name": "...",
  "scientific_capability_score": 64.2,   // weighted avg of research stages
  "paper_fidelity_score": 47.5,          // weighted avg of per-item fidelity
  "total_score": 59.2,                   // 0.7*scientific + 0.3*fidelity (primary)
  "total_weight": 1.0,                   // sum of checklist item weights
  "scoring_weights": {"scientific": 0.7, "fidelity": 0.3},
  "research_dimensions": [
    {"key": "problem_framing", "name": "Problem Framing", "weight": 1.0,
     "score": 70, "reasoning": "...",
     "gap": "what was missing / needed to score higher on this stage"}
    // ... one per stage
  ],
  "items": [
    {
      "index": 0, "type": "text", "weight": 0.3, "content": "...",
      "fidelity_score": 45, "fidelity_reasoning": "...",
      "score": 45,            // legacy: now mirrors fidelity (research is holistic)
      "reasoning": "..."      // legacy: mirrors fidelity_reasoning
    }
  ]
}
```

## Anchoring (both axes, every stage)
- 0: absent. 1-20: token effort only. **21-40: attempted with major flaws, OR fabricated /
  untraceable claims (the fabrication band).** 41-50: equivalent to the paper (50 = on par).
  51-70: clearly better than the paper. 71-90: substantially beyond. 91-100: exceptional.

## Config (`evaluation/config.py`)
```python
SCIENTIFIC_WEIGHT = 0.7   # env: SCIENTIFIC_WEIGHT
FIDELITY_WEIGHT   = 0.3   # env: FIDELITY_WEIGHT
total_score = SCIENTIFIC_WEIGHT*scientific_capability_score + FIDELITY_WEIGHT*paper_fidelity_score
```
Stages and their weights are edited in `RESEARCH_DIMENSIONS` (`score.py`).

## Cost / calls
Per run: one fidelity call **per checklist item** (`max_tokens=500`) + **one** holistic
research call (`max_tokens=2200`, larger input — it carries the artifacts). The research
call replaces the old per-item scientific calls, so total calls drop.

## UI (`static/app.js`, `static/style.css`)
- Run-details panel: Total + 🔬 Scientific + 📄 Fidelity, a **Research process** section
  listing each stage's ring + reasoning + a **"To improve"** gap note (what was missed to
  score higher), and per checklist item a single **fidelity** ring + reasoning.
- Leaderboard: total unchanged; each entry also shows `S`/`F` subscores.
- Back-compat: old `_score.json` files (per-item `scientific_score`) still render their
  original dual rings; files with only `total_score` render the single bar.

## Compatibility
- Checklist files are unchanged: paper-derived items drive **fidelity**; their text is also
  passed to the research call as "the paper's bar" (context only).
- Every item still carries `score` / `reasoning`, and each run still has `total_score`, so
  older consumers keep working.

## Limitations
- **Judge non-determinism** persists (temp 0 still varies, especially on image items);
  average ≥3 runs for comparisons.
- The judge reads artifacts but does **not execute** them, so Evidence Acquisition / Claim
  Handling verify that outputs *exist and are consistent*, not that the code is bug-free.
- Process stages depend on the trajectory being captured (`_agent_output.jsonl`); runs from
  harnesses that don't emit it fall back to report/code/outputs only.
- Long runs are distilled/truncated; very large trajectories keep head+tail with a marker.
