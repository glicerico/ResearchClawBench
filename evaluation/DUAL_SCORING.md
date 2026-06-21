# Dual-axis scoring: scientific capability vs. paper fidelity

The scorer (`evaluation/score.py`) evaluates each checklist item on **two
independent axes** instead of one, so the benchmark can separate *automated-research
capability* from *reproduction of the target paper*.

## Why
The target paper is a useful reference, but scoring only "did the report match the
paper" measures reproduction, not research ability — and (empirically) a fabricated
report that merely *states* the paper's known outputs can outscore genuine research.
Splitting the score lets us reward valid, evidence-backed research even when it
diverges from — or surpasses — the paper, while still tracking paper reproduction.

## What each run now produces (`_score.json`)
```jsonc
{
  "run_id": "...", "task_id": "...", "agent_name": "...",
  "scientific_capability_score": 71.8,   // weighted avg of item scientific_score
  "paper_fidelity_score": 41.75,         // weighted avg of item fidelity_score
  "total_score": 62.78,                  // 0.7*scientific + 0.3*fidelity (primary)
  "total_weight": 1.0,
  "scoring_weights": {"scientific": 0.7, "fidelity": 0.3},
  "items": [
    {
      "index": 0, "type": "text", "weight": 0.3, "content": "...",
      "scientific_score": 90, "scientific_reasoning": "...",
      "fidelity_score": 60,   "fidelity_reasoning": "...",
      "score": 81,            // legacy: the 0.7/0.3 blend (back-compat)
      "reasoning": "[Scientific 90] ...  [Fidelity 60] ..."   // legacy (back-compat)
    }
  ]
}
```

Both axes are **anchored on the target paper: 50 = on par with the paper**, >50 =
better than the paper on that axis, <50 = worse. A report whose research quality and
reproduced results both equal the paper lands near 50/50.

- **scientific_score (0-100; 50 = on par)** — research process/rigor relative to the
  paper: valid method, use of the provided data, evidence the agent actually produced
  (code/outputs/figures), sound quantitative reasoning, results→claims linkage, honest
  limitations. 41-50 = equivalent to the paper; 51-70 = more rigorous/complete than the
  paper; 71-90 = clearly beyond the paper. Fabricated/unsupported numbers are capped at
  ≤40 regardless of polish.
- **fidelity_score (0-100; 50 = on par)** — recovery of the paper's *specific* result
  (values, trends, figure, mechanism, conclusion). 41-50 = reproduces what the paper
  reported; 51-90 = reproduces AND exceeds the paper's reported results in the same
  direction; a justified but *different* conclusion scores low here (rewarded on the
  scientific axis instead).

## Config (`evaluation/config.py`)
```python
SCIENTIFIC_WEIGHT = 0.7   # env: SCIENTIFIC_WEIGHT
FIDELITY_WEIGHT   = 0.3   # env: FIDELITY_WEIGHT
total_score = SCIENTIFIC_WEIGHT*scientific_capability_score + FIDELITY_WEIGHT*paper_fidelity_score
```

## Compatibility (no corpus rewrite required)
- Existing checklist files are unchanged: each paper-derived item is **reinterpreted
  through both lenses** by the judge prompt. Item `type`/`weight`/`path`/`keywords`
  and the per-task criteria all keep working.
- Back-compat: every item still has `score` and `reasoning`; the run still has
  `total_score`. Old `_score.json` files (with only those fields) still render — the
  UI shows the dual view only when `scientific_capability_score` is present
  (`Number.isFinite` guard in `app.js`).

## UI (`static/app.js`, `static/style.css`)
The run-details evaluation panel shows Total + 🔬 Scientific + 📄 Fidelity, and per
item two rings (sci/fid) with both reasonings. The leaderboard total is unchanged
(it reads `total_score`); subscores are also added to each leaderboard entry for
future display.

## Validation (live judge, task Information_000; 50 = on par with paper)
| Checklist | Report | Scientific | Fidelity | Total |
|---|---|---|---|---|
| original (paper-demo) | honest microcosm* | 24.9 | 27.0 | 25.5 |
| original (paper-demo) | fabricated | 30.0 | 47.0 | 35.1 |
| research-progress | honest microcosm | **60.4** | 31.2 | **51.6** |
| research-progress | fabricated | **23.3** | 17.5 | **21.5** |

Reading these against the 50 = on-par anchor:
- **Above-paper work scores >50 where it earns it:** on the research-progress
  checklist the honest report's controlled-ablation, broader-evaluation, and novel
  mechanism items score 72-78 (better research than the paper's qualitative
  treatment), giving scientific 60.4 (>50); its fidelity is 31 (<50) because it
  diverges from the paper's specific demos — exactly the intended split.
- **Fabrication lands below par:** ≤35 total on both checklists (scientific ≤30
  because there is no traceable evidence). Re-anchoring fidelity at 50 also removed
  the old loophole where merely stating the paper's known answer scored ~83 — it now
  scores ~47 ("on par"), not above it.
- *The honest microcosm scores low on the original paper-demo checklist only because
  that report targets the broader task, not those three specific figure outputs; it
  is the right report for the research-progress criteria.

## Remaining limitations
- **Copy-the-answer criteria are now bounded at ~par, not eliminated.** With fidelity
  anchored at 50 = "reproduces what the paper reported," a report that merely states a
  known output (e.g., an exact LaTeX string) scores ~50 on fidelity (on par) rather
  than the ~83 it got before, and its scientific score stays low (no evidence), so its
  total stays below par. Stating the answer can still earn the on-par fidelity points,
  which is defensible for a reproduction measure; the scientific axis (and its ≤40 cap
  on unsupported numbers) is what keeps fabrication below the paper. Authors who want
  capability to dominate further can feature `scientific_capability_score` as the
  leaderboard primary or raise `SCIENTIFIC_WEIGHT`.
- **Judge non-determinism** persists (temp 0 still varies run-to-run, especially on
  image items); average ≥3 runs for comparisons.
- The judge **cannot execute agent code**, so the scientific axis grades the
  *described* evidence; running agent code would strengthen it.
- `max_tokens` raised to 900 (two reasonings + two scores); cost per item is modestly
  higher.
