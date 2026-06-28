# Research-Capability Redesign: Astronomy_000 & Earth_002

Companion to [`task_rubric_feasibility.md`](./task_rubric_feasibility.md). This records two tasks reworked to serve as **dedicated evaluators of an agent's general research capability** — its ability to form a hypothesis, design experiments to test it, execute, analyze, and validate or reject the hypothesis — rather than to test reproduction/transcription fidelity.

Originals are preserved next to each edited file as `*.orig`:

- `tasks/Astronomy_000/task_info.json.orig`, `tasks/Astronomy_000/target_study/checklist.json.orig`
- `tasks/Earth_002/task_info.json.orig`, `tasks/Earth_002/target_study/checklist.json.orig`

Restore any original with `cp <file>.orig <file>`.

## Why these two tasks

The feasibility audit surfaced a tension: the tasks that score highest on *reproducibility* do so because the answer is pre-baked into `data/` (pre-extracted figure points, shipped result CSVs, even target figures). They reproduce perfectly and test almost **no research** — there is no hypothesis to form.

For evaluating research capability we want the opposite band: data **raw enough that the agent must form a hypothesis and design experiments**, but **complete enough that it actually can**, with **no leak** so the work is genuine.

- **Astronomy_000 (ULB superradiance)** — the cleanest hypothesis → experiment → validate/reject arc in the suite. Raw mass/spin posteriors (not answers), a falsifiable physics question, multi-stage derivation, and it is computationally light, so it isolates *research reasoning* from engineering/infra grind.
- **Earth_002 (mangrove climate risk)** — the strongest *experiment-design* test: the risk-index method is deliberately under-specified, so the agent must invent and justify a methodology, integrate three heterogeneous real datasets, and test a falsifiable hypothesis across SSP scenarios. It complements Astronomy_000 (empirical/geospatial vs. theoretical physics; moderate vs. light compute).

## Design principles applied to both

1. **De-prescribe the method.** The original task texts named the technique ("develop a Bayesian framework that ingests the full posterior...", "develop a composite risk index..."). That handed the agent the key research insight. The rewrites pose a *question* and make the methodological choices the agent's responsibility.
2. **Remove giveaways.** Pure "report these summary statistics" items were reduced in weight.
3. **Reward reasoning over exact numbers.** Items that previously matched paper-private values (an exact coupling limit `g < Y`; exact `40–56%` / `1%→13%` percentages) now accept order-of-magnitude / directional / monotonic agreement achieved through transparent reasoning.
4. **Reward assumption-handling and sensitivity.** New explicit credit for stating the assumptions a conclusion rests on and showing how the conclusion moves when they change. (Deliberately *not* adding those assumptions to the inputs — their absence is what makes the item real research.)
5. **Add a validate/reject synthesis item.** Each task now explicitly grades the verdict: what is and isn't supported, and the principal caveats.
6. **Use all the shared data.** Astronomy_000's second black hole was previously shipped but never graded; it is now activated.

Net effect on both: ~40–45% of rubric weight now lands on things only genuine research produces (designing/justifying method, using all the data, stating + testing assumptions, reaching a defensible verdict). This also relieves the `scientific_capability ↔ paper_fidelity` tension in `evaluation/score.py`, since more weight sits on claims the holistic axis is meant to reward.

## Astronomy_000 — item-by-item

| # | New weight | (was) | Item | Change |
|---|-----------|-------|------|--------|
| 1 | 0.10 | 0.20 | Characterize **both** posteriors + justify using the full distribution | Weight halved; reframed from a giveaway ("report M±σ") to rewarding the *insight* that uncertainty propagation drives the constraint. |
| 2 | 0.25 | 0.30 | M33 X-7 exclusion curve P_excl(μ) via MC integration + superradiance condition | Graded on method soundness and the identified excluded μ-window, not pixel-match. |
| 3 | 0.20 | — | **New:** apply the same analysis to IRAS 09149-6206 | Activates the unused SMBH dataset; rewards realizing it probes a much lower μ range (complementary regimes). |
| 4 | 0.25 | 0.50 | Upper limit on self-interaction coupling g(μ) | Weight halved and reframed: rewards derivation + **explicit assumptions (τ_BH, saturation) + sensitivity analysis**; accepts order-of-magnitude agreement instead of exact `g < Y`. |
| 5 | 0.20 | — | **New:** validate/reject synthesis | Where bosons are excluded vs. not, why the two BHs are complementary, and the dominant caveats. |

**Task text:** changed from a method recipe ("develop and apply a novel Bayesian framework that ingests full posteriors...") to a research question requiring the agent to decide how uncertainties enter, use both black holes, and state/test its assumptions.

## Earth_002 — item-by-item

| # | New weight | (was) | Item | Change |
|---|-----------|-------|------|--------|
| 1 | 0.25 | 0.40 | Headline finding (~40–56% at high–severe risk; SSP245<370<585) | Graded on direction/magnitude + **monotonic scenario ordering**, not exact percentages. |
| 2 | 0.25 | — | **New:** methodology design + justification + sensitivity | The core research item — how hazards are quantified/normalized/combined, where thresholds sit, and which conclusions survive reasonable alternative choices. |
| 3 | 0.20 | 0.30 | Global risk maps + hotspot identification | Hotspots judged directionally; pixel pattern need not match. |
| 4 | 0.10 | 0.20 | SSP370 detailed hotspot view | Weight reduced; directional. |
| 5 | 0.10 | 0.10 | Risk-area % bar chart across SSPs | Rewards the escalation trend / mitigation framing. |
| 6 | 0.10 | — | **New:** validate/reject synthesis + conservation prioritization + caveats | Verdict on the emissions-escalation hypothesis tied back to the conservation question. |

**Task text:** reframed as a hypothesis to test (risk escalates with emissions), with index design explicitly the agent's responsibility to invent, justify, and sensitivity-test, using all three SSP scenarios and all provided datasets.

## Suggested follow-ups (not yet done)

- Draft an "ideal solution trajectory" for each (hypothesis → experiment plan → analysis the rubric should reward) to sanity-check the new rubrics against a genuine independent attempt.
- Confirm the scorer's holistic `scientific_capability` keywords reward the new assumption/sensitivity language (rather than a single blessed value) for the reframed items.
- Consider the same redesign pass on other Medium / Medium-High tasks with real research arcs (e.g. Chemistry_000, Energy_002).
