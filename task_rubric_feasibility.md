# Task ⇄ Rubric Feasibility Audit

Question being answered for each task: **given only the task description + the shared `data/` (and `related_work/` for context), would a smart, capable researcher agent — doing genuine independent research, *without ever seeing the target paper* — produce the outputs the rubric (`target_study/checklist.json`) evaluates?**

This is distinct from "can the agent find and re-run the published model/repo." The agent must arrive at the rubric's expected results from first principles on the shared inputs.

## How to read this

For every checklist item we tag what determines its expected output:

- **D — input-determined.** The expected result is fixed by the shared task + data (e.g. "OCR this image", "plot this column", "solve this linear system"). Any capable agent reproduces it.
- **P — paper-private.** The expected result is a specific artifact of the target paper (a number from the paper's own model/dataset, a named method/equation the paper invented, a schematic figure) that is *not derivable* from the shared inputs. Unreachable independently.
- **Mixed** — the direction/structure is D, but the exact value/figure is P or depends on a modeling choice the paper fixed.

**Scoring caveat (applies everywhere):** `evaluation/score.py` anchors both axes at the paper — **50 = on par with the paper**, 100 = "dramatically exceeds." A literal 100 is essentially unreachable by design, so "feasibility" below means *"reproduces what the rubric describes,"* not a numeric 100.

Feasibility ratings: **High / Medium-High / Medium / Low / Very Low**.

---

## Information tasks

| Task | Item (weight, type) | Expected output | Tag | Reproducible? | Review (alignment + issues) |
|------|--------------------|-----------------|-----|---------------|------------------------------|
| **Information_000** (Janus unified MLLM) | 1 (0.3, text) | OCR `equation.png` → LaTeX `A_n = a_0[1+¾Σ(4/9)^k]` | **D** | Yes | Clean: data file ↔ item is 1:1, answer lives in the image. Only mismatch is the *task* framing ("build a framework") vs item ("run OCR"). |
| | 2 (0.3, image) | Generate Janus two-face image (aged/cold vs youthful/warm) | **D** | Yes (cap.) | Well-specified by shared `generation_prompt.txt`; rubric grades prompt attributes, not pixel-match — fair. Gated on having *any* T2I capability, not on knowing the paper. |
| | 3 (0.4, text) | Extract meme text + interpret swole/cheems = decouple vs single encoder | **D** | Yes | Good: text is in image, semantics readable from meme + task wording. Slight risk the judge wants the paper's *exact* framing of the metaphor. |
| **Information_001** (ViCrop) | 1 (0.2, image) | bbox + attention heatmaps; correct date/pose on 2 demo imgs | **Mixed** | Partial | Only item backed by the shared data (2 demo imgs). Direction is reproducible; exact before/after values (02/20/2012, "squatting") are VLM-dependent, so a correct method may still "miss." |
| | 2 (0.4, text) | LLaVA TextVQA 45.18→55.54/54.93% | **P** | No | **Mis-posed.** TextVQA isn't shared and nothing cues the agent to use it; the numbers are artifacts of the paper's specific model+split. 0.4 of weight unreachable. |
| | 3 (0.4, text) | Qwen2.5-3B TextVQA 70.56→77.10%; "identical scores", "added via PR" | **P** | No | **Worst item.** Keywords reward paper *trivia* ("scores identical because resolution dominates", "Qwen support added via pull request") — unknowable and not research. |
| **Information_002** (Hartree-Fock) | 1 (0.3, text) | Construct + second-quantize Hamiltonians | **D** | Yes | Strong alignment: rubric stages mirror the derivation in `data/`. Caveat: the worked solution is shipped, so this tests transcription as much as derivation. |
| | 2 (0.4, text) | Fourier + particle-hole + Wick mean-field decomposition | **D** | Yes | Same — deterministic math forced by the provided Hamiltonian. Task text says "15 papers" but only 1 is provided (over-scoped description). |
| | 3 (0.3, text) | Quadratic reduction → final HF Hamiltonian | **D** | Yes | Same — fully determined; answer key present in `data/_auto.md`. |
| **Information_003** (DIDS-MFL) | 1 (0.15, image) | Fig.2 entanglement evidence (E-GraphSAGE MITM 18% vs DDoS 93%, t-SNE) | **Mixed** | Partial | EDA on the `.pt` can show feature overlap/imbalance (D), but the specific baselines and multi-panel figure are paper-private. Won't match the comparison target. |
| | 2 (0.20, image) | fig.7 framework (stat. disentangle Eq3-7, orth. reg Eq11, PM diffusion Eq14-16) | **P** | No | **Uncheckable.** It's the paper's *invented* architecture as a schematic — no independent agent reproduces these exact named components/equations, and a diagram can't match a pixel-comparison judge. |
| | 3 (0.30, text) | F1 96.0/NMI 92.0 on NF-UNSW; 71.9–125.2% over 14 baselines on 5 benchmarks | **Mixed** | Partial | Headline F1 on the *one provided* benchmark is plausibly reachable; the improvement ranges over 14 named baselines and 5 benchmarks (4 absent) are paper-private. |
| | 4 (0.20, image) | Ablations of SD/RD/MLGRAND (Tables IV/V, t-SNE Fig10/11) | **P** | No | **Mis-posed.** You can't ablate components the agent never built; requires reimplementing the exact paper framework first. |
| | 5 (0.15, text) | DIDS-MFL 99.52% vs GPT 66.67% on 100 samples | **Mixed** | Partial | Concept/direction (trained ≫ LLM on structured traffic) is reproducible; exact numbers and the 100-sample setup are paper-private. |

**Information verdict**

| Task | Input-determined rubric weight | Overall feasibility |
|------|-------------------------------|---------------------|
| Information_002 | ~100% | **High** (answer also shipped) |
| Information_000 | ~100% | **High** — gated only on generation capability |
| Information_001 | ~20% | **Low** — 0.8 of weight grades a benchmark not in the data |
| Information_003 | ~10–20% | **Very Low** — most weight grades the paper's private method/figures |

001 and 003 are **mis-posed** for independent research: the bulk of their rubric weight encodes datasets, baselines, equations, and figures that are not present in or derivable from the shared inputs.

---

## Astronomy tasks

A structural difference from Information: the Astronomy `data/` files were **curated or synthesized to contain exactly the ingredients the rubric scores** (best-fit parameters, pre-extracted figure points, synthetic distributions matching the paper's medians). This makes them well-posed for independent reproduction — sometimes to the point of being a plotting/solving exercise rather than open research.

| Task | Item (weight, type) | Expected output | Tag | Reproducible? | Review (alignment + issues) |
|------|--------------------|-----------------|-----|---------------|------------------------------|
| **Astronomy_000** (ULB superradiance) | 1 (0.2, text) | Characterize posteriors: M±σ, a*±σ for M33 X-7 | **D** | Yes | Trivially aligned — read the shared `.dat` and summarize. Free points. |
| | 2 (0.3, image) | Exclusion curve P_excl(μ) via MC integration + superradiance condition, 95% line | **Mixed** | Likely | Genuine research item. Physics is standard/derivable from related work; reproducibility hinges on implementing superradiance rates correctly + assuming a BH lifetime τ_BH (a choice the paper fixed). Curve *shape* reachable, exact placement sensitive. |
| | 3 (0.5, text) | Upper limit on self-interaction coupling g(μ) < Y GeV⁻¹ | **Mixed** | Hard | Highest weight on the hardest, most modeling-dependent step. Needs self-interaction/saturation theory beyond item 2; the specific g-limit depends on assumptions not fully pinned by the data. Fair as research, risky for "full points." |
| **Astronomy_001** (EDE / DESI) | 1 (0.2, text) | Gaussian MCMC chains from provided best-fits (20k pts, GetDist) | **D** | Yes | Fully determined — means/σ are in the data file. Note: "MCMC chains" are just Gaussian draws by design; tests pipeline mechanics, not inference. |
| | 2 (0.4, image) | Triangle plot of ΛCDM/EDE/w0wa (H0≈70.9 EDE, 63.5 w0wa) | **D** | Yes | Clean: plot the chains; all target values present in data. Strong data↔rubric alignment. |
| | 3 (0.4, image) | Distance plot: D_V/r_d, F_AP, Δμ vs DESI BAO + Union3 points | **Mixed** | Mostly | Data points pre-extracted in file; requires standard, derivable BAO-distance computation per model. Reproducible; the only item needing real cosmology calc. |
| **Astronomy_002** (H0 Distance Network) | 1 (0.2, text) | Build GLS: ~20 eqs, 14 params, dof=6 | **Mixed** | Mostly | System determined by the shared measurements + task method. Exact "20 eqs / 14 params / dof=6" depends on how the agent formulates it — a correct alternative formulation could score lower. |
| | 2 (0.5, text) | Solve → H0=73.48±0.81, M_SNIa=−19.25, M_SBF=−1.52, χ²=1.02 | **Mixed** | Likely-ballpark | GLS solution is data-determined; numbers look engineered to land at ~73.5. Exact match (esp. M_SNIa, χ²) needs the same weighting/formulation as item 1 — couples the two items. |
| | 3 (0.3, image) | Host residual plot (±0.1 mag, NGC1365/M101 consistency) | **D** | Yes | Downstream of item 2: residuals = solution − measurements. Reproducible once the GLS is solved. |
| **Astronomy_003** (SXS waveform catalog) | 1 (0.4, image) | fig6 histogram, log-normal, median 4×10⁻⁴, log y-axis | **D** | Yes | Trivial: histogram of a provided synthetic column. **Caveat:** rubric text says "3756 simulations" but data has 1500 rows — caption number can't match. Tests plotting, not research. |
| | 2 (0.3, image) | fig7 per-ℓ distributions, median rising ℓ=2→8 | **D** | Yes | Plot the 7 provided columns; the ℓ-trend is baked into the synthetic data. Fully aligned but not discovery. |
| | 3 (0.3, image) | fig8 N2vsN3 vs N2vsN4 (median 2×10⁻⁵ / 5×10⁻⁵) | **D** | Yes | Plot the 2 provided columns. Same: reproducible by construction. |

**Astronomy verdict**

| Task | Input-determined rubric weight | Overall feasibility |
|------|-------------------------------|---------------------|
| Astronomy_003 | ~100% | **Very High** — pure histogram-of-provided-synthetic-data |
| Astronomy_001 | ~90% | **High** — chains/plots from provided params; minor cosmology calc |
| Astronomy_002 | ~80% | **Medium-High** — self-contained GLS; exact numbers need matching formulation |
| Astronomy_000 | ~50% | **Medium** — genuine physics; item 3 (0.5 weight) is modeling-dependent |

**Astronomy notes / flags**
- **003**: data row counts (1500/1200) vs rubric text ("3756 simulations") are inconsistent — doesn't block histogram reproduction but the figure caption number won't match. Also: this task tests plotting pre-cooked synthetic data, not research.
- **001**: the "MCMC chains" are Gaussian draws from provided best-fits by design — a fidelity-of-visualization task more than discovery.
- **002**: numbers appear engineered so the GLS lands at ~73.5; reproducibility hinges on the agent formulating the system the paper's way.
- **000**: the only Astronomy task that demands real, hard physics derivation from raw posteriors — best-posed as actual *research*, but correspondingly the hardest to fully reproduce.

---

## Chemistry tasks

These lean on **implementing the paper's novel method** (KA-GNN, AlphaFold 3, LES) or **running heavy external software** (HADDOCK3). The recurring split: the method may be describable/implementable, but the rubric's *headline numbers* (success rates over many targets, "outperforms baseline by X%") are paper-private benchmarks the shared single-/few-sample data can't yield.

| Task | Item (weight, type) | Expected output | Tag | Reproducible? | Review (alignment + issues) |
|------|--------------------|-----------------|-----|---------------|------------------------------|
| **Chemistry_000** (KA-GNN molecular prop.) | 1 (0.45, image) | KA-GNN beats GCN/GAT across 7 datasets, SOTA ROC-AUC table | **Mixed** | Partial | Method is implementable from the task description; 5 of the 7 datasets are provided (real MoleculeNet CSVs). But "outperforms / SOTA" is an empirical *claim* — a faithful re-implementation may not actually beat baselines, and 2 datasets are missing. Heavy ML build. |
| | 2 (0.35, text) | Fourier KAN > B-spline > polynomial; Theorem 2; lower runtime | **Mixed** | Partial | Requires implementing 3 KAN variants + timing. KAN universal approximation is general knowledge, but "Theorem 2" and the exact ranking are paper-specific. |
| | 3 (0.20, text) | KA-GAT saliency identifies fluoro/amide/aromatic groups | **Mixed** | Partial | Saliency/GNNExplainer is standard; the *specific* groups surfaced are model/data-dependent and may not match the paper's. |
| **Chemistry_001** (AlphaFold 3) | 1 (0.30, text) | AF3 76.4% success vs Vina 52.1%, p<0.001, multiple test sets | **P** | No | **Mis-posed.** Building AF3 is infeasible; 76.4% is a benchmark over many targets but only one sample (2l3r) is shared. Unreachable. |
| | 2 (0.25, image) | Predicted complex viz: H-bonds, hydrophobic, π-stacking | **Mixed** | Partial | Agent can visualize/annotate the *provided experimental* structure, but the rubric wants a *prediction* vs experiment — and no predictor is runnable. |
| | 3 (0.20, text) | Protein–nucleic acid 85.2%, antibody–antigen 78.9% | **P** | No | Benchmark numbers; no such data provided. |
| | 4 (0.15, image) | Training convergence vs AF2, diffusion-step optimization | **P** | No | Can't train AF3; target curve lives only in `target_study/`. |
| | 5 (0.10, text) | Ablation: diffusion + PairFormer, >15% degradation | **P** | No | Requires training AF3 variants. |
| **Chemistry_002** (HADDOCK3) | 1 (0.30, image) | ΔHADDOCK vs SKEMPI ΔΔG scatter, 28 muts, Pearson 0.60 | **Mixed** | Partial | Data (1brs + SKEMPI) is present and HADDOCK3 is open software — reproducible *if* the agent installs/runs it (CNS setup is nontrivial). Exact r=0.60 is run-dependent. |
| | 2 (0.20, text) | Top-5 hot-spot residues (Arg59 −25.3, His102 −20.2, …) | **Mixed** | Partial | Same dependency: produced by the alanine scan; specific Δscores vary by run/protocol. |
| | 3 (0.20, text) | ΔHADDOCK range −25.3 to +5.9, qualitative SKEMPI agreement | **Mixed** | Partial | Same — reachable if the scan runs; the range is run-dependent. |
| | 4 (0.15, text) | Antibody–antigen DockQ 0.88; protein–glycan 60% top10 | **P** | No | Different complexes (no antibody–antigen/glycan data shared); paper case studies. |
| | 5 (0.15, text) | CAPRI round 57 target 268, DockQ 0.72, consensus scoring | **P** | No | A competition result — not reproducible from provided data. |
| **Chemistry_003** (LES / latent Ewald) | 1 (0.30, image) | Random charges: recover charges R²≈1, force MAE<0.05, parity plot | **Mixed** | Partial | Synthetic data provided *and `true_charges` are embedded in the `.xyz`* (a leak — parity could be trivially "passed"). Genuine recovery needs implementing CACE+LES, which is substantial. "~40% better than 4G-HDNNP" is paper-private. |
| | 2 (0.30, image) | Charged dimer binding curve matches reference within 0.05 eV | **Mixed** | Partial | Data designed so LES works; requires building LES + the reference Coulomb+LJ curve. Reachable in principle, heavy. |
| | 3 (0.30, image) | Ag₃⁺/Ag₃⁻ PES separation (~0.3 eV) vs Morse+Coulomb | **Mixed** | Partial | Same — implement charge-state-aware model on provided synthetic data; the ~0.3 eV split is data-determined if the method is built right. |
| | 4 (0.10, text) | Large-scale MD: O(N log N), energy drift <0.01 meV/atom/ps | **P** | No | No MD data shared (water/ionic-liquid claims); unreachable. |

**Chemistry verdict**

| Task | Input-determined rubric weight | Overall feasibility |
|------|-------------------------------|---------------------|
| Chemistry_002 | ~70% (gated on running HADDOCK3) | **Medium** |
| Chemistry_000 | ~60% (method derivable; outcome uncertain) | **Medium** |
| Chemistry_003 | ~50% (synthetic data helps; method heavy + a leak) | **Low–Medium** |
| Chemistry_001 | ~10% | **Very Low** — asks to rebuild AF3 and match its benchmarks from one sample |

**Chemistry flags**
- **001** is the category's mis-posed extreme (parallel to Information_003): full-model + multi-benchmark reproduction from a single example.
- **003** embeds `true_charges` in the input `.xyz`, so the "charge recovery" item can be satisfied without the model actually learning them — a scoring loophole.
- **000/003** require building the paper's *novel* architecture; feasibility hinges on whether the task description is a sufficient spec (it roughly is) and whether the agent has the compute to train.
- **002** is the best-posed: real data + open software, but the "extra case study" items (4,5) grade results from complexes that aren't in the data.

---

## Energy tasks

The strongest category for independent reproduction: most rubric items are **self-contained pipelines on curated data using open tools** (PyBaMM, PyPSA, GeoPandas, sklearn). Expected *exact* numbers are run-specific, but the judge rewards reproducing the result/trend, which the data determines.

| Task | Item (weight, type) | Expected output | Tag | Reproducible? | Review (alignment + issues) |
|------|--------------------|-----------------|-----|---------------|------------------------------|
| **Energy_000** (MMGA battery param. ID) | 1 (0.30, text) | LHS 20 samples → PyBaMM 1C sims, 20 valid pairs, 111.50 s | **Mixed** | Mostly | Method fully described; PyBaMM is open. Structure reproducible; "111.50 s" and "100% valid" are run artifacts. Note: rubric centers on *PyBaMM-simulated* data, while the heavy provided NASA/CALCE/Oxford datasets are barely used — data↔rubric mismatch. |
| | 2 (0.30, text) | 4-layer NN surrogate, 500 epochs, MSE 0.0018→0.00025 | **Mixed** | Mostly | Standard surrogate training; exact MSE run-dependent but low error is reachable. |
| | 3 (0.40, image) | GA inverse ID, RMSE 0.0117, heat-coef error 0.03%, true vs ID | **Mixed** | Mostly | ANN+GA identification is reproducible on a synthetic target; specific RMSE/error% won't match exactly. |
| **Energy_001** (PyPSA-GB) | 1 (0.10, text) | 20-bus model, 50 GW Scottish wind, 1500 MW boundary links | **D** | Yes | **Verified:** the provided CSVs *encode* this scenario (20 buses, Bus1–5 = 10 GW each). Item ≈ "load and describe the data." |
| | 2 (0.30, text) | LOPF with/without constraints → 45.94% curtailment (~paper 44.5%) | **D** | Yes | Determined by data + a standard PyPSA LOPF run. Reproducible; curtailment value follows from the encoded scenario. |
| | 3 (0.20, image) | fig5a dispatch vs curtailment stacked area | **D** | Yes | Plot of LOPF output; reproducible. |
| | 4 (0.20, image) | fig5b boundary-link loading ~100% | **D** | Yes | Plot of LOPF output; reproducible. |
| | 5 (0.20, text) | "PyPSA-GB is the first open-source high-res GB model" narrative | **Mixed** | Partial | A significance *claim*, not a result; partially assertable, not really verifiable from the case study. |
| **Energy_002** (African green H₂ LCOH) | 1 (0.40, image) | LCOH map: Namibia/Morocco/SA cost advantages | **Mixed** | Mostly | Inputs (resource potential, distances) are in the CSV; task specifies the LCOH model. Spatial pattern is data-determined; absolute $ depends on financing assumptions the rubric doesn't pin (so trend is enough). |
| | 2 (0.25, text) | African H₂ competitive for export vs fossil routes | **Mixed** | Mostly | Conclusion derivable from the LCOH results; threshold for "competitive" is assumption-dependent. |
| | 3 (0.15, text) | Cost driven by infrastructure accessibility (road/grid/port) | **D** | Yes | Distances are columns; a sensitivity analysis shows this directly. |
| | 4 (0.20, image) | min-LCOH map (contoured), South Africa advantage | **Mixed** | Mostly | Near-duplicate of item 1 (same keywords) — redundant rubric entry; same reproducibility. |
| **Energy_003** (HEEW dataset) | 1 (0.30, image) | Inject 5% missing/2% outliers, TCCA+CRAC clean, K-means k=3 recovers clusters | **Mixed** | Mostly | Corruption is self-defined; TCCA/CRAC are paper-named but *generic* imputation+outlier correction achieves the same recovery. Reachable. |
| | 2 (0.30, image) | Pearson corr: elec-temp r≈0.75, PV-temp 0.60, PV-humidity −0.40 | **D** | Yes | Directly computed from the provided data; r-values are data-determined. |
| | 3 (0.40, image) | Hierarchical check: Σ BN001–010 = CN01, zero error, overlap 2014-01-15 | **D** | Yes | Pure verification on provided files; trivially reproducible. |

**Energy verdict**

| Task | Input-determined rubric weight | Overall feasibility |
|------|-------------------------------|---------------------|
| Energy_003 | ~85% | **High** — mostly analysis on a curated mini-dataset |
| Energy_001 | ~80% | **High** — data encodes the scenario; run PyPSA + plot |
| Energy_002 | ~70% | **Medium-High** — build LCOH from provided site data; trends data-determined |
| Energy_000 | ~65% | **Medium-High** — self-contained PyBaMM+ANN+GA; exact numbers run-specific |

**Energy flags**
- **000**: ships large real battery datasets that the checklist barely uses (it grades a PyBaMM synthetic pipeline) — a data↔rubric mismatch, though the task itself is reproducible.
- **001**: cleanest design — the scenario is fully baked into the CSVs; almost purely input-determined.
- **002**: items 1 and 4 are near-duplicate maps with identical keywords (redundant weight).
- **003**: like Astronomy, the mini-dataset is curated so two of three items are direct data checks.

---

## Earth tasks

A wide spread: one task is explicitly designed as reproduce-from-CSV (001), one ships the reconciled answer (000), one is a genuine geospatial-modeling task with an under-specified method (002), and one demands full-benchmark skill metrics from a single forecast case (003).

| Task | Item (weight, type) | Expected output | Tag | Reproducible? | Review (alignment + issues) |
|------|--------------------|-----------------|-----|---------------|------------------------------|
| **Earth_000** (GlaMBIE glacier mass) | 1 (0.40, text) | Cumulative −6542±387 Gt; rate +36%; 2023 max 548 Gt; vs ice sheets | **Mixed** | Mostly | **Leak:** `results/calendar_years/0_global.csv` already holds the reconciled series — the Gt numbers are a direct aggregation, not a reconciliation. The "vs Greenland/Antarctic" comparison needs external data (P). |
| | 2 (0.10, image) | Regional shares: Alaska 22%, Canadian Arctic 20%, … | **D** | Yes | Computed from the provided per-region result CSVs. |
| | 3 (0.20, image) | Method intercomparison (DEM vs altimetry vs gravimetry diffs) | **Mixed** | Mostly | Per-method estimates are in `input/`; the specific differences (0.08±0.08) are derivable but sensitive to how the agent matches the paper's grouping. |
| | 4 (0.30, image) | Comparison with IPCC AR6 projections; 25%→50% loss by 2100 | **P** | No | Needs IPCC AR6 model-ensemble projection data not in the dataset; observations alone can't reproduce this. |
| **Earth_001** (cloud-seeding records) | 1 (0.22, text) | 832 records, 2000–2025, 13 states; LLM extraction 98.38% accuracy | **Mixed** | Mostly | Counts/coverage are direct from CSV (**D**) — though data has **831 rows, not 832**. The 98.38% extraction accuracy (manual n=200) is paper-private (P). |
| | 2 (0.20, image) | Choropleth: CA/CO/UT dominate + jittered points | **D** | Yes | Aggregate-by-state + map; task is *explicitly* "recover from the published dataset." |
| | 3 (0.20, image) | Annual trend: peak mid-2000s, decline, rebound post-2021 | **D** | Yes | Year counts from CSV. |
| | 4 (0.19, image) | Purpose mix: snowpack + precip >70% | **D** | Yes | Parse/normalize purpose field, count. |
| | 5 (0.19, image) | Agent–apparatus heatmap: silver iodide dominant | **D** | Yes | Cross-tab from rows. |
| **Earth_002** (mangrove risk index) | 1 (0.40, text) | 40–56% mangroves high–severe risk; SSP245<370<585 | **Mixed** | Partial | All inputs present (mangroves, SLR .nc, TC tracks), but the composite index (score 1–5 thresholds, how TC regime shift is quantified) is paper-specific → exact percentages hard; direction reachable. |
| | 2 (0.30, image) | Global risk maps; hotspots SE Asia/C.America/etc. | **Mixed** | Partial | Hotspots are physically determined (mangrove∩TC∩SLR), so directionally reproducible; exact classes depend on the index. |
| | 3 (0.20, image) | SSP370 hotspot detail (Philippines/Vietnam/Mozambique) | **Mixed** | Partial | Same dependency on index definition. |
| | 4 (0.10, image) | Risk-area % bar chart, severe ~1%→13% | **Mixed** | Partial | Threshold-sensitive; the escalation trend is reachable. |
| **Earth_003** (FuXi forecasting) | 1 (0.20, text) | Skillful lead time Z500 9.25→10.5 d, T2M 10→14.5 d; beats EM on 67.92%/53.75% of 240 combos | **P** | No | Aggregate skill over a full 2018 test set vs ECMWF/GraphCast; only **one** forecast case + the FuXi output are shared. Unreachable. |
| | 2 (0.30, image) | Fig1 ACC/RMSE curves vs HRES/GraphCast, 15 d | **P** | No | Needs test set + baseline forecasts + ERA5 verification — none provided. |
| | 3 (0.20, image) | Fig2 FuXi vs ECMWF EM differences | **P** | No | Same. |
| | 4 (0.30, image) | Fig3 spatial Z500 maps (ERA5 vs HRES vs FuXi) at 1–10 d | **Mixed** | Partial | FuXi fields *are* plottable from the provided `006.nc`, but the ERA5/HRES comparison panels aren't (no ground truth/baseline). |

**Earth verdict**

| Task | Input-determined rubric weight | Overall feasibility |
|------|-------------------------------|---------------------|
| Earth_001 | ~80% | **High** — purpose-built as reproduce-from-CSV |
| Earth_000 | ~55% | **Medium-High** — results shipped (leak); item 4 needs external projections |
| Earth_002 | ~50% | **Medium** — full data, but paper-specific index → exact % hard |
| Earth_003 | ~10% | **Very Low** — full-benchmark skill metrics from one forecast case |

**Earth flags**
- **000**: the reconciled `0_global.csv` makes the headline item a lookup/aggregation, not the reconciliation the task describes (leak parallel to Information_002 / the GlaMBIE "results" dir).
- **001**: best-posed in the whole suite — the task statement *is* "recover the paper's conclusions from this CSV"; only the LLM-accuracy keyword reaches outside the data. Minor count bug (831 vs 832).
- **002**: genuine research, but the risk-index methodology is under-specified, so exact percentages are a matching exercise.
- **003**: same failure mode as Information_001/003 and Chemistry_001 — single sample vs full evaluation-suite rubric.

---

## Life tasks

Two well-posed analysis/tool tasks (003, 002), one heavily curated simulation task (001), and one wet-lab-heavy task whose rubric grades physical results the agent can't produce (000).

| Task | Item (weight, type) | Expected output | Tag | Reproducible? | Review (alignment + issues) |
|------|--------------------|-----------------|-----|---------------|------------------------------|
| **Life_000** (ML hydrogel design) | 1 (0.15, text) | R1-max >1 MPa underwater, order-of-magnitude gain | **P** | No | Wet-lab achievement; agent can't synthesize. Data may *contain* a >1 MPa row, but the claim is experimental. |
| | 2 (0.10, text) | "Golden Triangle" BA/PEA/ATAC design principle | **Mixed** | Partial | SHAP/feature importance on the data can surface these monomers; the named principle + mechanism is paper framing. |
| | 3 (0.10, text) | Ideal random copolymerization, reactivity ratios ≈1 | **P** | No | Polymer-chemistry claim not derivable from composition↔strength data. |
| | 4 (0.15, image) | Fig1 workflow + literature comparison chart | **Mixed** | Partial | Workflow is drawable; comparison chart from data; but "vs literature" values are external. |
| | 5 (0.10, image) | Fig2 bioinformatics of 24,707 adhesive proteins | **P** | No | That protein corpus isn't in the data (only hydrogel experiments). |
| | 6 (0.10, image) | Fig3 initial 180-hydrogel screening, G-max 147 kPa | **D** | Yes | The 180-sample data is provided; plot distribution + max. |
| | 7 (0.15, image) | Fig4 ML opt: RFR-GP, UMAP, SHAP | **D** | Yes | Train surrogate + UMAP/SHAP on provided data; method described. |
| | 8 (0.15, image) | Fig5 R1-max mechanical/multi-substrate/durability | **P** | No | Wet-lab characterization not in data. |
| **Life_001** (NeoAgDT vaccine) | 1 (0.35, image) | MinSum, 10k cells, 7 patients violin response dist. | **Mixed** | Mostly | Response-likelihood + vaccine-element CSVs are provided (incl. selected elements), so violins are plottable. Flag: provided reps are "100-cells × 10x" while rubric says 10,000 cells. |
| | 2 (0.20, image) | Coverage ratio vs threshold, 95% CI over 10 reps | **D** | Yes | Computed from the provided response likelihoods. |
| | 3 (0.15, image) | Runtime scaling, ≤10 s at 5k cells | **D** | Yes | `optimization_runtime_data.csv` is shipped — just plot it. |
| | 4 (0.30, text) | Recall vs 11 ranking methods (NetMHCpan…), Tran 2015 | **P** | No | Needs running 11 external tools + experimentally validated neoantigens — neither provided. |
| **Life_002** (Foldseek-Multimer) | 1 (0.20, text) | 9 chain-to-chain matches 7xg4↔6n40 | **D** | Yes (tool) | Determined by the two provided structures; reproducible by running open Foldseek-Multimer. |
| | 2 (0.30, text) | TM-score 0.82 (≥0.65 threshold) | **D** | Yes (tool) | Tool output on the provided pair; value is structure-determined. |
| | 3 (0.20, text) | Rotation/translation vectors consistent | **D** | Yes (tool) | Same alignment output. |
| | 4 (0.20, text) | Runtime 0.8 s vs US-align 12 s, >10× | **Mixed** | Partial | Speedup direction reproducible; absolute times machine-dependent; needs both tools. |
| | 5 (0.10, text) | Seq identity ~4.5%, twilight-zone detection | **D** | Yes | Computable from the alignment/sequences. |
| **Life_003** (Uncalled4 nanopore) | 1 (0.30, image) | Substitution-profile heatmaps (central base, purine/pyrimidine, dual-reader) | **D** | Yes | Pore-model CSVs (k-mer current stats) are provided; profiles computed directly. |
| | 2 (0.30, image) | Performance benchmark: 1.3–6.8× faster, 20× smaller | **D** | Yes | `performance_summary.csv` is shipped — plot it. |
| | 3 (0.40, image) | m6A PR curves: Uncalled4 AUPRC ~0.60 vs Nanopolish ~0.50 | **D** | Yes | Both prediction CSVs + `m6a_labels.csv` provided → compute PR/AUPRC directly. |

**Life verdict**

| Task | Input-determined rubric weight | Overall feasibility |
|------|-------------------------------|---------------------|
| Life_003 | ~100% | **High** — all items computable from shipped CSVs |
| Life_002 | ~80% | **Medium-High** — run Foldseek-Multimer on 2 structures |
| Life_001 | ~70% | **Medium-High** — plot provided simulation data; item 4 external |
| Life_000 | ~35% | **Low–Medium** — half the rubric grades wet-lab/external results |

**Life flags**
- **003**: outputs pre-shipped as CSVs (predictions, labels, performance) — a pure analysis task, very reproducible but low on actual research.
- **002**: clean tool-run task; only the runtime item is environment-dependent.
- **001**: heavy intermediate data provided (even selected vaccine elements), making the optimization items near-lookups; item 4 reaches outside. Cell-count mismatch (100×10 vs 10,000).
- **000**: the mis-posed one — task says "de novo design hydrogels achieving >1 MPa," but synthesis/characterization is physical; only the data-mining/ML figures are reachable.

---

## Material tasks

A recurring pattern here: the `data/` is a **single file curated/engineered so the rubric numbers are reachable** (000 synthetic graphs, 001 toy arrays, 002 a parameter sheet + model download link). The genuinely hard one (003) ships data covering only a small sub-task.

| Task | Item (weight, type) | Expected output | Tag | Reproducible? | Review (alignment + issues) |
|------|--------------------|-----------------|-----|---------------|------------------------------|
| **Material_000** (altermagnet search) | 1 (0.20, text) | 3 datasets (5k/2k/1k, 5% pos, ~50 hidden pos) | **D** | Yes | Data is *provided* (rubric says "generated" but it's shipped); item ≈ verify/describe it. |
| | 2 (0.30, text) | GNN: CGCNN layers, gating, residual, SSL decoder, classifier | **D** | Yes | Standard architecture, fully described in task; implementable. |
| | 3 (0.50, image) | Pretrain loss 0.25→0.05, val acc ~56%, discovery ~60% (p>0.9) | **Mixed** | Mostly | Train on the synthetic data + plot. Bar is low (acc ~56%); planted positives make discovery achievable. Exact numbers run-dependent. |
| **Material_001** (multimodal materials AI) | 1 (0.30, text) | GNN formation-energy MAE 0.15 eV, loss <0.012 | **Mixed** | Mostly | Data is a **toy/dummy array set** engineered for these workflows; train a small GNN. MAE target may not land exactly on degenerate data. |
| | 2 (0.35, image) | VAE structures, lattice 5.1–5.9 Å, KL 0.15, 85% coverage | **Mixed** | Mostly | Lattice values in data are literally 5.12–5.90; VAE reproduces the range by construction. |
| | 3 (0.35, image) | Bayesian opt finds 352.4°C/19.8 bar (≈350/20), TOF 9.8 | **D** | Yes | The data **specifies** the objective's optimum as (350, 20); BO recovers it. Effectively input-determined. |
| **Material_002** (MACE-MP-0 foundation) | 1 (0.33, image) | Water O–O RDF, first peak ~2.8 Å, height ~2.5 (zero-shot MD) | **Mixed** | Mostly | The data file **links the released MACE-MP-0 model + all MD params** — so it's "download model + run ASE MD," not build-from-scratch. Reachable; gated on installing `mace` + compute. |
| | 2 (0.34, image) | O/OH adsorption scaling on 6 metals, slope ≈0.71 | **Mixed** | Mostly | Same: provided lattice constants/slab params + downloadable model → run relaxations. Heavy but specified. |
| | 3 (0.33, image) | CRBH20 reaction barriers, MAE ~0.3 eV | **Mixed** | Partial | Same path; but rubric itself admits "simplified geometries" won't match — true CRBH20 geometries aren't shipped. |
| **Material_003** (vitrimer inverse design) | 1 (0.30, image) | Framework: dual-encoder VAE, 1M chemistries, MD Tg for 8424 | **P** | No | Needs the 1M dataset + 8424 MD runs; only small calibration/MD CSVs shared. Also an overview schematic. |
| | 2 (0.30, image) | BO in VAE latent space → novel vitrimers (Tg>500/373/248 K) | **Mixed** | Partial | Requires the trained VAE (needs the large dataset) + MD validation. Out of reach from shipped data. |
| | 3 (0.25, image) | Experimental synthesis Tg=323 K (measured 311–317 K), healability | **P** | No | Wet-lab synthesis/characterization. |
| | 4 (0.15, image) | GP calibration (Tanimoto kernel), LOOCV MAE ~13 K | **D** | Yes | `tg_calibration.csv` has SMILES + exp Tg + MD Tg → train GP, LOOCV directly. |

**Material verdict**

| Task | Input-determined rubric weight | Overall feasibility |
|------|-------------------------------|---------------------|
| Material_000 | ~85% | **Medium-High** — self-contained ML on shipped synthetic data, low bar |
| Material_001 | ~80% | **Medium-High** — toy data engineered to hit the three workflow targets |
| Material_002 | ~65% | **Medium** — download specified model + run ASE experiments (compute-gated) |
| Material_003 | ~20% | **Low** — only the GP-calibration item is reachable; rest needs 1M data + MD + wet-lab |

**Material flags**
- **001**: `data/` is fabricated dummy arrays (sequential floats, cycling lattice values, a BO optimum hard-coded at 350/20) — the task tests running standard workflows, not research; numbers are baked in.
- **002**: well-scoped *because* the data file names the public model and every parameter; the only weak item (3) is one the rubric itself flags as non-matching.
- **000**: "altermagnet" framing is a thin wrapper over a generic pretrain→finetune→screen on synthetic graphs.
- **003**: the only Material task that demands true large-scale infrastructure + experiments; shipped data covers ~15% of the rubric.

---

## Math tasks

One clean numerical-experiment task (001), three that demand a full trained system (tracker / MARL planner / AlphaGeometry) evaluated against benchmarks or baselines the shared data doesn't include.

| Task | Item (weight, type) | Expected output | Tag | Reproducible? | Review (alignment + issues) |
|------|--------------------|-----------------|-----|---------------|------------------------------|
| **Math_000** (SparseTrack MOT) | 1 (0.15, text) | Pseudo-depth definition (bbox-bottom → image-bottom) | **D** | Yes | Concept fully described; implement/explain on the data. |
| | 2 (0.15, text) | Depth Cascade Matching (DCM) hierarchical association | **D** | Yes | Implementable from the description. |
| | 3 (0.70, image) | SOTA on MOT17/20/DanceTrack, +2.0 HOTA vs ByteTrack, opt 4–6 levels | **P** | No | **Dominant weight is unreachable:** MOT17/20/DanceTrack aren't provided (only one 40-frame simulated sequence). The depth-levels ablation could run on the sim, but HOTA/MOTA SOTA can't. |
| **Math_001** (VOS optimization) | 1 (0.40, text) | VOS-Accelerated beats Prox-GD on Lasso (1000×2000), tol 1e-7 | **D** | Yes | Standard FISTA vs PGD on the provided Lasso `.npy`; result follows from data+algorithm. Theory (Lyapunov/ADMM derivation) is *not* in the rubric. |
| | 2 (0.30, image) | Log-scale convergence plot, accelerated below baseline | **D** | Yes | Plot of the runs; reproducible. |
| | 3 (0.30, text) | Robustness on ill-conditioned (cond=10), L1 prox, restart | **D** | Yes | Data is ill-conditioned by construction; implement prox + adaptive restart. |
| **Math_002** (LNS2+RL MAPF) | 1 (0.40, image) | Beats LNS2/LaCAM/EECBS/SCRIMP; >50% success where baselines hit 0% | **Mixed** | Partial | Maps provided, but needs a *trained MARL* policy + 4 baseline solvers; specific success rates are method/training-dependent. Heavy. |
| | 2 (0.30, image) | Fewer colliding pairs vs vanilla LNS2 (−26.3%, random-small 65 agents) | **Mixed** | Partial | Vanilla LNS2 is implementable; the RL variant requires training. CP reduction is model-dependent. |
| | 3 (0.30, text) | Zero-shot generalization to maze/room/warehouse | **P** | No | A property of the trained RL model the agent would have to reproduce. |
| **Math_003** (AlphaGeometry) | 1 (0.40, text) | Solves 25/30 IMO-AG-30 vs Wu 10/30, ≈gold 25.9/30 | **P** | No | Requires building AlphaGeometry (LM trained on 100M synthetic + symbolic engine). 25/30 is the paper's result. |
| | 2 (0.35, text) | 100M synthetic examples, no human demos, verifiable proofs | **P** | No | The paper's training methodology; not reproducible. |
| | 3 (0.25, text) | traceback.py finds unused premise in IMO 2004 P1 → general theorem | **P** | No | Specific to the paper's codebase/result. defs/rules give the DSL but not the construction-proposing LM. |

**Math verdict**

| Task | Input-determined rubric weight | Overall feasibility |
|------|-------------------------------|---------------------|
| Math_001 | ~100% | **High** — textbook FISTA-vs-PGD on provided data; theory excluded from rubric |
| Math_000 | ~30% | **Low** — concepts reachable, but 0.7-weight item needs MOT benchmarks |
| Math_002 | ~30% | **Low** — maps provided but rubric needs trained MARL + 4 baselines |
| Math_003 | ~8% | **Very Low** — requires building AlphaGeometry |

**Math flags**
- **001**: best-posed — the heavy theory in the task description is *not* what the rubric grades; only a standard numerical experiment is, and the data is exactly that problem.
- **000/002/003**: classic "single instance / map set, but benchmark-scale rubric" mismatch; 003 additionally needs a model that can't be built in a sandbox.
- **002**: data coverage is actually good (all map families shipped); the blocker is method complexity (RL training + baselines), not missing data.

---

## Neuroscience tasks

A recurring problem: the rubric assumes **multiple conditions / a full benchmark / a paper-specific pipeline**, but the shared data is a single project, a generic tabular file, or a model ensemble whose analysis is highly specialized.

| Task | Item (weight, type) | Expected output | Tag | Reproducible? | Review (alignment + issues) |
|------|--------------------|-----------------|-----|---------------|------------------------------|
| **Neuroscience_000** (SimBA behavior) | 1 (0.20, image) | PR curves, Attack across 6 conditions (AP>0.85), Sniffing 0.88 | **Mixed** | Partial | Only **one** project (Together_1 = Lab1) is shipped; Lab1 Attack/Sniffing classifiers are trainable, but the 6 "simulated conditions" (with unspecified weight adjustments) aren't derivable. |
| | 2 (0.20, image) | SHAP Lab1 vs Lab2 | **P** | No | Needs the Lab2 condition's data/classifier — not provided. |
| | 3 (0.20, image) | SHAP Male vs Female | **P** | No | Same — second condition missing. |
| | 4 (0.20, image) | SHAP RI vs CSDS | **P** | No | Same. |
| | 5 (0.20, image) | Permutation importance, Lab1 top-15 features | **D** | Yes | Train Lab1 classifier on provided data → permutation importance. |
| **Neuroscience_001** (Drosophila DMN) | 1 (0.15, image) | Connectome→DMN architecture (45,669 neurons, 734 params) | **Mixed** | Partial | 50 pre-trained models **are shipped**, but Fig1 is an architecture schematic (paper figure) — describable, not pixel-matchable. |
| | 2 (0.25, image) | FRI validation, 30/32 ON/OFF cells correct | **Mixed** | Partial | Computable by running provided models on flash stimuli, but needs specialized FRI analysis + connectome metadata; target is the Nature figure. |
| | 3 (0.25, image) | T4/T5 direction selectivity (DSI), TmY novel predictions | **Mixed** | Partial | Very specialized: stimulate models, compute DSI/preferred direction. Heavy domain code; target is paper figure. |
| | 4 (0.25, image) | Ablation: 7 DMN variants, coarse-graining | **P** | No | Requires retraining variants (connectome-only, task-only, E/I/Mix merges) — not in the shipped ensemble. |
| | 5 (0.10, image) | UMAP of 50 models, T4c clusters, Mi4/Mi9 coupling | **Mixed** | Partial | Models available → UMAP feasible, but the mechanistic coupling analysis is intricate; target is paper figure. |
| **Neuroscience_002** (FlyTracing connectivity) | 1 (0.15, text) | FlyTracing dataset stats (3.2e6 μm³, 1.6e6 pairs) vs PNI/Kasthuri | **P** | No | Factual claim about the real EM dataset; agent has a generic simulated CSV. |
| | 2 (0.25, text) | PointNet++ + Connect-Embed best R/P/F1 across scenarios | **P** | No | Needs point clouds/EM + the specific multi-modal pipeline; data is 20-feature tabular. |
| | 3 (0.25, text) | EmbedNet adaptive λ₃, mean rank 1.41 | **P** | No | Paper-specific training ablation; not reproducible from tabular CSV. |
| | 4 (0.20, image) | PR curves on EM blocks with artifacts | **P** | No | Needs EM data + models. |
| | 5 (0.15, image) | Connect-Embed PCA pos/neg pairs | **P** | No | Needs embeddings/EM data. |
| **Neuroscience_003** (DELVE feat. selection) | 1 (0.15, text) | DELVE two-step framework description | **D** | Yes | Method describable/implementable from task. |
| | 2 (0.25, text) | Beats 11 methods (precision@k, kNN acc, pseudotime) across topologies | **P** | No | Needs 11 baselines + simulated linear/bifurcating/tree datasets with ground truth — not provided. |
| | 3 (0.25, text) | Robustness to noise vs Laplacian/MCFS | **P** | No | Needs simulated noise experiments + baselines. |
| | 4 (0.15, image) | PHATE all-features → diffuse embedding | **D** | Yes | Run PHATE on the provided RPE `.h5ad`. |
| | 5 (0.20, image) | PHATE DELVE-selected → smooth; acc 0.96, NMI 0.60, pt 0.66 | **D** | Yes | Implement DELVE on RPE → select → PHATE + metrics; data-determined. |

**Neuroscience verdict**

| Task | Input-determined rubric weight | Overall feasibility |
|------|-------------------------------|---------------------|
| Neuroscience_003 | ~50% | **Medium** — RPE items reproducible; benchmark items need external sims/baselines |
| Neuroscience_000 | ~35% | **Low–Medium** — single project shipped; cross-condition items need unspecified simulated conditions |
| Neuroscience_001 | ~25% | **Low** — models shipped but analysis is highly specialized + targets are paper figures |
| Neuroscience_002 | ~10% | **Very Low** — generic simulated tabular data vs an EM-pipeline rubric |

**Neuroscience flags**
- **002**: the shared `*_simulated.csv` (20 features + label + degradation) is a generic binary-classification placeholder with no relation to the EM/PointNet++/EmbedNet methods the rubric grades — the most severe data↔rubric mismatch in the suite.
- **001**: rare case where heavy artifacts (50 trained models) *are* shipped, yet feasibility stays low because the analyses are paper-specific and the targets are the original Nature figures (incl. schematics).
- **000**: rubric written for a 6-condition multi-site design, but only one SimBA project is provided; 3 of 5 items grade comparisons across conditions that don't exist in the data.
- **003**: clean split — the RPE dataset supports the RPE figures (items 1/4/5), while the 11-method benchmark items reach outside.

---

## Physics tasks

The most uniformly well-posed (and most heavily scaffolded) category: the shared data is curated "reproduction datasets" or real measurement subsets that contain — sometimes literally — the numbers the rubric checks. Mostly fit/plot/compute-from-formula.

| Task | Item (weight, type) | Expected output | Tag | Reproducible? | Review (alignment + issues) |
|------|--------------------|-----------------|-----|---------------|------------------------------|
| **Physics_000** (multi-component icosahedra) | 1 (0.30, image) | Caspar-Klug: T=h²+hk+k², Mackay [1,13,55,147,309] + novel [1,13,45,117,239,431] | **D** | Yes | **Both magic-number sequences are written verbatim in the data txt** — item is compute/verify from given formula. |
| | 2 (0.40, image) | Optimal mismatch: 0.04 (MC→MC), 0.14 (MC→Ch1); atomic pairs | **D** | Yes | Formula + parameters in the data; mismatch values computed directly. |
| | 3 (0.30, image) | Growth sim: symmetry breaking, Na₁₃@Rb₃₂→Ch1, >70% conservative | **Mixed** | Mostly | The only item needing real work — a MC/LJ growth simulation; rules described, phenomena reproducible with effort. |
| **Physics_001** (MATBG superfluid stiffness) | 1 (0.35, image) | D_s vs density, ~10× enhancement, vF_QG 3000 vs conv 700 | **D** | Yes | Simulated D_s data + both vF values are in the data txt; plot + overlay theory curves. |
| | 2 (0.35, image) | D_s(T) power law n≈2.5, rules out BCS | **D** | Yes | Fit provided D_s(T) to power law. |
| | 3 (0.30, image) | D_s(I) quadratic, I_c=50 nA | **D** | Yes | Fit provided D_s(I). |
| **Physics_002** (random circuit sampling) | 1 (0.20, image) | Fidelity vs N at d=12; XEB/log-XEB/MB consistent | **D** | Yes | XEB is a defined formula over the provided results+amplitudes; computable. |
| | 2 (0.20, image) | Fidelity vs depth at N=40 (0.6→0.3) | **D** | Yes | XEB across depths from provided data. |
| | 3 (0.15, image) | N=56 MB regression + mirror circuits | **Mixed** | Partial | Needs mirror-circuit construction; more involved than direct XEB. |
| | 4 (0.20, text) | Gate-counting fidelity model with 1σ band | **Mixed** | Partial | Requires building a gate-level error model; only partially data-determined. |
| | 5 (0.25, image) | N=40 counts-weighted XEB vs depth, 50 instances/depth | **D** | Yes | Direct XEB computation over the provided instances. |
| **Physics_003** (Floquet-Bloch tr-ARPES) | 1 (0.50, image) | E–k map: Dirac cone + replica band, ~1 ps | **D** | Yes | Raw 4D h5 + `processed_band_data.json` provided → render the map. |
| | 2 (0.30, image) | Replica intensity vs polarization angle, Volkov fit | **D** | Yes | `polarization_dependence_data.csv` shipped (and a `.png` of the figure is in `data/` — a leak); plot + fit. |
| | 3 (0.20, text) | Anisotropy ⇒ Volkov final-state scattering mechanism | **D** | Yes | Interpretation follows from the polarization data. |

**Physics verdict**

| Task | Input-determined rubric weight | Overall feasibility |
|------|-------------------------------|---------------------|
| Physics_003 | ~95% | **High** — data directly yields the figures (target png even shipped) |
| Physics_001 | ~90% | **High** — fit/plot curated simulated data + theory |
| Physics_000 | ~80% | **Medium-High** — magic numbers literally in data; only the growth sim is real work |
| Physics_002 | ~70% | **Medium-High** — XEB from provided data; mirror/gate-model items harder |

**Physics flags**
- **000/001**: the "reproduction datasets" pre-contain the rubric's target numbers (magic sequences, vF values) — tests transcription/plotting more than physics.
- **003**: the target polarization figure itself sits in `data/` alongside its CSV — a direct leak.
- **002**: the genuinely well-designed one — real RCS results+amplitudes, and XEB is a fair, computable reproduction; only the mirror-circuit/gate-model items reach beyond a direct computation.

---

## Master summary (all 40 tasks)

Feasibility of an independent agent producing what the rubric grades, from task + data alone (no target paper). Sorted within each tier.

| Tier | Tasks |
|------|-------|
| **High** | Astronomy_003, Astronomy_001, Energy_003, Energy_001, Earth_001, Life_003, Math_001, Physics_003, Physics_001 |
| **Medium-High** | Information_000, Information_002, Astronomy_002, Energy_002, Energy_000, Earth_000, Life_002, Life_001, Material_000, Material_001, Physics_000, Physics_002 |
| **Medium** | Chemistry_002, Chemistry_000, Earth_002, Material_002, Neuroscience_003 |
| **Low / Low-Medium** | Chemistry_003, Information_001, Life_000, Material_003, Math_000, Math_002, Neuroscience_000, Neuroscience_001 |
| **Very Low** | Information_003, Chemistry_001, Earth_003, Math_003, Neuroscience_002 |

(Counts: ~9 High, ~12 Medium-High, ~5 Medium, ~9 Low, ~5 Very Low.)

## Cross-cutting observations

1. **The single best predictor of feasibility is data curation, not task difficulty.** Tasks where `data/` was synthesized/curated to contain the rubric's answer ingredients (all of Astronomy, Energy_001/003, Earth_001, Life_003, Physics_000/001/003) are reproducible regardless of how hard the underlying science is. Tasks shipping one real sample against a benchmark-scale rubric fail regardless of how "easy" the method is.

2. **The dominant failure mode — "single instance vs benchmark-scale rubric."** Information_001/003, Chemistry_001, Earth_003, Math_000/003, Neuroscience_002 all give one example / one map-set / one tabular placeholder, then grade aggregate metrics (success rates, SOTA tables, skill scores) over datasets and baselines that were never shared. No agent can recover those numbers.

3. **Paper-private artifacts that recur as uncheckable items:** (a) named methods/equations the paper *invented* (DIDS-MFL Eq3-16, EmbedNet λ₃, CACE/LES); (b) comparison tables vs N external baselines (11 methods in Life_001/Neuro_003, 14 in Information_003, 4 in Math_002); (c) schematic/architecture figures matched by a pixel-comparison judge (Information_003 fig.7, Chemistry AF3 curves, Neuro_001 Nature figures); (d) wet-lab results (Life_000, Material_003, Chemistry_001).

4. **The opposite problem — leaks / over-scaffolding.** Several tasks ship the answer: Information_002 (`_auto.md` worked solution), Earth_000 (`0_global.csv` reconciled series), Physics_000 (magic-number sequences in the txt), Physics_003 (target figure png in `data/`), Material_001 (BO optimum hard-coded), Chemistry_003 (`true_charges` embedded). These score high on *fidelity* but test transcription/plotting, not research — and the holistic "scientific capability" axis should (and partly does) penalize that.

5. **Image items are the most fragile.** When the target is a *result plot* the data determines (histograms, PR curves, maps), they're fine; when it's a schematic or a paper-specific multi-panel figure, they're effectively unscorable. Re-typing those to text claims or cutting them would improve many rubrics.

6. **Diagnostic to apply when fixing a rubric item:** "Could the agent know to produce this from the description + `data/` alone?" If it names a dataset, baseline, equation, or number that only appears in the target paper, it grades reproduction-of-the-paper, not research → cut it, or add the supporting data/baseline to the inputs. Conversely, if the answer is sitting in `data/`, the item tests transcription → make the agent *derive* it (withhold the solution) or move the weight to the scientific-capability axis.
