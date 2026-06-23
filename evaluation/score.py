"""Scorer: evaluate a research run on two independent axes using structai.

  * PAPER FIDELITY (per checklist item) — did the agent recover the target paper's
    SPECIFIC result (values, trends, figure, mechanism, conclusion)? Judged from the
    report + images, one call per checklist item.

  * SCIENTIFIC CAPABILITY (holistic, one call per run) — did the agent actually
    perform good *research*? Scored across process stages (problem framing, process
    design, evidence handling, adaptation, synthesis, ...) against the static
    artifacts the agent produced: the report, the code it wrote, the outputs its code
    produced, run metadata, and a distilled execution trajectory. The judge READS
    these artifacts statically — it never executes anything.

Aggregated per run into:
  paper_fidelity_score        = weighted avg of item fidelity scores
  scientific_capability_score = weighted avg of research-stage scores
  total_score                 = SCIENTIFIC_WEIGHT * scientific_capability_score
                              + FIDELITY_WEIGHT   * paper_fidelity_score

Why split the granularity: the checklist items are paper-derived *demonstrations*
(reproduce this figure/output). Reproduction has a real per-item axis (fidelity) but
no per-item "research process" axis — that is a property of the whole effort. So the
research axis is judged once, holistically, against the actual work product, while
fidelity stays per-item. Both axes are anchored at 50 = on par with the target paper.
"""

import json
import os
from pathlib import Path
from typing import Optional

from structai import LLMAgent, multi_thread

from .config import (
    JUDGE_MODEL_NAME,
    IMAGE_EXTENSIONS,
    MAX_IMAGE_SIZE,
    TASKS_DIR,
    SCIENTIFIC_WEIGHT,
    FIDELITY_WEIGHT,
)
from .utils import get_run_workspace, safe_resolve

# --- artifact size caps (keep the research prompt bounded) -------------------
MAX_REPORT_CHARS = 40000
MAX_CODE_CHARS = 16000
MAX_OUTPUTS_CHARS = 12000
MAX_TRAJECTORY_CHARS = 30000
MAX_TOOL_RESULT_CHARS = 220

# --- research-axis stages ----------------------------------------------------
# Each stage is scored 0-100, anchored at 50 = on par with the target paper.
# "evidence" notes where the stage is best observed (informational; the judge sees
# all artifacts at once). Weights set each stage's contribution to the research score.
RESEARCH_DIMENSIONS = [
    {
        "key": "problem_framing",
        "name": "Problem Framing",
        "weight": 1.0,
        "evidence": "report",
        "desc": ("Did the agent correctly understand and frame the research problem — "
                 "the question, hypothesis, scope, and success criteria — given the task "
                 "and the provided data?"),
    },
    {
        "key": "process_design",
        "name": "Process Design",
        "weight": 1.0,
        "evidence": "code + trajectory",
        "desc": ("Was the overall research plan sound and appropriate — sensible method "
                 "selection, controlled comparisons (e.g. testing the core hypothesis "
                 "rather than asserting it), and a coherent path from question to answer?"),
    },
    {
        "key": "experiment_design",
        "name": "Experiment / Implementation Design",
        "weight": 0.5,
        "evidence": "code",
        "desc": ("Were the concrete experiments / implementation well designed and "
                 "correctly built — appropriate baselines, controls, metrics, and code "
                 "that actually implements the intended method?"),
    },
    {
        "key": "evidence_acquisition",
        "name": "Evidence Acquisition",
        "weight": 0.5,
        "evidence": "outputs + trajectory",
        "desc": ("Did the agent actually gather/produce the evidence it needed — running "
                 "code, using the provided data, producing real outputs/logs/figures — "
                 "rather than asserting results it never generated?"),
    },
    {
        "key": "claim_handling",
        "name": "Claim Handling",
        "weight": 1.0,
        "evidence": "report vs outputs",
        "desc": ("Are the report's claims actually backed by the produced evidence? Reward "
                 "claims traceable to real outputs and honest uncertainty; penalize "
                 "fabricated, unsupported, or untraceable numbers and overclaiming."),
    },
    {
        "key": "adaptation",
        "name": "Adaptation / Pivoting",
        "weight": 0.75,
        "evidence": "trajectory",
        "desc": ("When something failed or surprised the agent, did it diagnose and adapt "
                 "sensibly (debug, revise approach, pivot) rather than ignore the problem? "
                 "Best seen in the execution trajectory; do not penalize a clean run that "
                 "genuinely needed no pivots."),
    },
    {
        "key": "synthesis",
        "name": "Final Output / Synthesis",
        "weight": 1.0,
        "evidence": "report",
        "desc": ("Quality of the final synthesis — does the report draw conclusions that "
                 "follow from the evidence, integrate the findings coherently, and honestly "
                 "state limitations and scope?"),
    },
]

_RESEARCH_RETURN_EXAMPLE = {
    d["key"]: {"score": 0, "reasoning": "str"} for d in RESEARCH_DIMENSIONS
}

# --- rubrics -----------------------------------------------------------------
FIDELITY_RUBRIC = """You are a strict scientific peer reviewer for an automated-research benchmark. An AI agent was given a task derived from a target paper and produced a research report. For ONE evaluation criterion (derived from the target paper), score how well the report recovers the paper's SPECIFIC result on a single axis.

=== PAPER FIDELITY (0-100; 50 = reproduces the paper's reported result) ===
Did the agent recover the target paper's SPECIFIC result for this criterion — its claimed values, trends, figure, mechanism, or conclusion?
Reward: matching the paper's specific claims, values, and trends; recovering the same mechanism/interpretation; for image criteria, matching the target figure in variables, trend, scale, labels, and content; and reproductions even more complete/stronger than the paper reported, in the SAME direction / supporting the SAME claim.
Penalize: missing the paper's specific values/trends/figures/mechanisms; a DIFFERENT conclusion than the paper; figures visually or quantitatively inconsistent with the target; absent required artifacts.
Scale (anchored on the target paper):
- 0: the paper's specific result is absent from the report.
- 1-20: mentioned but not actually reproduced.
- 21-40: partial reproduction with notable deviations.
- 41-50: reproduces the paper's specific result, matching what the paper reported; 50 = on par.
- 51-70: reproduces AND improves on the paper's reported numbers/trends in the same direction.
- 71-90: substantially exceeds the paper's reported results while remaining consistent with its claims.
- 91-100: dramatically exceeds the paper's reported results.
A well-justified but DIFFERENT conclusion scores LOW here (it did not reproduce the paper) — that is expected and is rewarded on the separate scientific-capability axis, not here.
Be skeptical of fluent but unsupported text. Substance over style; longer is not better."""

RESEARCH_RUBRIC = """You are a strict scientific peer reviewer for an automated-research benchmark. An AI agent was given a research task (derived from a target paper) and worked autonomously: it wrote and ran code, produced outputs, and wrote a final report. You are given the agent's actual work product — its report, the code it wrote, the outputs its code produced, run metadata, and a distilled log of its execution trajectory (its tool-by-tool actions).

Your job is to judge the QUALITY OF THE RESEARCH PROCESS — not whether it matched the target paper (a separate axis handles reproduction). Score each research dimension below on 0-100.

ANCHORING: 50 = on par with the target paper on that dimension. Above 50 = the agent's research is better than the paper's on that dimension; below 50 = worse. Research as strong as the published paper lands near 50 — a high bar.

GENERAL BANDS (apply per dimension):
- 0: dimension absent / not addressed at all.
- 1-20: token effort only; no real substance.
- 21-40: attempted but with major flaws, OR claims/evidence that are fabricated or untraceable. USE THIS BAND for unsupported or fabricated quantitative claims no matter how polished the prose.
- 41-50: roughly equivalent to the target paper's rigor on this dimension; 50 = on par.
- 51-70: clearly more rigorous / better grounded / more complete than the paper.
- 71-90: significantly stronger, with validation or insight well beyond the paper.
- 91-100: exceptional, far beyond the paper.

CRITICAL EVIDENCE RULE: Trust what the agent ACTUALLY DID over what the report CLAIMS. The code, outputs, and trajectory are ground truth for whether work was really performed. If the report claims a result but no code/output produced it, that is fabrication → cap the relevant dimensions (especially Claim Handling and Evidence Acquisition) in the 21-40 band. Conversely, real reproducible evidence that backs the claims should score well even if the prose is plain.
Honest acknowledgement of limitations and scope is a POSITIVE. Substance over style; longer is not better."""

_RESEARCH_TASK_INSTRUCTIONS = """## Task
Score EACH research dimension on 0-100 (50 = on par with the target paper), using the bands above and the CRITICAL EVIDENCE RULE. Base each score on the agent's actual artifacts (report, code, outputs, trajectory), preferring what the agent demonstrably did over what it merely claims.

Return a JSON object keyed by dimension, each with a score and 2-3 sentence reasoning:
{return_shape}"""

_FIDELITY_TASK_INSTRUCTIONS = """## Task
Score this report against the criterion on the PAPER FIDELITY axis, applying the rubric strictly.

Return a JSON object:
{{"fidelity_reasoning": "<2-3 sentences>", "fidelity_score": <0-100>}}"""

_FIDELITY_RETURN_EXAMPLE = {"fidelity_reasoning": "str", "fidelity_score": 0}


def _read_report(workspace: Path) -> Optional[str]:
    report_path = workspace / "report" / "report.md"
    if report_path.exists():
        return report_path.read_text(encoding="utf-8", errors="replace")
    report_dir = workspace / "report"
    if report_dir.exists():
        for md in report_dir.glob("*.md"):
            return md.read_text(encoding="utf-8", errors="replace")
    return None


def _find_generated_images(workspace: Path) -> list[Path]:
    images = []
    for search_dir in [workspace / "outputs", workspace / "report"]:
        if search_dir.exists():
            for ext in IMAGE_EXTENSIONS:
                images.extend(search_dir.rglob(f"*{ext}"))
    return images


# --- static artifact gathering (no execution) --------------------------------
def _cap(text: str, limit: int) -> str:
    """Truncate text to limit chars, keeping head and tail with a marker."""
    if len(text) <= limit:
        return text
    head = text[: limit * 2 // 3]
    tail = text[-(limit // 3):]
    return f"{head}\n...[truncated {len(text) - limit} chars]...\n{tail}"


def _gather_text_files(files: list[Path], per_file_limit: int, total_limit: int) -> str:
    """Concatenate readable text files with headers, capped per-file and overall."""
    chunks: list[str] = []
    used = 0
    for f in sorted(files):
        if used >= total_limit:
            chunks.append("...[remaining files omitted]...")
            break
        try:
            if not f.is_file() or f.stat().st_size > MAX_IMAGE_SIZE:
                continue
            body = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        body = _cap(body, per_file_limit)
        chunk = f"--- {f.name} ---\n{body}"
        chunks.append(chunk)
        used += len(chunk)
    return "\n\n".join(chunks)


def _gather_code(workspace: Path) -> str:
    """Read the scripts the agent wrote (code/ dir + top-level .py)."""
    files: list[Path] = []
    code_dir = workspace / "code"
    if code_dir.is_dir():
        for ext in (".py", ".sh", ".ipynb"):
            files.extend(code_dir.rglob(f"*{ext}"))
    files.extend(workspace.glob("*.py"))
    return _gather_text_files(files, per_file_limit=6000, total_limit=MAX_CODE_CHARS)


def _gather_outputs(workspace: Path) -> str:
    """Read the outputs the agent's code produced (outputs/ dir)."""
    out_dir = workspace / "outputs"
    if not out_dir.is_dir():
        return ""
    files: list[Path] = []
    for ext in (".json", ".txt", ".log", ".csv", ".md", ".yaml", ".yml"):
        files.extend(out_dir.rglob(f"*{ext}"))
    return _gather_text_files(files, per_file_limit=5000, total_limit=MAX_OUTPUTS_CHARS)


def _meta_summary(workspace: Path) -> str:
    meta_path = workspace / "_meta.json"
    if not meta_path.exists():
        return "(no run metadata)"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return "(unreadable run metadata)"
    keys = ["status", "exit_code", "model", "agent_name", "duration_seconds"]
    parts = [f"{k}={meta[k]}" for k in keys if k in meta]
    return ", ".join(parts) if parts else "(no notable metadata)"


def _summarize_tool_input(name: str, inp: dict) -> str:
    """One-line summary of a tool call's input for the distilled trajectory."""
    if not isinstance(inp, dict):
        return str(inp)[:150]
    for key in ("command", "file_path", "path", "pattern", "query", "url", "description"):
        if key in inp and inp[key]:
            return f"{key}={str(inp[key])[:160]}"
    return json.dumps(inp, default=str)[:160]


def _extract_tool_result_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, str):
                parts.append(b)
            elif isinstance(b, dict):
                if b.get("type") == "text":
                    parts.append(b.get("text", ""))
                elif b.get("type") == "tool_result":
                    # tool results nest their payload under "content"
                    parts.append(_extract_tool_result_text(b.get("content")))
        return " ".join(p for p in parts if p)
    return ""


def _distill_trajectory(workspace: Path) -> str:
    """Distill _agent_output.jsonl (Claude Code stream-json) to a compact action log.

    Keeps assistant reasoning (THINK), tool calls (ACTION), and truncated tool
    results (RESULT). Drops session/token-accounting noise. Statically read only.
    """
    path = workspace / "_agent_output.jsonl"
    if not path.exists():
        return ""
    lines: list[str] = []
    try:
        fh = path.open(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    with fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                d = json.loads(raw)
            except json.JSONDecodeError:
                continue
            t = d.get("type")
            if t == "assistant":
                for b in d.get("message", {}).get("content", []):
                    if not isinstance(b, dict):
                        continue
                    bt = b.get("type")
                    if bt == "text":
                        txt = b.get("text", "").strip()
                        if txt:
                            lines.append(f"THINK: {txt[:400]}")
                    elif bt == "tool_use":
                        lines.append(
                            f"ACTION {b.get('name', '?')}: "
                            f"{_summarize_tool_input(b.get('name', ''), b.get('input', {}))}"
                        )
            elif t == "user":
                txt = _extract_tool_result_text(d.get("message", {}).get("content"))
                txt = txt.strip().replace("\n", " ")
                if txt:
                    lines.append(f"RESULT: {txt[:MAX_TOOL_RESULT_CHARS]}")
            elif t == "result":
                summary = d.get("result") or d.get("subtype") or ""
                if summary:
                    lines.append(f"FINAL: {str(summary)[:400]}")
    return _cap("\n".join(lines), MAX_TRAJECTORY_CHARS)


def _gather_research_artifacts(workspace: Path) -> dict:
    return {
        "code": _gather_code(workspace),
        "outputs": _gather_outputs(workspace),
        "meta": _meta_summary(workspace),
        "trajectory": _distill_trajectory(workspace),
    }


# --- prompt builders ---------------------------------------------------------
def _build_fidelity_prompt(report_text: str, item: dict, instructions: str,
                           is_image: bool) -> str:
    criteria = item.get("content", "")
    keywords = item.get("keywords", [])
    keywords_str = ", ".join(keywords) if keywords else "None specified"
    report_block = (
        f"## AI-Generated Report Text (excerpt)\n{report_text[:10000] if report_text else 'No report text available.'}"
        if is_image else
        f"## AI-Generated Research Report\n{report_text}"
    )
    image_note = (
        "\n## Images\nWhen images are attached, the FIRST image is the ground-truth target "
        "figure from the target paper; all subsequent images are from the AI agent's "
        "workspace/report. Match the agent's figure to the TARGET in variables, trend, "
        "scale, labels, and content. Superficially similar plots with wrong scales, "
        "missing data, or incorrect trends get a LOW fidelity score.\n"
        if is_image else ""
    )
    return f"""{FIDELITY_RUBRIC}

## Research Task Background (INSTRUCTIONS.md given to the AI agent)
{instructions}

## Evaluation Criterion (derived from the target paper)
{criteria}

## Key Aspects to Verify
{keywords_str}

{report_block}
{image_note}
{_FIDELITY_TASK_INSTRUCTIONS}"""


def _build_research_prompt(report_text: str, instructions: str,
                           checklist: list[dict], artifacts: dict) -> str:
    dims_desc = "\n".join(
        f"- {d['name']} (key: {d['key']}): {d['desc']}" for d in RESEARCH_DIMENSIONS
    )
    criteria = "\n".join(
        f"  {i + 1}. {c.get('content', '')[:300]}" for i, c in enumerate(checklist)
    ) or "  (none)"
    return_shape = json.dumps(_RESEARCH_RETURN_EXAMPLE, indent=2)
    report_block = _cap(report_text or "(no report found)", MAX_REPORT_CHARS)
    return f"""{RESEARCH_RUBRIC}

## Research dimensions to score (each 0-100; 50 = on par with the target paper)
{dims_desc}

## Research Task (INSTRUCTIONS.md given to the agent)
{instructions}

## What the target paper demonstrated (reference bar — the paper's specific results)
{criteria}

## The agent's research report (report/report.md)
{report_block}

## The code the agent wrote
{artifacts['code'] or '(no code found)'}

## The outputs the agent's code produced (logs / results)
{artifacts['outputs'] or '(no outputs found)'}

## Run metadata
{artifacts['meta']}

## The agent's execution trajectory (distilled; THINK=reasoning, ACTION=tool call, RESULT=truncated output)
{artifacts['trajectory'] or '(no trajectory captured)'}

{_RESEARCH_TASK_INSTRUCTIONS.format(return_shape=return_shape)}"""


def _clamp(v) -> int:
    try:
        return max(0, min(100, int(v)))
    except (TypeError, ValueError):
        return 0


# --- per-item fidelity + holistic research scoring ---------------------------
def _score_item_fidelity(agent: LLMAgent, report_text: str, item: dict,
                         target_image_path: Optional[Path],
                         generated_images: list[Path],
                         instructions: str) -> dict:
    """Score one checklist item on the paper-fidelity axis (text or image)."""
    is_image = item.get("type", "text") == "image"
    prompt = _build_fidelity_prompt(report_text, item, instructions, is_image)
    if is_image:
        img_paths = []
        if target_image_path and target_image_path.exists():
            img_paths.append(str(target_image_path))
        for img in generated_images[:5]:
            if img.exists() and img.stat().st_size <= MAX_IMAGE_SIZE:
                img_paths.append(str(img))
        result = agent(prompt, image_paths=img_paths if img_paths else None,
                       return_example=_FIDELITY_RETURN_EXAMPLE, max_try=2)
    else:
        result = agent(prompt, return_example=_FIDELITY_RETURN_EXAMPLE, max_try=2)

    if result and isinstance(result, dict):
        return {
            "fidelity_score": _clamp(result.get("fidelity_score", 0)),
            "fidelity_reasoning": str(result.get("fidelity_reasoning", "")),
        }
    return {"fidelity_score": 0, "fidelity_reasoning": "Failed to parse scoring response."}


def _normalize_research_result(result) -> dict:
    """Coerce the holistic research response into {key: {score, reasoning}}."""
    out: dict = {}
    result = result if isinstance(result, dict) else {}
    for d in RESEARCH_DIMENSIONS:
        key = d["key"]
        entry = result.get(key)
        if isinstance(entry, dict):
            score = entry.get("score", 0)
            reasoning = entry.get("reasoning", "")
        else:
            # tolerate flat shape: {"<key>_score": .., "<key>_reasoning": ..}
            score = result.get(f"{key}_score", 0)
            reasoning = result.get(f"{key}_reasoning", "")
        out[key] = {"score": _clamp(score), "reasoning": str(reasoning)}
    return out


def _score_research(agent: LLMAgent, report_text: str, instructions: str,
                    checklist: list[dict], artifacts: dict) -> dict:
    """One holistic call scoring all research stages from the static artifacts."""
    prompt = _build_research_prompt(report_text, instructions, checklist, artifacts)
    result = agent(prompt, return_example=_RESEARCH_RETURN_EXAMPLE, max_try=2)
    return _normalize_research_result(result)


def _combined(scientific: float, fidelity: float) -> float:
    return SCIENTIFIC_WEIGHT * scientific + FIDELITY_WEIGHT * fidelity


def aggregate_scores(checklist: list[dict], fidelity_results: list[dict],
                     research_result: dict) -> dict:
    """Pure aggregation of per-item fidelity + holistic research scores.

    Side-effect free so it is unit-testable. Returns items[] (per-item fidelity),
    research_dimensions[] (per-stage), and run-level scientific/fidelity/total.
    """
    # paper fidelity — weighted average over checklist items
    items = []
    tot_fid = tot_w = 0.0
    for i, (item, fr) in enumerate(zip(checklist, fidelity_results)):
        weight = float(item.get("weight", 1.0))
        fr = fr or {}
        fid = _clamp(fr.get("fidelity_score", 0))
        fid_reason = str(fr.get("fidelity_reasoning", ""))
        items.append({
            "index": i,
            "type": item.get("type", "text"),
            "content": item.get("content", "")[:200],
            "weight": weight,
            "fidelity_score": fid,
            "fidelity_reasoning": fid_reason,
            # legacy fields (back-compat with older consumers/UI)
            "score": fid,
            "reasoning": fid_reason,
        })
        tot_fid += fid * weight
        tot_w += weight
    fid_agg = (tot_fid / tot_w) if tot_w > 0 else 0.0

    # scientific capability — weighted average over research stages
    research_result = research_result or {}
    dimensions = []
    tot_sci = tot_dw = 0.0
    for d in RESEARCH_DIMENSIONS:
        dr = research_result.get(d["key"], {}) or {}
        sc = _clamp(dr.get("score", 0))
        reasoning = str(dr.get("reasoning", ""))
        w = float(d.get("weight", 1.0))
        dimensions.append({
            "key": d["key"],
            "name": d["name"],
            "weight": w,
            "score": sc,
            "reasoning": reasoning,
        })
        tot_sci += sc * w
        tot_dw += w
    sci_agg = (tot_sci / tot_dw) if tot_dw > 0 else 0.0

    total = _combined(sci_agg, fid_agg)
    return {
        "items": items,
        "research_dimensions": dimensions,
        "scientific_capability_score": round(sci_agg, 2),
        "paper_fidelity_score": round(fid_agg, 2),
        "total_score": round(total, 2),
        "total_weight": tot_w,
        "scoring_weights": {"scientific": SCIENTIFIC_WEIGHT, "fidelity": FIDELITY_WEIGHT},
    }


def _make_agent(max_tokens: int) -> LLMAgent:
    return LLMAgent(
        api_key=os.environ.get("JUDGE_API_KEY", ""),
        api_base=os.environ.get("JUDGE_API_BASE", ""),
        model_version=JUDGE_MODEL_NAME,
        system_prompt=(
            "You are a strict scientific peer reviewer for an automated-research "
            "benchmark. Score the agent's work against the rubric; do not attempt to "
            "solve the research task yourself."
        ),
        temperature=0,
        max_tokens=max_tokens,
        time_limit=180,
        max_try=2,
    )


def score_workspace(workspace: str | Path) -> dict:
    """Score a completed run workspace against its task's checklist."""
    workspace = Path(workspace)
    if not workspace.is_dir():
        return {"error": "Workspace not found"}
    meta_path = workspace / "_meta.json"
    if not meta_path.exists():
        return {"error": "Run metadata not found"}
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    run_id = meta.get("run_id")
    if not run_id:
        return {"error": "Run metadata missing run_id"}
    task_id = meta.get("task_id")
    if not task_id:
        return {"error": "Run metadata missing task_id"}
    agent_name = meta.get("agent_name", "Unknown")

    checklist_path = TASKS_DIR / task_id / "target_study" / "checklist.json"
    if not checklist_path.exists():
        return {"error": "Checklist not found for this task"}
    with open(checklist_path, "r", encoding="utf-8") as f:
        checklist = json.load(f)

    report_text = _read_report(workspace)
    if not report_text:
        return {"error": "No report found in workspace"}

    instructions_path = workspace / "INSTRUCTIONS.md"
    instructions = ""
    if instructions_path.exists():
        instructions = instructions_path.read_text(encoding="utf-8", errors="replace")

    generated_images = _find_generated_images(workspace)

    judge_api_key = os.environ.get("JUDGE_API_KEY", "")
    judge_api_base = os.environ.get("JUDGE_API_BASE", "")
    if not judge_api_key or not judge_api_base or not JUDGE_MODEL_NAME:
        return {
            "error": (
                "Judge API configuration is missing. Set JUDGE_API_KEY, "
                "JUDGE_API_BASE, and JUDGE_MODEL_NAME in evaluation/.env."
            )
        }

    # --- paper fidelity: one call per checklist item (report + images) ---
    fidelity_agent = _make_agent(max_tokens=500)

    def score_item(index, item_data):
        item_type = item_data.get("type", "text")
        target_path = None
        if item_type == "image":
            target_rel = item_data.get("path", "")
            target_base = TASKS_DIR / task_id / "target_study"
            target_path = safe_resolve(target_base, target_rel)
        return _score_item_fidelity(fidelity_agent, report_text, item_data, target_path,
                                    generated_images, instructions)

    inputs = [{"index": i, "item_data": item} for i, item in enumerate(checklist)]
    fidelity_results = multi_thread(inputs, score_item,
                                    max_workers=min(len(checklist), 16), use_tqdm=False)

    # --- scientific capability: one holistic call over static artifacts ---
    artifacts = _gather_research_artifacts(workspace)
    research_agent = _make_agent(max_tokens=2200)
    research_result = _score_research(research_agent, report_text, instructions,
                                      checklist, artifacts)

    agg = aggregate_scores(checklist, fidelity_results, research_result)
    score_data = {
        "run_id": run_id,
        "task_id": task_id,
        "agent_name": agent_name,
        **agg,
    }

    score_path = workspace / "_score.json"
    with open(score_path, "w", encoding="utf-8") as f:
        json.dump(score_data, f, indent=2)

    return score_data


def score_run(run_id: str) -> dict:
    """Score a completed run by resolving its workspace from the run ID."""
    workspace = get_run_workspace(run_id)
    if not workspace:
        return {"error": "Workspace not found"}
    return score_workspace(workspace)
