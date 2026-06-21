"""Scorer: evaluate a research report against a checklist using structai.

Each checklist item is scored on TWO independent axes (0-100 each):

  * scientific_score  — Did the agent perform valid, evidence-backed research for
                        this criterion using the provided data/instructions,
                        regardless of whether it matches the target paper?
  * fidelity_score    — Did the agent recover the target paper's SPECIFIC result
                        (values, trends, figure, mechanism, conclusion)?

Aggregated per run into:
  scientific_capability_score = weighted avg of item scientific_score
  paper_fidelity_score        = weighted avg of item fidelity_score
  total_score                 = SCIENTIFIC_WEIGHT * scientific_capability_score
                              + FIDELITY_WEIGHT   * paper_fidelity_score

The target paper is a reference instrument, not the sole definition of success:
the benchmark measures automated-research capability first, reproduction second.
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

RUBRIC = """You are a strict scientific peer reviewer for an automated-research benchmark. An AI agent was given a task (data + instructions) derived from a target paper, and produced a research report. For ONE evaluation criterion (itself derived from the target paper), score the report on TWO INDEPENDENT axes.

The target paper is a REFERENCE, not the sole definition of success: measure scientific research capability first, paper reproduction second.

Both axes are anchored on the target paper: 50 = ON PAR WITH THE PAPER, above 50 = better than the paper on that axis, below 50 = worse. A report whose research quality AND reproduced results both equal the paper should land near 50 on both axes.

=== AXIS 1 — SCIENTIFIC CAPABILITY (0-100; 50 = on par with the target paper) ===
Did the agent perform valid, evidence-backed research for this criterion using the provided data and instructions? Judge the research PROCESS and its rigor relative to the target paper's, regardless of whether the specific numbers match.
Reward: appropriate use of the provided data; sound method selection; analysis backed by the agent's OWN reproducible evidence (code, outputs, figures); correct quantitative reasoning; a clear connection between results and claims; honest handling of uncertainty, scope, and limitations; and genuine scientific progress, including work that goes beyond the paper.
Penalize: vague claims without evidence; analysis not grounded in the provided data; inappropriate method; fabricated, unsupported, or untraceable numbers; conclusions that do not follow from the evidence; and reports that merely paraphrase a paper-like conclusion without doing the research work.
Scale (anchored on the target paper):
- 0: criterion absent or essentially unaddressed.
- 1-20: mentioned only; no real method or evidence.
- 21-40: some analysis but major gaps, clearly weaker than the paper, OR evidence that is unsupported/fabricated. USE THIS BAND for fabricated or untraceable quantitative claims, no matter how polished.
- 41-50: research quality (validity, evidence, rigor, reasoning) roughly EQUIVALENT to the target paper; 50 = on par with the paper.
- 51-70: more rigorous, better-grounded, or more complete research than the paper.
- 71-90: significantly stronger science, with validation or insight clearly beyond the paper.
- 91-100: exceptional research far beyond the paper.

=== AXIS 2 — PAPER FIDELITY (0-100; 50 = reproduces the paper's reported result) ===
Did the agent recover the target paper's SPECIFIC result for this criterion — its claimed values, trends, figure, mechanism, or conclusion?
Reward: matching the paper's specific claims, values, and trends; recovering the same mechanism/interpretation; for image criteria, matching the target figure in variables, trend, scale, labels, and content; and reproductions that are even more complete or quantitatively stronger than the paper reported, in the SAME direction / supporting the SAME claim.
Penalize: missing the paper's specific values/trends/figures/mechanisms; reaching a DIFFERENT conclusion than the paper; figures visually or quantitatively inconsistent with the target; absent required artifacts.
Scale (anchored on the target paper):
- 0: the paper's specific result is absent from the report.
- 1-20: mentioned but not actually reproduced.
- 21-40: partial reproduction with notable deviations.
- 41-50: reproduces the paper's specific result, matching what the paper reported; 50 = on par with the paper.
- 51-70: reproduces AND improves on the paper's reported numbers/trends in the same direction (stronger or more complete than the paper).
- 71-90: substantially exceeds the paper's reported results while remaining consistent with its claims.
- 91-100: dramatically exceeds the paper's reported results.
A well-justified but DIFFERENT conclusion scores LOW here (it did not reproduce the paper) — that is expected, and is rewarded on Axis 1 instead.

CRITICAL: The two axes are independent — a report can be strong on one and weak on the other. 50 means "as good as the published paper" on each axis — a high bar. Be skeptical of fluent but unsupported text. Substance over style; longer is not better; honest acknowledgement of limitations is a positive."""

_RETURN_EXAMPLE = {
    "scientific_reasoning": "str",
    "scientific_score": 0,
    "fidelity_reasoning": "str",
    "fidelity_score": 0,
}


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


_TASK_INSTRUCTIONS = """## Task
Score this report against the criterion on BOTH axes, applying the rubric strictly.
First judge Scientific Capability (valid evidence-backed research for this criterion,
match-or-not), then Paper Fidelity (recovery of the paper's specific result).

Return a JSON object:
{{"scientific_reasoning": "<2-3 sentences>", "scientific_score": <0-100>, "fidelity_reasoning": "<2-3 sentences>", "fidelity_score": <0-100>}}"""


def _build_text_prompt(report_text: str, item: dict, instructions: str) -> str:
    criteria = item.get("content", "")
    keywords = item.get("keywords", [])
    keywords_str = ", ".join(keywords) if keywords else "None specified"
    return f"""{RUBRIC}

## Research Task Background (INSTRUCTIONS.md given to the AI agent)
{instructions}

## Evaluation Criterion (derived from the target paper)
{criteria}

## Key Technical Aspects to Verify
{keywords_str}

## AI-Generated Research Report
{report_text}

{_TASK_INSTRUCTIONS}"""


def _build_image_prompt(report_text: str, item: dict, instructions: str) -> str:
    criteria = item.get("content", "")
    keywords = item.get("keywords", [])
    keywords_str = ", ".join(keywords) if keywords else "None specified"
    return f"""{RUBRIC}

## Research Task Background (INSTRUCTIONS.md given to the AI agent)
{instructions}

## Evaluation Criterion (derived from the target paper)
{criteria}

## Key Visual/Technical Aspects to Verify
{keywords_str}

## AI-Generated Report Text (excerpt)
{report_text[:10000] if report_text else 'No report text available.'}

## Images
When images are attached, the FIRST image is the ground-truth target figure from the
target paper; all subsequent images are from the AI agent's workspace/report.
For image criteria apply the two axes as follows:
- Scientific score: does the agent's figure provide valid, well-constructed evidence
  for the scientific claim (correct variables, sound method, readable, honest)?
- Fidelity score: does the agent's figure match the TARGET figure in variables, trend,
  scale, labels, and visual content? Superficially similar plots with wrong scales,
  missing data, or incorrect trends get a LOW fidelity score.

{_TASK_INSTRUCTIONS}"""


def _clamp(v) -> int:
    try:
        return max(0, min(100, int(v)))
    except (TypeError, ValueError):
        return 0


def _score_single_item(agent: LLMAgent, report_text: str, item: dict,
                       target_image_path: Optional[Path],
                       generated_images: list[Path],
                       instructions: str) -> dict:
    """Score a single checklist item on both axes (text or image)."""
    item_type = item.get("type", "text")

    if item_type == "image":
        prompt = _build_image_prompt(report_text, item, instructions)
        img_paths = []
        if target_image_path and target_image_path.exists():
            img_paths.append(str(target_image_path))
        for img in generated_images[:5]:
            if img.exists() and img.stat().st_size <= MAX_IMAGE_SIZE:
                img_paths.append(str(img))
        result = agent(prompt, image_paths=img_paths if img_paths else None,
                       return_example=_RETURN_EXAMPLE, max_try=2)
    else:
        prompt = _build_text_prompt(report_text, item, instructions)
        result = agent(prompt, return_example=_RETURN_EXAMPLE, max_try=2)

    if result and isinstance(result, dict):
        return {
            "scientific_score": _clamp(result.get("scientific_score", 0)),
            "scientific_reasoning": str(result.get("scientific_reasoning", "")),
            "fidelity_score": _clamp(result.get("fidelity_score", 0)),
            "fidelity_reasoning": str(result.get("fidelity_reasoning", "")),
        }
    return {
        "scientific_score": 0,
        "scientific_reasoning": "Failed to parse scoring response.",
        "fidelity_score": 0,
        "fidelity_reasoning": "Failed to parse scoring response.",
    }


def _combined(scientific: float, fidelity: float) -> float:
    return SCIENTIFIC_WEIGHT * scientific + FIDELITY_WEIGHT * fidelity


def aggregate_scores(checklist: list[dict], raw_results: list[dict]) -> dict:
    """Pure aggregation of per-item dual scores into run-level scores.

    Returns a dict with items + scientific_capability_score, paper_fidelity_score,
    total_score, total_weight. Kept side-effect-free so it is unit-testable.
    """
    items = []
    tot_sci = tot_fid = tot_w = 0.0
    for i, (item, sr) in enumerate(zip(checklist, raw_results)):
        weight = float(item.get("weight", 1.0))
        sr = sr or {}
        sci = _clamp(sr.get("scientific_score", 0))
        fid = _clamp(sr.get("fidelity_score", 0))
        sci_reason = str(sr.get("scientific_reasoning", ""))
        fid_reason = str(sr.get("fidelity_reasoning", ""))
        combined = round(_combined(sci, fid))
        items.append({
            "index": i,
            "type": item.get("type", "text"),
            "content": item.get("content", "")[:200],
            "weight": weight,
            "scientific_score": sci,
            "scientific_reasoning": sci_reason,
            "fidelity_score": fid,
            "fidelity_reasoning": fid_reason,
            # legacy fields (backward compatibility with older consumers/UI)
            "score": combined,
            "reasoning": f"[Scientific {sci}] {sci_reason}  [Fidelity {fid}] {fid_reason}".strip(),
        })
        tot_sci += sci * weight
        tot_fid += fid * weight
        tot_w += weight

    sci_agg = (tot_sci / tot_w) if tot_w > 0 else 0.0
    fid_agg = (tot_fid / tot_w) if tot_w > 0 else 0.0
    total = _combined(sci_agg, fid_agg)
    return {
        "items": items,
        "scientific_capability_score": round(sci_agg, 2),
        "paper_fidelity_score": round(fid_agg, 2),
        "total_score": round(total, 2),
        "total_weight": tot_w,
        "scoring_weights": {"scientific": SCIENTIFIC_WEIGHT, "fidelity": FIDELITY_WEIGHT},
    }


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
    agent = LLMAgent(
        api_key=judge_api_key,
        api_base=judge_api_base,
        model_version=JUDGE_MODEL_NAME,
        system_prompt=(
            "You are a strict scientific peer reviewer for an automated-research "
            "benchmark. Score the report against the criterion on two independent "
            "axes (scientific capability and paper fidelity); do not attempt to "
            "solve the research task yourself."
        ),
        temperature=0,
        max_tokens=900,
        time_limit=120,
        max_try=2,
    )

    def score_item(index, item_data):
        item_type = item_data.get("type", "text")
        target_path = None
        if item_type == "image":
            target_rel = item_data.get("path", "")
            target_base = TASKS_DIR / task_id / "target_study"
            target_path = safe_resolve(target_base, target_rel)
        return _score_single_item(agent, report_text, item_data, target_path,
                                  generated_images, instructions)

    inputs = [{"index": i, "item_data": item} for i, item in enumerate(checklist)]
    raw_results = multi_thread(inputs, score_item,
                               max_workers=min(len(checklist), 16), use_tqdm=False)

    agg = aggregate_scores(checklist, raw_results)
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
