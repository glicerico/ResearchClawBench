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

import base64
import json
import os
import re
import statistics
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

# --- judge output-token budgets ----------------------------------------------
# These must be generous enough for *reasoning* judge models (e.g. gpt-5.5,
# gemini-3.1-pro): their internal reasoning tokens count against max_tokens, so a
# tight cap truncates the JSON before the answer is emitted (finish_reason
# "length") and parsing falls back to 0. The visible JSON is small; the headroom
# is for the model's thinking.
FIDELITY_MAX_TOKENS = 20000   # per checklist item: a one-line JSON verdict
RESEARCH_MAX_TOKENS = 20000   # one holistic call: nested JSON over all dimensions
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
    d["key"]: {"score": 0, "reasoning": "str", "gap": "str"} for d in RESEARCH_DIMENSIONS
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

Return a JSON object keyed by dimension. For each dimension provide:
- "score": 0-100.
- "reasoning": 2-3 sentences justifying the score from the evidence.
- "gap": 1-2 sentences naming SPECIFICALLY and ACTIONABLY what was missing or needed to earn a higher score on this dimension (e.g. "no ablation isolating the decoupled encoder", "claims X but no output file shows it"). If the dimension already clearly exceeds the paper, say what would push it even higher.

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
                lines.append(f"LOG: {raw[:MAX_TOOL_RESULT_CHARS]}")
                continue
            if not isinstance(d, dict):
                lines.append(f"LOG: {str(d)[:MAX_TOOL_RESULT_CHARS]}")
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
            elif "action" in d and "result" in d:
                lines.append(f"ACTION {d['action']}:")
                res = str(d["result"]).strip().replace("\n", " ")
                if res:
                    lines.append(f"RESULT: {res[:MAX_TOOL_RESULT_CHARS]}")
            else:
                # Completely generic fallback for unknown JSON schemas
                raw_str = json.dumps(d, default=str)
                lines.append(f"LOG: {raw_str[:MAX_TOOL_RESULT_CHARS]}")
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
            gap = entry.get("gap", "")
        else:
            # tolerate flat shape: {"<key>_score": .., "<key>_reasoning": .., "<key>_gap": ..}
            score = result.get(f"{key}_score", 0)
            reasoning = result.get(f"{key}_reasoning", "")
            gap = result.get(f"{key}_gap", "")
        out[key] = {"score": _clamp(score), "reasoning": str(reasoning), "gap": str(gap)}
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
        gap = str(dr.get("gap", ""))
        w = float(d.get("weight", 1.0))
        dimensions.append({
            "key": d["key"],
            "name": d["name"],
            "weight": w,
            "score": sc,
            "reasoning": reasoning,
            "gap": gap,
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


# --- structai image media-type correction -----------------------------------
# structai's LLMAgent encodes every image as PNG bytes (encode_image saves with
# format="PNG") but hardcodes a `data:image/jpeg;base64,` URL prefix. Strict
# providers (Anthropic / Google / Bedrock via OpenRouter, etc.) reject the
# mismatch with HTTP 400. We correct the declared media type from the actual
# image bytes at the OpenAI request boundary of the judge agent, so the judge
# keeps the lossless PNG and any provider accepts it.
# Upstream root cause: https://github.com/black-yt/structai (llm_api.py image_url builder).

_DATA_URI_RE = re.compile(r"^data:([^;,]+);base64,(.*)$", re.DOTALL)


def _detect_image_mime(raw: bytes) -> Optional[str]:
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if raw.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if raw.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    if raw.startswith(b"BM"):
        return "image/bmp"
    return None


def _corrected_image_data_uri(url: str) -> str:
    if not isinstance(url, str) or not url.startswith("data:"):
        return url
    match = _DATA_URI_RE.match(url)
    if not match:
        return url
    declared, b64 = match.group(1), match.group(2)
    head = b64[:88]  # multiple of 4 -> ~66 decoded bytes, enough for any signature
    if len(head) % 4:
        head += "=" * (4 - len(head) % 4)
    try:
        raw = base64.b64decode(head)
    except Exception:
        return url
    actual = _detect_image_mime(raw)
    if actual and actual != declared:
        return f"data:{actual};base64,{b64}"
    return url


def _fix_message_image_media_types(messages: list) -> list:
    for msg in messages:
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "image_url":
                image_url = block.get("image_url")
                if isinstance(image_url, dict) and "url" in image_url:
                    image_url["url"] = _corrected_image_data_uri(image_url["url"])
    return messages


def _install_image_media_type_fix(agent: LLMAgent) -> None:
    """Wrap the agent's OpenAI client so image data URLs declare their true media type."""
    completions = agent.client.chat.completions
    original_create = completions.create

    def patched_create(*args, **kwargs):
        messages = kwargs.get("messages")
        if isinstance(messages, list):
            kwargs["messages"] = _fix_message_image_media_types(messages)
        return original_create(*args, **kwargs)

    completions.create = patched_create


def _make_agent(max_tokens: int, *, model: str = "", api_base: str = "", api_key: str = "") -> LLMAgent:
    agent = LLMAgent(
        api_key=api_key or os.environ.get("JUDGE_API_KEY", ""),
        api_base=api_base or os.environ.get("JUDGE_API_BASE", ""),
        model_version=model or JUDGE_MODEL_NAME,
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
    _install_image_media_type_fix(agent)
    return agent


def score_workspace(
    workspace: str | Path,
    *,
    model: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    out_name: str = "_score.json",
) -> dict:
    """Score a completed run workspace against its task's checklist.

    The judge model/credentials can be passed explicitly (per-judge, re-entrant) or
    left as ``None`` to fall back to the JUDGE_MODEL_NAME global and JUDGE_API_KEY /
    JUDGE_API_BASE environment variables (single-judge / legacy callers). ``out_name``
    is the filename written under the workspace, so concurrent judges don't collide.
    """
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

    judge_api_key = api_key if api_key is not None else os.environ.get("JUDGE_API_KEY", "")
    judge_api_base = api_base if api_base is not None else os.environ.get("JUDGE_API_BASE", "")
    judge_model = model if model is not None else JUDGE_MODEL_NAME
    if not judge_api_key or not judge_api_base or not judge_model:
        return {
            "error": (
                "Judge API configuration is missing. Set JUDGE_API_KEY, "
                "JUDGE_API_BASE, and JUDGE_MODEL_NAME in evaluation/.env."
            )
        }

    # --- paper fidelity: one call per checklist item (report + images) ---
    fidelity_agent = _make_agent(
        max_tokens=FIDELITY_MAX_TOKENS, model=judge_model,
        api_base=judge_api_base, api_key=judge_api_key
    )

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
    research_agent = _make_agent(
        max_tokens=RESEARCH_MAX_TOKENS, model=judge_model,
        api_base=judge_api_base, api_key=judge_api_key
    )
    research_result = _score_research(research_agent, report_text, instructions,
                                      checklist, artifacts)

    agg = aggregate_scores(checklist, fidelity_results, research_result)
    score_data = {
        "run_id": run_id,
        "task_id": task_id,
        "agent_name": agent_name,
        "judge_model": judge_model,
        **agg,
    }

    score_path = workspace / out_name
    with open(score_path, "w", encoding="utf-8") as f:
        json.dump(score_data, f, indent=2)

    return score_data


# --- multi-judge ensemble -----------------------------------------------------
#
# Scoring one fixed research run with several judge models (and aggregating per
# axis) measures *judge* agreement/disagreement on the same artifact. This is the
# default for the dashboard "Score" button; the CLI reaches the same aggregation
# through these helpers. It is orthogonal to ``repeats`` (which re-runs research).

DEFAULT_JUDGE_ENSEMBLE_BASE = "https://openrouter.ai/api/v1"


def _judge_slug(model: str, fallback: str = "judge") -> str:
    """Filesystem-safe slug for a judge model name (for per-judge score files)."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", model or "").strip("-").lower()
    return slug or fallback


def agg_stat(values: list[float]) -> tuple[Optional[float], Optional[float]]:
    """Mean and population std of the per-judge values for one axis."""
    if not values:
        return None, None
    mean = round(sum(values) / len(values), 2)
    std = round(statistics.pstdev(values), 2) if len(values) > 1 else 0.0
    return mean, std


def aggregate_judges(per_judge: list[dict]) -> dict:
    """Combine N per-judge score dicts into one aggregate (mean + per-axis std).

    Headline scores are the mean across judges; each axis also carries its own
    standard deviation (inter-judge spread / disagreement). The first judge's
    per-item/per-dimension detail is preserved so the run-detail UI still renders.
    """
    def axis(key: str) -> list[float]:
        return [float(d[key]) for d in per_judge if d.get(key) is not None]

    total_mean, total_std = agg_stat(axis("total_score"))
    sci_mean, sci_std = agg_stat(axis("scientific_capability_score"))
    fid_mean, fid_std = agg_stat(axis("paper_fidelity_score"))

    aggregate = dict(per_judge[0])  # keep run_id/task_id/items/research_dimensions
    aggregate.pop("judge_model", None)  # singular doesn't apply to an ensemble
    aggregate.update({
        "judges": len(per_judge),
        "judge_models": [d.get("judge_model") for d in per_judge],
        "aggregation": "mean_across_judges",
        "total_score": total_mean,
        "scientific_capability_score": sci_mean,
        "paper_fidelity_score": fid_mean,
        "total_score_std": total_std,
        "scientific_capability_score_std": sci_std,
        "paper_fidelity_score_std": fid_std,
        "per_judge": [
            {
                "judge_model": d.get("judge_model"),
                "total_score": d.get("total_score"),
                "scientific_capability_score": d.get("scientific_capability_score"),
                "paper_fidelity_score": d.get("paper_fidelity_score"),
            }
            for d in per_judge
        ],
    })
    return aggregate


def default_judge_ensemble() -> list[dict]:
    """Resolve the default judge ensemble for the dashboard from the environment.

    Returns a list of ``{model, api_base, api_key}`` dicts. When ``JUDGE_MODELS``
    (comma-separated slugs) is set and a shared key is available, those judges run
    over a shared OpenRouter-style base. Otherwise this falls back to the single
    legacy judge (``JUDGE_MODEL_NAME`` + ``JUDGE_API_BASE``/``JUDGE_API_KEY``), so
    existing single-judge setups keep working unchanged.
    """
    models_raw = os.environ.get("JUDGE_MODELS", "").strip()
    if models_raw:
        models = [m.strip() for m in models_raw.split(",") if m.strip()]
        base = (
            os.environ.get("JUDGE_ENSEMBLE_API_BASE")
            or os.environ.get("OPENROUTER_API_BASE")
            or DEFAULT_JUDGE_ENSEMBLE_BASE
        )
        key = (
            os.environ.get("JUDGE_ENSEMBLE_API_KEY")
            or os.environ.get("OPENROUTER_API_KEY", "")
        )
        if models and base and key:
            return [{"model": m, "api_base": base, "api_key": key} for m in models]
    return [{
        "model": JUDGE_MODEL_NAME,
        "api_base": os.environ.get("JUDGE_API_BASE", ""),
        "api_key": os.environ.get("JUDGE_API_KEY", ""),
    }]


def score_workspace_ensemble(workspace: str | Path, judges: Optional[list[dict]] = None) -> dict:
    """Score one workspace with a set of judges and write an aggregate ``_score.json``.

    ``judges`` is a list of ``{model, api_base, api_key}`` dicts; when omitted the
    default ensemble (see :func:`default_judge_ensemble`) is used. A single judge
    writes ``_score.json`` directly (legacy shape). Multiple judges each write a
    ``_score_<slug>.json`` and the mean+std aggregate is written to ``_score.json``.
    """
    workspace = Path(workspace)
    if judges is None:
        judges = default_judge_ensemble()
    enabled = [j for j in judges if j.get("model") and j.get("api_base") and j.get("api_key")]
    if not enabled:
        return {
            "error": (
                "Judge API configuration is missing. Set JUDGE_MODELS + "
                "OPENROUTER_API_KEY (ensemble) or JUDGE_MODEL_NAME + JUDGE_API_KEY "
                "+ JUDGE_API_BASE (single judge) in evaluation/.env."
            )
        }

    if len(enabled) == 1:
        j = enabled[0]
        return score_workspace(
            workspace, model=j["model"], api_base=j["api_base"], api_key=j["api_key"]
        )

    per_judge: list[dict] = []
    errors: list[str] = []
    used_slugs: dict[str, int] = {}
    for j in enabled:
        slug = _judge_slug(j["model"])
        used_slugs[slug] = used_slugs.get(slug, 0) + 1
        if used_slugs[slug] > 1:
            slug = f"{slug}-{used_slugs[slug]}"
        try:
            data = score_workspace(
                workspace, model=j["model"], api_base=j["api_base"], api_key=j["api_key"],
                out_name=f"_score_{slug}.json",
            )
        except Exception as exc:  # noqa: BLE001 - record and continue with other judges
            errors.append(f"{j['model']}: {type(exc).__name__}: {exc}")
            continue
        if not isinstance(data, dict) or data.get("error"):
            detail = data.get("error") if isinstance(data, dict) else "non-dict result"
            errors.append(f"{j['model']}: {detail}")
        else:
            per_judge.append(data)

    if not per_judge:
        return {"error": "; ".join(errors) or "All judges failed."}

    aggregate = aggregate_judges(per_judge)
    if errors:
        aggregate["judge_errors"] = errors
    with open(workspace / "_score.json", "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)
    return aggregate


def score_run(run_id: str) -> dict:
    """Score a completed run by resolving its workspace from the run ID.

    Uses the default judge ensemble (multiple judges when configured), so the
    dashboard "Score" button produces per-axis means + inter-judge spread.
    """
    workspace = get_run_workspace(run_id)
    if not workspace:
        return {"error": "Workspace not found"}
    return score_workspace_ensemble(workspace)
