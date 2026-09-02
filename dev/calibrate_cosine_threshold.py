"""Calibrate Agent 2's cosine threshold against the current V6 fixtures.

This script is deliberately read-only: it loads the two retained fixtures,
embeds the reviewed non-noise requirements, sweeps thresholds from 0.50 to
0.99, and prints the best conservative threshold.  Labels live here so the
calibration needs no second fixture or generated report file.

Run:
    python -m dev.calibrate_cosine_threshold
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from config import EMBEDDING_MODEL, get_embeddings
from core.cosine_matcher import (
    _literal_skill_match,
    _semantic_skill_pair_allowed,
    cosine_similarity,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CV_FIXTURE = PROJECT_ROOT / "fixtures" / "cv_parser_fixture.json"
JOB_FIXTURE = PROJECT_ROOT / "fixtures" / "linkedin_job_parser_fixture.json"
PRECISION_TARGET = 0.90


@dataclass(frozen=True)
class Case:
    job: str
    skill: str
    should_match: bool


@dataclass(frozen=True)
class Result:
    case: Case
    candidate_skill: str
    similarity: float
    literal: bool = False


def _cases(job: str, positives: list[str], negatives: list[str]) -> list[Case]:
    return [Case(job, skill, True) for skill in positives] + [
        Case(job, skill, False) for skill in negatives
    ]


# Only clear technical match/no-match decisions are included. Business terms,
# parser noise, and genuinely debatable requirements are intentionally left out
# so parser mistakes do not determine the embedding threshold.
CASES = [
    *_cases(
        "Senior AI/ML Engineer",
        positives=[
            "large language model (LLM) applications",
            "OpenAI",
            "Anthropic/Claude",
            "Gemini",
            "LLaMA",
            "Mistral",
            "chunking strategies",
            "vector database integration",
            "agentic AI systems",
            "prompt hardening",
            "responsible AI guardrails",
            "testing",
            "monitoring",
            "observability",
        ],
        negatives=[
            "transformer-based architectures",
            "document ingestion",
            "embedding generation",
            "CI/CD",
        ],
    ),
    *_cases(
        "Senior Machine Learning Engineer",
        positives=[
            "Python",
            "evaluation systems",
            "modern LLM tooling",
            "reason across multi-step workflows",
            "use tools",
            "retrieve context",
            "measure model quality",
            "safety",
        ],
        negatives=[
            "PyTorch",
            "ML data pipelines",
            "training workflows",
            "model serving",
            "inference frameworks",
            "machine learning fundamentals",
            "model performance tradeoffs",
            "fine-tune and adapt large language models",
            "supervised fine-tuning",
            "post-training techniques",
            "synthetic data pipelines",
            "operate reliably in production",
        ],
    ),
    *_cases(
        "Senior Engineer - Machine Learning",
        positives=[
            "Python",
            "LLMs",
            "retrieval-augmented generation",
            "prompt engineering",
            "AI agents",
            "tool use",
            "workflow automation",
            "AI evaluation harnesses",
            "monitoring",
            "observability",
            "agentic systems",
            "information retrieval",
            "generative AI platform",
        ],
        negatives=[
            "regression",
            "classification",
            "clustering",
            "memory/context management",
            "cloud infrastructure",
            "MLOps practices",
            "semi-structured data",
            "machine learning",
            "Agile methodologies",
            "CI/CD practices",
            "intelligent document processing",
            "notebooks",
            "model deployment",
            "system designs",
        ],
    ),
]

# False matches observed in the 25 August live Agent 2 run. These are
# pair-level regressions rather than fixture requirements: each one records the
# exact candidate skill that cosine incorrectly chose. Protected pairs receive
# similarity -1.0 so they remain visible without forcing the global threshold
# upward and creating unrelated false negatives.
LIVE_FALSE_PAIR_CASES = (
    ("LlamaIndex", "Llama"),
    ("RAG systems", "RAGAS"),
    ("LLM APIs", "LLM evaluation"),
    ("LLM pipeline", "LLM evaluation"),
    ("AWS ECS", "AWS Bedrock"),
    ("DeepAgent", "DeepEval"),
    ("version control", "prompt versioning"),
)


def _load_inputs() -> tuple[
    list[str],
    dict[str, str],
    dict[str, dict[str, str]],
    dict[str, set[str]],
]:
    cv_data = json.loads(CV_FIXTURE.read_text(encoding="utf-8"))
    job_data = json.loads(JOB_FIXTURE.read_text(encoding="utf-8"))
    candidate_skills = list(cv_data["parsed_cv"]["skills"])
    candidate_evidence = dict(cv_data["parsed_cv"].get("skill_evidence", {}))
    available = {
        entry["linkedin_job"]["title"]: set(
            entry["parsed_requirements"]["required_skills"]
        )
        for entry in job_data["jobs"]
        if entry.get("parsed_requirements")
    }
    job_evidence = {
        entry["linkedin_job"]["title"]: dict(
            entry["parsed_requirements"].get("required_skill_evidence", {})
        )
        for entry in job_data["jobs"]
        if entry.get("parsed_requirements")
    }
    missing_labels = [
        f"{case.job}: {case.skill}"
        for case in CASES
        if case.job not in available or case.skill not in available[case.job]
    ]
    if missing_labels:
        raise ValueError(
            "Calibration labels no longer match the current job fixture:\n- "
            + "\n- ".join(missing_labels)
        )
    return candidate_skills, candidate_evidence, job_evidence, available


def _embedding_text(skill: str, evidence: str, include_evidence: bool) -> str:
    if not include_evidence or not evidence:
        return skill
    return f"Technical skill: {skill}. Evidence: {evidence}"


def _evaluate(
    candidate_skills: list[str],
    candidate_evidence: dict[str, str],
    job_evidence: dict[str, dict[str, str]],
    embeddings,
    *,
    include_evidence: bool,
) -> tuple[list[Result], list[Result]]:
    literal_results: list[Result] = []
    semantic_cases: list[Case] = []
    for case in CASES:
        literal_hit = next(
            (
                candidate
                for candidate in candidate_skills
                if _literal_skill_match(case.skill, candidate)
            ),
            None,
        )
        if literal_hit is not None:
            literal_results.append(Result(case, literal_hit, 1.0, literal=True))
        else:
            semantic_cases.append(case)

    unique_job_skills = list(dict.fromkeys(case.skill for case in semantic_cases))
    candidate_texts = [
        _embedding_text(
            skill,
            candidate_evidence.get(skill, ""),
            include_evidence,
        )
        for skill in candidate_skills
    ]
    # The same job skill can occur in multiple jobs with different evidence,
    # so evidence mode embeds cases individually rather than deduplicating by
    # skill name. Raw mode retains the smaller unique list.
    if include_evidence:
        job_texts = [
            _embedding_text(
                case.skill,
                job_evidence.get(case.job, {}).get(case.skill, ""),
                True,
            )
            for case in semantic_cases
        ]
    else:
        job_texts = unique_job_skills
    vectors = embeddings.embed_documents(candidate_texts + job_texts)
    candidate_vectors = vectors[: len(candidate_skills)]
    if include_evidence:
        case_vectors = vectors[len(candidate_skills) :]
    else:
        job_vectors = dict(
            zip(unique_job_skills, vectors[len(candidate_skills) :], strict=True)
        )
        case_vectors = [job_vectors[case.skill] for case in semantic_cases]

    semantic_results: list[Result] = []
    for case, job_vector in zip(semantic_cases, case_vectors, strict=True):
        similarities = [
            (
                cosine_similarity(job_vector, candidate_vector)
                if _semantic_skill_pair_allowed(case.skill, candidate_skill)
                else -1.0
            )
            for candidate_skill, candidate_vector in zip(
                candidate_skills, candidate_vectors
            )
        ]
        best_index = max(range(len(similarities)), key=similarities.__getitem__)
        semantic_results.append(
            Result(
                case=case,
                candidate_skill=candidate_skills[best_index],
                similarity=round(similarities[best_index], 6),
            )
        )
    return literal_results, semantic_results


def _evaluate_live_false_pairs(embeddings) -> list[Result]:
    """Evaluate exact live false pairs through the production pair gate."""

    texts = list(
        dict.fromkeys(
            text
            for job_skill, candidate_skill in LIVE_FALSE_PAIR_CASES
            for text in (job_skill, candidate_skill)
        )
    )
    vectors = embeddings.embed_documents(texts)
    by_text = dict(zip(texts, vectors, strict=True))
    results: list[Result] = []
    for job_skill, candidate_skill in LIVE_FALSE_PAIR_CASES:
        allowed = _semantic_skill_pair_allowed(job_skill, candidate_skill)
        similarity = (
            cosine_similarity(by_text[job_skill], by_text[candidate_skill])
            if allowed
            else -1.0
        )
        results.append(
            Result(
                case=Case("Live false-pair regressions", job_skill, False),
                candidate_skill=candidate_skill,
                similarity=round(similarity, 6),
            )
        )
    return results


def _metrics(results: list[Result], threshold: float) -> dict[str, float | int]:
    tp = sum(r.case.should_match and r.similarity >= threshold for r in results)
    fp = sum(not r.case.should_match and r.similarity >= threshold for r in results)
    fn = sum(r.case.should_match and r.similarity < threshold for r in results)
    tn = sum(not r.case.should_match and r.similarity < threshold for r in results)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _select_threshold(results: list[Result]) -> tuple[float, dict[str, float | int]]:
    # Domain-tuned embedding models can place both positive and negative
    # skill pairs above 0.90, so keep the upper tail in the calibration sweep.
    sweep = [round(value / 100, 2) for value in range(50, 100)]
    scored = [(threshold, _metrics(results, threshold)) for threshold in sweep]
    eligible = [item for item in scored if item[1]["precision"] >= PRECISION_TARGET]
    pool = eligible or scored
    best_key = max(
        (metrics["recall"], metrics["f1"], metrics["precision"])
        for _, metrics in pool
    )
    plateau = [
        (threshold, metrics)
        for threshold, metrics in pool
        if (metrics["recall"], metrics["f1"], metrics["precision"]) == best_key
    ]
    return plateau[len(plateau) // 2]


def _percent(value: float | int) -> str:
    return f"{float(value) * 100:5.1f}%"


def main() -> int:
    candidate_skills, candidate_evidence, job_evidence, available = _load_inputs()
    embeddings = get_embeddings()
    literal, semantic = _evaluate(
        candidate_skills,
        candidate_evidence,
        job_evidence,
        embeddings,
        include_evidence=False,
    )
    live_false_pairs = _evaluate_live_false_pairs(embeddings)
    semantic.extend(live_false_pairs)
    selected, selected_metrics = _select_threshold(semantic)

    print("AGENT 2 COSINE THRESHOLD CALIBRATION")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Candidate skills: {len(candidate_skills)}")
    print(f"Reviewed requirements: {len(CASES)}")
    print(f"Literal decisions: {len(literal)}")
    print(f"Semantic decisions used for threshold: {len(semantic)}")
    blocked_live_pairs = sum(result.similarity < 0 for result in live_false_pairs)
    print(
        f"Live false-pair regressions: {len(live_false_pairs)} "
        f"({blocked_live_pairs} blocked before threshold)"
    )
    ignored = sum(len(skills) for skills in available.values()) - len(CASES)
    print(f"Ambiguous/parser-noise requirements ignored: {ignored}")

    literal_errors = [result for result in literal if not result.case.should_match]
    print(f"\nLiteral-pass false positives (threshold cannot fix): {len(literal_errors)}")
    for result in literal_errors:
        print(f"  - {result.case.skill} <- {result.candidate_skill}")

    print("\nThreshold sweep")
    print("threshold  precision  recall     F1       FP  FN")
    displayed = set(range(50, 100, 5)) | {99, round(selected * 100)}
    for value in sorted(displayed):
        threshold = value / 100
        metrics = _metrics(semantic, threshold)
        marker = "  <-- selected" if threshold == selected else ""
        print(
            f"   {threshold:.2f}     {_percent(metrics['precision'])}   "
            f"{_percent(metrics['recall'])}  {_percent(metrics['f1'])}  "
            f"{metrics['fp']:3} {metrics['fn']:3}{marker}"
        )

    print(
        f"\nSelected threshold: {selected:.2f} "
        f"(precision={_percent(selected_metrics['precision']).strip()}, "
        f"recall={_percent(selected_metrics['recall']).strip()}, "
        f"F1={_percent(selected_metrics['f1']).strip()})"
    )

    false_positives = [
        result
        for result in semantic
        if not result.case.should_match and result.similarity >= selected
    ]
    false_negatives = [
        result
        for result in semantic
        if result.case.should_match and result.similarity < selected
    ]
    print("\nFalse positives at selected threshold:")
    if not false_positives:
        print("  None")
    for result in sorted(false_positives, key=lambda item: -item.similarity):
        print(
            f"  - {result.case.skill} <- {result.candidate_skill} "
            f"({result.similarity:.3f})"
        )

    print("\nFalse negatives at selected threshold:")
    if not false_negatives:
        print("  None")
    for result in sorted(false_negatives, key=lambda item: -item.similarity):
        print(
            f"  - {result.case.skill} <- {result.candidate_skill} "
            f"({result.similarity:.3f})"
        )

    print("\nLeave-one-job-out check")
    for job in sorted({case.job for case in CASES}):
        training = [result for result in semantic if result.case.job != job]
        validation = [result for result in semantic if result.case.job == job]
        threshold, _ = _select_threshold(training)
        metrics = _metrics(validation, threshold)
        print(
            f"  {job}: train threshold={threshold:.2f}, "
            f"validation precision={_percent(metrics['precision']).strip()}, "
            f"recall={_percent(metrics['recall']).strip()}, "
            f"FP={metrics['fp']}, FN={metrics['fn']}"
        )

    _, evidence_semantic = _evaluate(
        candidate_skills,
        candidate_evidence,
        job_evidence,
        embeddings,
        include_evidence=True,
    )
    evidence_semantic.extend(live_false_pairs)
    evidence_threshold, evidence_metrics = _select_threshold(evidence_semantic)
    evidence_false_positives = [
        result
        for result in evidence_semantic
        if not result.case.should_match
        and result.similarity >= evidence_threshold
    ]
    evidence_false_negatives = [
        result
        for result in evidence_semantic
        if result.case.should_match
        and result.similarity < evidence_threshold
    ]
    print("\nEvidence-enriched embedding comparison")
    print(
        f"  Selected threshold: {evidence_threshold:.2f}, "
        f"precision={_percent(evidence_metrics['precision']).strip()}, "
        f"recall={_percent(evidence_metrics['recall']).strip()}, "
        f"F1={_percent(evidence_metrics['f1']).strip()}, "
        f"FP={len(evidence_false_positives)}, "
        f"FN={len(evidence_false_negatives)}"
    )
    print("  Leave-one-job-out:")
    for job in sorted({case.job for case in CASES}):
        training = [
            result for result in evidence_semantic if result.case.job != job
        ]
        validation = [
            result for result in evidence_semantic if result.case.job == job
        ]
        threshold, _ = _select_threshold(training)
        metrics = _metrics(validation, threshold)
        print(
            f"    {job}: train threshold={threshold:.2f}, "
            f"precision={_percent(metrics['precision']).strip()}, "
            f"recall={_percent(metrics['recall']).strip()}, "
            f"FP={metrics['fp']}, FN={metrics['fn']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
