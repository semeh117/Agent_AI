"""Evaluate production skill rules on pinned captured and synthetic cases.

--check validates data without embeddings. --output selects the JSON report.
This script never changes the production threshold or calls a model API.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
from dev.evaluation_cases import CAPTURED_LABELS, SYNTHETIC_PROFILES

ROOT = Path(__file__).resolve().parent.parent


def load_cases():
    manifest = json.loads((ROOT / "fixtures/evaluation_manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["sha256"].items():
        if hashlib.sha256((ROOT / "fixtures" / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"{name} changed. Review labels and update the evaluation manifest.")
    from dev.replay_parser_fixtures import replay_cv, replay_jobs
    cv, jobs = replay_cv(), replay_jobs()
    raw = json.loads((ROOT / "fixtures/linkedin_job_parser_fixture.json").read_text(encoding="utf-8"))["jobs"]
    cases, found = [], set()
    for entry, parsed in zip(raw, jobs, strict=True):
        match = re.search(r"(\d+)(?:[/?#].*)?$", entry["linkedin_job"]["url"])
        identity = match.group(1) if match else ""
        labels = CAPTURED_LABELS.get(identity)
        if labels is None:
            continue
        found.add(identity)
        for polarity, expected in [("positive", True), ("negative", False)]:
            for skill in labels[polarity]:
                if skill not in parsed["required_skills"]:
                    raise ValueError(f"Label absent from replayed requirements: {identity}: {skill}")
                cases.append({"group": "captured-cv", "job": identity, "kind": "captured",
                              "skills": cv["skills"], "required": skill, "expected": expected})
    if found != set(CAPTURED_LABELS):
        raise ValueError("Labelled job IDs are missing from the fixtures.")
    for profile in SYNTHETIC_PROFILES:
        for polarity, expected in [("positive", True), ("negative", False)]:
            cases.extend({"group": profile["id"], "job": profile["id"], "kind": "synthetic",
                          "skills": profile["skills"], "required": skill, "expected": expected}
                         for skill in profile[polarity])
    return cases


def metrics(rows):
    tp = sum(row["expected"] and row["matched"] for row in rows)
    fp = sum(not row["expected"] and row["matched"] for row in rows)
    fn = sum(row["expected"] and not row["matched"] for row in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"count": len(rows), "tp": tp, "fp": fp, "fn": fn, "tn": len(rows)-tp-fp-fn,
            "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(2*precision*recall/(precision+recall), 4) if precision+recall else 0.0}


class CachedEmbeddings:
    def __init__(self, inner):
        self.inner, self.cache = inner, {}

    def embed_documents(self, texts):
        missing = list(dict.fromkeys(text for text in texts if text not in self.cache))
        if missing:
            self.cache.update(zip(missing, self.inner.embed_documents(missing), strict=True))
        return [self.cache[text] for text in texts]


def evaluate(cases, embeddings, threshold):
    from core.cosine_matcher import _skills_match_cosine
    rows = []
    for case in cases:
        result = _skills_match_cosine(case["skills"], [case["required"]], threshold, embeddings, use_esco=False)
        rows.append({key: value for key, value in case.items() if key != "skills"} |
                    {"matched": bool(result["matching"]), "evidence": result["matching"]})
    return rows


def select_threshold(sweep):
    eligible = [(threshold, metrics(rows)) for threshold, rows in sweep.items() if metrics(rows)["precision"] >= 0.9]
    return max(eligible, key=lambda item: (item[1]["recall"], item[1]["f1"], item[0]))[0] if eligible else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", default="output/evaluation.json")
    args = parser.parse_args()
    cases = load_cases()
    if args.check:
        print(f"Evaluation labels validated: {len(cases)} cases, 5 captured jobs, 5 synthetic profiles.")
        return 0
    from config import get_embeddings, EMBEDDING_MODEL
    from core.cosine_matcher import DEFAULT_COSINE_THRESHOLD
    embeddings = CachedEmbeddings(get_embeddings())
    embeddings.embed_documents(list(dict.fromkeys(text for case in cases for text in [*case["skills"], case["required"]])))
    thresholds = sorted({DEFAULT_COSINE_THRESHOLD, *[value/100 for value in range(50, 100, 5)]})
    sweep = {threshold: evaluate(cases, embeddings, threshold) for threshold in thresholds}
    baseline, held_out = sweep[DEFAULT_COSINE_THRESHOLD], []
    # Keep jobs sharing a real CV together to avoid candidate leakage.
    for group in sorted({case["group"] for case in cases}):
        training = {threshold: [row for row in rows if row["group"] != group] for threshold, rows in sweep.items()}
        chosen = select_threshold(training)
        held_out.append({"group": group, "training_selected_threshold": chosen,
                         "metrics": metrics([row for row in sweep[chosen] if row["group"] == group]) if chosen else None})
    report = {
        "label_status": "Assistant-authored provisional labels; independent domain review required.",
        "scope": "One real CV / five jobs plus five synthetic profiles. Individual skill evidence only; no hiring or ranking accuracy claim.",
        "matcher": "Production literal, alias, capability and semantic rules; ESCO disabled to isolate rules and embeddings.",
        "embedding_model": EMBEDDING_MODEL, "production_threshold": DEFAULT_COSINE_THRESHOLD,
        "production_metrics": {kind: metrics([row for row in baseline if row["kind"] == kind]) for kind in ["captured", "synthetic"]},
        "exploratory_threshold": select_threshold(sweep),
        "threshold_sweep": {str(threshold): metrics(rows) for threshold, rows in sweep.items()},
        "held_out_by_profile": held_out,
        "errors_at_production_threshold": [row for row in baseline if row["matched"] != row["expected"]],
        "decisions_at_production_threshold": baseline,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"metrics": report["production_metrics"], "exploratory_threshold": report["exploratory_threshold"], "report": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
