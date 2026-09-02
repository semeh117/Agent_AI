# Agent 2 Embedding Model Comparison

**Decision:** Use `sentence-transformers/all-MiniLM-L6-v2` for Agent 2's cosine-similarity skill matching.

**Evaluation date:** 25 August 2026

## Objective

Agent 2 parses a candidate CV and LinkedIn job descriptions, then compares every required job skill with the candidate's skills. This evaluation determines which embedding model gives the most reliable semantic skill matches.

## Fair test setup

All three models were tested with exactly the same:

- Parsed CV fixture and three parsed LinkedIn job fixtures.
- 62 candidate skills.
- 65 manually reviewed job requirements.
- Literal-matching logic and cosine-matching implementation.
- Human labels defining whether each requirement should match the CV.
- Precision target of at least 90%.

Of the 65 reviewed decisions, 7 were resolved by deterministic literal matching and 58 required semantic embedding comparison. Another 20 ambiguous or noisy parser outputs were excluded from calibration so they would not unfairly affect the embedding comparison.

## Results

| Model | Selected threshold | Precision | Recall | F1 | True positives | False positives | False negatives |
|---|---:|---:|---:|---:|---:|---:|---:|
| **MiniLM** | **0.59** | **90.0%** | **32.1%** | **47.4%** | **9** | 1 | **19** |
| BGE-base | 0.87 | 100.0% | 10.7% | 19.4% | 3 | **0** | 25 |
| iMocha skill extractor | 0.98 | 100.0% | 0.0% | 0.0% | 0 | **0** | 28 |

Threshold values are model-specific because each model produces a different similarity-score distribution. A higher numerical threshold does not mean that one model is better than another.

## How to read the metrics

- **Precision:** Of the skills classified as matches, how many were correct? High precision prevents misleading recommendations.
- **Recall:** Of all real skill matches, how many did the model find? Low recall incorrectly reports skills the candidate has as missing.
- **F1:** The combined balance between precision and recall.

## Why MiniLM was selected

MiniLM was the only tested model that met the 90% precision target while retaining useful recall. It found 9 of the 28 true semantic matches and produced only one false positive.

BGE-base produced no false positives, but it found only 3 of the 28 true matches. It therefore marked 25 skills as missing even though the candidate had relevant evidence. Its F1 score was less than half of MiniLM's.

At the precision target, iMocha found none of the 28 true matches. Lowering its threshold to `0.60` increased recall to 92.9% and F1 to 68.4%, but precision fell to 54.2%, producing 22 false positives. Examples included:

- `PyTorch` matched with `pgvector`.
- `classification` matched with `golden datasets`.
- `CI/CD` matched with `prompt versioning`.
- `regression` matched with `regression suites`.

This behaviour is consistent with a task mismatch: iMocha is designed primarily to map job-description sentences to standardized taxonomy skills, whereas Agent 2 currently compares individual job-skill phrases with individual CV-skill phrases.

## Known MiniLM limitation

MiniLM's single false positive was:

- `regression` matched with `regression suites` at cosine similarity `0.665`.

The terms share vocabulary but refer to different concepts: machine-learning regression versus software regression testing. This should be addressed with contextual disambiguation or a targeted matching rule rather than raising the global threshold and losing additional true matches.

## Conclusion

For the current Agent 2 data and matching architecture, MiniLM provides the strongest evidence-based trade-off:

- It satisfies the 90% precision requirement.
- It has three times the recall of BGE-base.
- It has the highest F1 score among models that satisfy the precision requirement.
- It produces useful matches instead of rejecting every semantic relationship.

The selected embedding configuration is:

```env
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

The calibrated matcher threshold is `0.59`. In the current code this is a matcher argument/default, not an environment variable.

## Evaluation scope

This result supports the current prototype decision; it is not a universal benchmark. The calibration currently represents one CV and three LinkedIn jobs. Before production deployment, the evaluation set should be expanded with more CVs, job domains, and manually reviewed skill pairs.

## Reproducibility

The comparison can be reproduced using:

- `fixtures/cv_parser_fixture.json`
- `fixtures/linkedin_job_parser_fixture.json`
- `dev/calibrate_cosine_threshold.py`
