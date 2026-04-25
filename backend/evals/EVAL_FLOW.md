
# Legal Precedent Research Agent Evaluation Workflow

## 1. Architecture: Hybrid Rule-Based + LLM-as-Judge

```mermaid
graph TD
    A[python -m evals.runner] --> B[_bootstrap()]
    B --> C[LegalResearchAgent + LegalAgentEvaluator (shared LLM)]
    C --> D[For each BenchmarkCase in benchmark.py]
    D --> E[agent.run(query)]
    E --> F[_run_case() normalizes response]
    F --> G[evaluator.evaluate(EvaluationInput)]
    G --> H{Scoring}
    H --> I[Rule-based (if ground truth)]
    H --> J[LLM judge (all 4 dimensions)]
    I --> K[Precision, Recall, Adverse]
    J --> L[Precision, Recall, Reasoning, Adverse]
    K --> M[Merge: Rule-based overrides LLM if ground truth]
    L --> M
    M --> N[_write_report() → evals/results/report_<timestamp>.json]
```

---

## 2. Evaluation Steps

1. **Bootstrap**
   - Loads real DB and Qdrant vector store (same as production)
   - Instantiates `LegalResearchAgent` and `LegalAgentEvaluator` with a shared LLM provider

2. **Benchmark Execution**
   - For each `BenchmarkCase` in benchmark.py:
     - Calls `agent.run(query)`
     - Normalizes the agent's response shape
     - Builds an `EvaluationInput` and calls `evaluator.evaluate()`

3. **Dimension Scoring**
   - **Precision**: Rule-based if ground truth provided (`relevant_cited / total_cited`), else LLM judge
   - **Recall**: Rule-based if ground truth provided (`gt_found / total_gt`), else LLM judge
   - **Reasoning**: Always LLM judge
   - **Adverse**: Hard override (score 0.0 if no adverse precedents), else LLM judge

4. **Merging Results**
   - Rule-based results override LLM judge for precision/recall if ground truth exists
   - Adverse override always applies if adverse list is empty
   - Reasoning always from LLM judge

5. **Overall Score & Verdict**
   - Equal 0.25 weight per dimension
   - Verdict: `poor`, `average`, `good`, `excellent` (thresholds in evaluator)

6. **Reporting**
   - Writes a JSON report to `evals/results/report_<timestamp>.json`
   - Prints a summary table to stdout

---

## 3. Dimension Details

| Dimension   | Rule-Based? | LLM Judge? | Notes                                 |
|-------------|-------------|------------|---------------------------------------|
| Precision   | Yes         | Fallback   | relevant_cited / total_cited          |
| Recall      | Yes         | Fallback   | gt_found / total_gt                   |
| Reasoning   | No          | Always     | Depth, legal accuracy, hallucinations |
| Adverse     | Override    | Otherwise  | 0.0 if adverse list empty             |

---

## 4. Benchmark Cases

| Case ID             | Purpose                                      | Adverse Threshold |
|---------------------|----------------------------------------------|------------------|
| motor_accident_001  | Insurer liability, unlicensed driver         | 0.40             |
| fatal_accident_002  | Fatal accident compensation, multiplier      | 0.40             |
| general_query_003   | General query (no adverse expected)          | 0.0              |
| adverse_heavy_004   | Intoxicated driver, must surface adverse     | 0.60             |

---

## 5. Running the Evaluation

```bash
python -m evals.runner
```

- Requires: `LLM_API_KEY`, `LLM_PROVIDER`, `LLM_MODEL` env vars
- Corpus must be indexed in Qdrant
- Output: JSON report + summary table

---

## References
- [evals/runner.py](evals/runner.py)
- [evals/evaluator.py](evals/evaluator.py)
- [evals/benchmark.py](evals/benchmark.py)
- [evals/schemas.py](evals/schemas.py)
- [evals/prompts.py](evals/prompts.py)