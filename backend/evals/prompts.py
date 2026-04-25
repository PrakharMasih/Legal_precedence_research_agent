"""LLM-as-judge evaluation prompt for the legal research agent."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# System prompt fed to the judge LLM
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """
You are an expert legal evaluator assessing the performance of a legal research AI agent.

Your task is to evaluate the agent's response across four dimensions:

1. Precision
2. Recall
3. Reasoning Quality
4. Adverse Precedent Identification

You must be strict, objective, and critical. Do not assume the agent is correct.
Base your evaluation only on the provided documents and response.
Output ONLY valid JSON — no markdown fences, no commentary before or after.
""".strip()

# ---------------------------------------------------------------------------
# User-turn prompt template (filled in by the evaluator at runtime)
# ---------------------------------------------------------------------------

JUDGE_USER_TEMPLATE = """
## INPUTS

### User Query / Case Brief:

{query}

### Retrieved Documents (Top-K):

{retrieved_docs}

### Ground Truth Relevant Documents (if available):

{ground_truth_docs}

### Agent Response:

{agent_response}

---

## EVALUATION CRITERIA

### 1. Precision

* Identify all precedents cited by the agent.
* Determine how many are actually relevant to the query.
* Penalize inclusion of weakly related or irrelevant cases.

Output:
* precision_score (0–1)
* relevant_cases_count
* total_cases_cited
* explanation

---

### 2. Recall

* Compare agent's cited cases with ground truth (if available).
* If ground truth is NOT available, infer whether important legal angles or obvious precedents
  are missing.

Output:
* recall_score (0–1)
* missed_key_precedents (list)
* explanation

---

### 3. Reasoning Quality

* Evaluate whether the agent correctly explains:
  * Why each precedent applies (or does not apply)
  * The legal principles involved
  * Alignment of facts between precedent and current case
* Penalize shallow, vague, or incorrect reasoning.

Output:
* reasoning_score (0–1)
* strengths
* weaknesses
* hallucinations (if any)

---

### 4. Adverse Precedent Identification

* Check whether the agent identified cases that could harm the client's position.
* Evaluate whether risks are clearly explained and not downplayed.

Output:
* adverse_score (0–1)
* adverse_cases_identified (list)
* missing_adverse_cases (if any)
* risk_analysis_quality

---

## FINAL OUTPUT FORMAT (STRICT JSON)

{{
  "precision": {{
    "score": <float 0-1>,
    "relevant_cases_count": <int>,
    "total_cases_cited": <int>,
    "explanation": "<string>"
  }},
  "recall": {{
    "score": <float 0-1>,
    "missed_key_precedents": ["<string>"],
    "explanation": "<string>"
  }},
  "reasoning": {{
    "score": <float 0-1>,
    "strengths": ["<string>"],
    "weaknesses": ["<string>"],
    "hallucinations": ["<string>"]
  }},
  "adverse": {{
    "score": <float 0-1>,
    "adverse_cases_identified": ["<string>"],
    "missing_adverse_cases": ["<string>"],
    "risk_analysis_quality": "<string>"
  }},
  "overall_score": <float 0-1>,
  "final_verdict": "<poor|average|good|excellent>"
}}

Be concise but precise. Avoid generic statements. Justify every score with evidence from the
response.
""".strip()
