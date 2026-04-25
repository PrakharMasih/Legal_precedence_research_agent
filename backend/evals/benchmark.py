"""
Benchmark dataset for automated evaluation of the Legal Precedent Research Agent.

Each BenchmarkCase encodes:
  - A realistic legal query drawn from the Indian motor accident / insurance domain
    (the primary use case from the spec).
  - Expected themes that a correct response *must* address (used for recall inference
    when no ground-truth document IDs are available from the live corpus).
  - Expected adverse themes to validate adverse-precedent identification.
  - Minimum acceptable dimension scores (used as pass/fail thresholds in pytest).

Ground-truth document IDs (ground_truth_doc_ids) are intentionally left empty here
because they depend on which PDFs are actually ingested.  The runner populates them
at runtime by querying the document repository for IDs whose file names match
GROUND_TRUTH_FILE_PATTERNS.
"""

from __future__ import annotations

from evals.schemas import BenchmarkCase

# ---------------------------------------------------------------------------
# Case 1 — motor accident, unlicensed driver, insurer denying liability
# (primary test case referenced throughout the spec)
# ---------------------------------------------------------------------------
MOTOR_ACCIDENT_INSURANCE = BenchmarkCase(
    case_id="motor_accident_001",
    query=(
        "Mrs. Lakshmi Devi was severely injured when a commercial truck (owned by Ram Transport "
        "Pvt. Ltd.) ran a red light and struck her vehicle. The truck driver, Suresh Kumar, did "
        "not hold a valid driving licence at the time of the accident. The insurer, National "
        "Assurance Co., has denied liability citing a policy exclusion for unlicensed drivers. "
        "Mrs. Devi is seeking compensation for permanent disability, loss of income, and medical "
        "expenses totalling approximately ₹35 lakh. "
        "What precedents support her claim against both the vehicle owner and the insurer? "
        "What adverse precedents could the insurer rely upon?"
    ),
    expected_themes=[
        "insurer liability unlicensed driver",
        "pay and recover doctrine",
        "third party motor accident claim",
        "MACT compensation permanent disability",
        "contributory negligence",
    ],
    expected_adverse_themes=[
        "policy exclusion breach condition",
        "insurer not liable gratuitous passenger",
        "no compensation rash negligent victim",
    ],
    min_precision=0.55,
    min_recall=0.50,
    min_reasoning=0.55,
    min_adverse=0.50,
)

# ---------------------------------------------------------------------------
# Case 2 — fatal accident, commercial vehicle, compensation quantum
# ---------------------------------------------------------------------------
FATAL_ACCIDENT_COMPENSATION = BenchmarkCase(
    case_id="fatal_accident_002",
    query=(
        "Mr. Rajesh Sharma, 38 years old, earning ₹25,000 per month, was killed when a KSRTC "
        "bus ran over him at a road crossing. His wife and two minor children are the claimants. "
        "The bus driver was found to have been driving rashly. "
        "What is the applicable compensation formula? "
        "Which judgments establish the multiplier method and notional income for homemakers?"
    ),
    expected_themes=[
        "multiplier method Sarla Verma",
        "notional income deceased non-earning",
        "loss of dependency calculation",
        "future prospects addition income",
        "KSRTC government corporation liability",
    ],
    expected_adverse_themes=[
        "contributory negligence pedestrian award reduced",
        "no proof actual income compensation nominal",
    ],
    min_precision=0.55,
    min_recall=0.45,
    min_reasoning=0.55,
    min_adverse=0.45,
)

# ---------------------------------------------------------------------------
# Case 3 — general corpus query (non-research mode)
# Tests that general queries return relevant docs, not a full research analysis
# ---------------------------------------------------------------------------
GENERAL_COMMERCIAL_VEHICLES = BenchmarkCase(
    case_id="general_query_003",
    query="Which of the indexed judgments involve commercial vehicles?",
    expected_themes=[
        "commercial vehicle",
        "truck bus lorry transport",
    ],
    expected_adverse_themes=[],  # general query — adverse dim less relevant
    min_precision=0.60,
    min_recall=0.40,
    min_reasoning=0.50,
    min_adverse=0.0,  # general query may legitimately have no adverse section
)

# ---------------------------------------------------------------------------
# Case 4 — adverse-heavy scenario: all precedents likely unfavourable
# Tests the agent's honesty when corpus evidence cuts against the client
# ---------------------------------------------------------------------------
ADVERSE_HEAVY_CASE = BenchmarkCase(
    case_id="adverse_heavy_004",
    query=(
        "Our client was the intoxicated driver who caused a fatal accident. "
        "He is now being sued by the victim's family. "
        "Are there any precedents that could limit his liability or reduce the compensation award?"
    ),
    expected_themes=[
        "drunk driving liability",
        "insurer not liable drunken driver",
        "compensation reduction rash driving",
    ],
    expected_adverse_themes=[
        "insurer entitled to recover from drunk driver",
        "enhanced compensation rash negligence",
        "criminal negligence enhanced damages",
    ],
    min_precision=0.50,
    min_recall=0.40,
    min_reasoning=0.55,
    min_adverse=0.60,  # must surface adverse cases prominently
)

# ---------------------------------------------------------------------------
# Full benchmark suite used by the runner
# ---------------------------------------------------------------------------
ALL_CASES: list[BenchmarkCase] = [
    MOTOR_ACCIDENT_INSURANCE,
    FATAL_ACCIDENT_COMPENSATION,
    GENERAL_COMMERCIAL_VEHICLES,
    ADVERSE_HEAVY_CASE,
]
