"""
test_pipeline.py — Smoke test with synthetic docs.

Creates 5 fake text documents, runs the full ingestion pipeline,
then runs a query. Use this to verify everything is wired up correctly
before challenge day.

Run:
  docker compose up -d        # start Qdrant
  python test_pipeline.py
"""

import asyncio
import os
import time
from pathlib import Path

# ── Create synthetic test corpus ──────────────────────────────────────────────

TEST_DIR = "./data/test_corpus"

FAKE_DOCS = {
    "climate_report.txt": """
Climate Change Report 2024

Executive Summary
Global temperatures have risen by 1.2 degrees Celsius since pre-industrial levels.
The primary drivers are carbon dioxide emissions from fossil fuels and deforestation.

Key Findings
Arctic ice coverage has decreased by 40% over the last 50 years.
Sea levels are rising at approximately 3.3mm per year globally.
Extreme weather events have increased in frequency by 25% since 1990.

Recommendations
Immediate reduction of CO2 emissions by 45% by 2030 is required.
Renewable energy investments must triple to meet Paris Agreement targets.
Carbon capture technology deployment should be accelerated globally.
""",
    "economic_analysis.txt": """
Global Economic Outlook Q3 2024

GDP Growth Projections
The United States GDP is projected to grow at 2.4% in 2024.
The European Union faces slower growth at 1.1% due to energy costs.
Emerging markets, particularly India and Vietnam, show strong 6-7% growth.

Inflation Trends
Core inflation in the US has fallen to 3.2% from a peak of 9.1% in 2022.
The Federal Reserve has maintained rates at 5.25-5.50% since August 2023.
Energy price volatility remains the primary inflation risk factor.

Employment Data
US unemployment rate stands at 3.7%, near historical lows.
Technology sector layoffs total approximately 260,000 jobs in 2024.
Healthcare and green energy sectors show the strongest hiring growth.
""",
    "medical_research.txt": """
Advances in Cancer Treatment 2024

Immunotherapy Breakthroughs
CAR-T cell therapy shows 85% remission rates in certain blood cancers.
PD-1 inhibitors demonstrate effectiveness across 12 different cancer types.
Combination immunotherapy protocols reduce mortality by up to 40%.

Clinical Trial Results
Phase 3 trials of mRNA cancer vaccines show 44% reduction in recurrence.
Targeted therapies for BRCA mutations reduce mortality by 57% in breast cancer.
Liquid biopsy technology enables early detection in 70% of pancreatic cancer cases.

Drug Approvals
FDA approved 15 new oncology drugs in 2024, a record high.
The average cost of new cancer treatments is $180,000 per year.
Generic alternatives for key biologics reduced costs by 30% in 2024.
""",
    "technology_trends.txt": """
Technology Industry Report 2024

Artificial Intelligence
Global AI market size reached $200 billion in 2024.
Large language model parameters have grown 1000x in 5 years.
AI automation is projected to affect 300 million jobs by 2030.

Semiconductor Industry
TSMC holds 54% of global chip fabrication market share.
The US CHIPS Act invested $52 billion in domestic semiconductor production.
3nm chip production began commercial scaling in Q1 2024.

Cybersecurity
Ransomware attacks increased 50% year-over-year in 2024.
The average cost of a data breach reached $4.45 million globally.
Zero-trust architecture adoption grew to 61% of enterprises.
""",
    "legal_framework.txt": """
International Trade Law Framework

WTO Agreements
The World Trade Organization oversees 164 member countries.
The TRIPS agreement governs intellectual property rights globally.
Anti-dumping duties can be imposed when goods are sold below production cost.

Recent Developments
The US-China Phase 1 trade deal remains partially implemented as of 2024.
The EU Carbon Border Adjustment Mechanism took effect in October 2023.
Digital trade agreements now cover $7 trillion in annual transactions.

Dispute Resolution
The WTO Appellate Body has been inactive since 2019 due to US blocking appointments.
Bilateral investment treaties number over 3,000 globally.
Investor-State Dispute Settlement cases increased 40% in 2023.
"""
}

TEST_QUESTIONS = [
    "What percentage has Arctic ice coverage decreased?",
    "What is the US GDP growth projection for 2024?",
    "What remission rate does CAR-T cell therapy show?",
    "What is the global AI market size in 2024?",
    "How many member countries does the WTO have?",
]


def setup_test_corpus():
    Path(TEST_DIR).mkdir(parents=True, exist_ok=True)
    for filename, content in FAKE_DOCS.items():
        path = Path(TEST_DIR) / filename
        path.write_text(content.strip())
    print(f"[TEST] ✅ Created {len(FAKE_DOCS)} test documents in {TEST_DIR}")


async def run_test():
    from pipeline import ingest, query

    print("\n" + "="*60)
    print("LUCIO PIPELINE — SMOKE TEST")
    print("="*60 + "\n")

    setup_test_corpus()

    # Ingest
    print("--- INGESTION PHASE ---")
    t0 = time.perf_counter()
    timings = await ingest(TEST_DIR)
    ingest_time = time.perf_counter() - t0
    print(f"\nIngestion wall time: {ingest_time:.2f}s")

    # Query
    print("\n--- QUERY PHASE ---")
    t0 = time.perf_counter()
    results = await query(TEST_QUESTIONS)
    query_time = time.perf_counter() - t0

    # Print results
    print("\n--- ANSWERS ---")
    for i, r in enumerate(results, 1):
        print(f"\nQ{i:02d}: {r['question']}")
        print(f"  → {r['answer']}")
        print(f"     Source: {r['doc_name']} | Pages: {r['page_numbers']} | Conf: {r['confidence']}")

    print(f"\n{'='*60}")
    print(f"TOTAL pipeline time: {ingest_time + query_time:.2f}s")
    print(f"  Ingest : {ingest_time:.2f}s")
    print(f"  Query  : {query_time:.2f}s")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Check Anthropic key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠️  Set ANTHROPIC_API_KEY in your .env file")
    else:
        asyncio.run(run_test())
