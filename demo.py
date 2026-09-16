#!/usr/bin/env python3
"""
Demo: run 5 representative cases through the full agent pipeline and show
the reasoning behind each decision. Requires trained artifacts in
models/ (run scripts/train_and_save_agent.py first if missing).

Usage:
    python demo.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))

import joblib
from app.providers.llm.factory import get_llm_provider
from app.core.config import get_settings
from app.services.agent import SupportAgent
from app.escalation.policy import EscalationThresholds

REPO_ROOT = Path(__file__).resolve().parent
MODELS_DIR = REPO_ROOT / "models"

CASES = [
    ("Easy AUTO case (clear intent, strong evidence)",
     "My package never arrived even though tracking says it was delivered."),
    ("Medium AUTO case (common intent, decent evidence)",
     "Where is my order? It was supposed to arrive yesterday."),
    ("High-risk ESCALATION (financial/account intent)",
     "My card was charged twice for the same order, please refund the extra charge immediately."),
    ("Weak retrieval -> ESCALATION (unusual/specific phrasing)",
     "The barcode on my replacement unit doesn't match the serial number on my original invoice, is that expected?"),
    ("Ambiguous/OOD -> ESCALATION (vague, no clear actionable intent)",
     "This is getting ridiculous at this point honestly."),
]


def main() -> None:
    settings = get_settings()
    brand = settings.brand_name or "AmazonHelp"

    classifier_path = MODELS_DIR / f"classifier_{brand}.joblib"
    retriever_path = MODELS_DIR / f"retriever_{brand}.joblib"
    if not classifier_path.exists() or not retriever_path.exists():
        print(f"Model artifacts not found in {MODELS_DIR}. Run:\n"
              f"  python scripts/train_and_save_agent.py --brand {brand}\nfirst.")
        sys.exit(1)

    classifier = joblib.load(classifier_path)
    retriever = joblib.load(retriever_path)
    weights = json.loads((MODELS_DIR / "evidence_weights.json").read_text())
    intents_cfg = json.loads((MODELS_DIR / "intents_cfg.json").read_text())
    provider = get_llm_provider(settings)

    print(f"LLM provider: {provider.provider_name} "
          f"({'MOCK MODE -- see README for what this means' if settings.is_mock_mode() else 'LIVE'})")
    print("=" * 100)

    agent = SupportAgent(classifier, retriever, provider, weights, intents_cfg, EscalationThresholds())

    for label, message in CASES:
        print(f"\n### {label}")
        print(f"Customer: {message}\n")
        result = agent.respond(message, k=5)

        print(f"  Intent:              {result.intent}  (confidence={result.intent_confidence:.3f})")
        print(f"  Evidence quality:    {result.evidence_score:.3f}  "
              f"({result.evidence.num_cases} supporting cases, "
              f"top similarity={result.evidence.top_similarity:.3f})")
        if result.evidence.cases:
            top = result.evidence.cases[0]
            print(f"  Top historical case: \"{top.customer_message[:80]}\"")
            print(f"                    -> \"{(top.resolution or '')[:80]}\"")
        if result.generated:
            print(f"  Draft reply:         {result.generated.draft_reply[:150]}")
            print(f"  Grounding score:     {result.grounding_score}")
        print(f"  DECISION:            {result.decision.public_decision}  (risk={result.decision.risk_level.value})")
        print(f"  Reason:              {result.decision.reason}")
        print("-" * 100)


if __name__ == "__main__":
    main()
