#!/usr/bin/env python3
"""
Apply manual label corrections to the golden set, made by the primary
labeler (Claude) reading every one of the 200 messages individually
against the configs/intents.yaml taxonomy definitions -- NOT by re-running
the classifier. This step exists because the initial "draft_label"
function in build_golden_set.py fell back to the TF-IDF+LogReg
classifier's own prediction whenever no strong keyword rule fired (188 of
200 examples), which would make evaluating that same classifier against
this "golden" set circular. Every correction below is the result of
reading the actual message text; where a message was genuinely
ambiguous, the original label was left in place rather than forced to
change. See data/golden/README.md, "Labeling methodology and
limitations" for a full accounting, including how many labels changed
in this pass (a concrete data point on the classifier's real-world error
rate on hard/ambiguous examples, discussed in reports/misleading_headline_number.md).
"""
import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_CSV = REPO_ROOT / "data" / "golden" / "golden_set.csv"

# id -> corrected_intent (only entries that actually changed from the draft)
CORRECTIONS = {
    "golden_001": "cancellation_or_refund_request",
    "golden_003": "delivery_delay",
    "golden_008": "customer_service_complaint",
    "golden_012": "general_other",
    "golden_019": "customer_service_complaint",
    "golden_020": "delivery_delay",
    "golden_021": "content_availability_inquiry",
    "golden_022": "order_status_inquiry",
    "golden_024": "delivery_not_received",
    "golden_025": "cancellation_or_refund_request",
    "golden_026": "app_or_device_technical_issue",
    "golden_028": "delivery_delay",
    "golden_031": "payment_or_billing_issue",
    "golden_035": "return_or_replacement_request",
    "golden_038": "delivery_delay",
    "golden_047": "general_other",
    "golden_048": "general_other",
    "golden_049": "return_or_replacement_request",
    "golden_050": "delivery_not_received",
    "golden_054": "account_access_issue",
    "golden_057": "prime_membership_issue",
    "golden_058": "content_availability_inquiry",
    "golden_060": "cancellation_or_refund_request",
    "golden_061": "customer_service_complaint",
    "golden_063": "delivery_not_received",
    "golden_065": "return_or_replacement_request",
    "golden_066": "delivery_not_received",
    "golden_068": "general_other",
    "golden_073": "general_other",
    "golden_078": "customer_service_complaint",
    "golden_079": "cancellation_or_refund_request",
    "golden_080": "general_other",
    "golden_082": "delivery_delay",
    "golden_083": "return_or_replacement_request",
    "golden_084": "return_or_replacement_request",
    "golden_086": "return_or_replacement_request",
    "golden_087": "payment_or_billing_issue",
    "golden_090": "general_other",
    "golden_092": "delivery_delay",
    "golden_095": "order_status_inquiry",
    "golden_097": "order_status_inquiry",
    "golden_098": "delivery_not_received",
    "golden_099": "payment_or_billing_issue",
    "golden_100": "return_or_replacement_request",
    "golden_101": "delivery_not_received",
    "golden_104": "payment_or_billing_issue",
    "golden_109": "cancellation_or_refund_request",
    "golden_111": "app_or_device_technical_issue",
    "golden_112": "delivery_delay",
    "golden_113": "cancellation_or_refund_request",
    "golden_119": "return_or_replacement_request",
    "golden_123": "delivery_delay",
    "golden_124": "return_or_replacement_request",
    "golden_125": "account_access_issue",
    "golden_129": "delivery_delay",
    "golden_131": "customer_service_complaint",
    "golden_134": "customer_service_complaint",
    "golden_135": "general_other",
    "golden_137": "return_or_replacement_request",
    "golden_139": "app_or_device_technical_issue",
    "golden_142": "payment_or_billing_issue",
    "golden_143": "delivery_not_received",
    "golden_144": "delivery_not_received",
    "golden_148": "payment_or_billing_issue",
    "golden_149": "cancellation_or_refund_request",
    "golden_153": "customer_service_complaint",
    "golden_154": "app_or_device_technical_issue",
    "golden_156": "content_availability_inquiry",
    "golden_159": "general_other",
    "golden_161": "delivery_not_received",
    "golden_163": "app_or_device_technical_issue",
    "golden_165": "return_or_replacement_request",
    "golden_167": "account_access_issue",
    "golden_169": "app_or_device_technical_issue",
    "golden_170": "delivery_delay",
    "golden_171": "customer_service_complaint",
    "golden_173": "delivery_not_received",
    "golden_175": "delivery_not_received",
    "golden_178": "return_or_replacement_request",
    "golden_180": "account_access_issue",
    "golden_182": "delivery_delay",
    "golden_184": "account_access_issue",
    "golden_187": "app_or_device_technical_issue",
    "golden_188": "account_access_issue",
    "golden_189": "general_other",
    "golden_190": "app_or_device_technical_issue",
    "golden_191": "app_or_device_technical_issue",
    "golden_192": "app_or_device_technical_issue",
    "golden_193": "delivery_not_received",
    "golden_194": "general_other",
    "golden_195": "account_access_issue",
    "golden_196": "app_or_device_technical_issue",
    "golden_198": "cancellation_or_refund_request",
    "golden_200": "general_other",
}

with GOLDEN_CSV.open(encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
    fieldnames = list(rows[0].keys())

n_changed = 0
for r in rows:
    if r["id"] in CORRECTIONS:
        new_intent = CORRECTIONS[r["id"]]
        if new_intent != r["intent"]:
            r["labeler_notes"] = (r["labeler_notes"] + " " if r["labeler_notes"] else "") + \
                f"[corrected from '{r['intent']}' to '{new_intent}' during manual primary-labeler review]"
            r["intent"] = new_intent
            r["label_source"] = "manual_primary_labeler_review"
            n_changed += 1

with GOLDEN_CSV.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Applied {len(CORRECTIONS)} corrections; {n_changed} labels actually changed value.")
print(f"({len(CORRECTIONS) - n_changed} entries in CORRECTIONS already matched the draft label.)")
