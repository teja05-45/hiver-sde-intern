# Live Groq Validation

- **Run timestamp (UTC):** 2026-09-16 13:49:58 
- **Provider:** groq  
- **Mode:** `LIVE` (measured, not inferred — see health check below)  
- **Model:** `openai/gpt-oss-120b`  
- **Brand / taxonomy:** AmazonHelp  
- **Retrieval:** hybrid reranking ON; index temporal (train+dev only); k=5  
- **Secrets:** none. API key is read from the environment only and is never written to this report; corpus @mentions/URLs are sanitized as `@user`.  
- **Provenance:** every row below was produced by an actual pipeline execution in this run (`scripts/validate_live_groq.py`); nothing is hand-typed or copied from an earlier run.

## Provider health check (measured)

Real verification, not configuration inference: `GET /openai/v1/models` (auth + reachability), then a minimal 1-token completion through the same provider object the API uses.

| Check | Result |
|---|---|
| models list | ok (HTTP 200) |
| 1-token completion | ok — “openai/gpt-oss-120b” responded in 961 ms |
| healthy | **true** |

## Probe results

### Probe 1: delivery_not_received (canonical AUTO demo)

> “My package says delivered but I never received it.”

| Stage | Result |
|---|---|
| Intent | `delivery_not_received` @ 1.00 (alternatives: cancellation_or_refund_request 0.00, delivery_delay 0.00) |
| Retrieval | 5 cases, top similarity 0.674, intent agreement 1.00, resolution agreement 1.00 (hybrid: True) |
| Evidence score | 0.785 |
| Generation | PASS via `groq/``openai/gpt-oss-120b` in 2236.1 ms |
| Grounding | score 0.0 |
| Claim verification | 1 supported / 1 unsupported / 1 neutral (score 0.5, passed: False) |
| OOD | score 0.2, flagged: False |
| **Decision** | **ESCALATE** (risk high) — codes: UNSUPPORTED_CLAIMS |
| Total latency | 2276 ms |

Top retrieved cases:

| # | sim | final | intent | historical customer message | recorded resolution |
|---|---|---|---|---|---|
| 1 | 0.674 | 1.185 | `delivery_not_received` | @user This order says delivered but I never received it #403-2589731-9684305 Please help | @user Please don't provide your order details, we consider it personal information. Our Twitter page is visible to public. ^HA |
| 2 | 0.613 | 1.137 | `delivery_not_received` | @user my package says delivered but isn't here.. 🙄 | @user If it's been more than 36 hours, please reach us here: for available options. ^JM |
| 3 | 0.670 | 1.12 | `delivery_not_received` | @user my package says delivered but never got it. Where is my package !!!! | @user Thanks for reaching out to us. Here are some steps to try in situations like these: ^GG |

**Generated response (actual):**

> I’m sorry to hear that your package is marked as delivered but you haven’t received it
> To help us investigate this for you, please get in touch with our support team through the official help channels
> If it’s been more than 36 hours since the delivery status was updated, let us know so we can look into the available options for you.

**Claim verification (extracted from the actual draft):**

- · `greeting` — “I'm sorry to hear that your package is marked as delivered but you haven't received it.” (confidence 0.00; Social filler / empathy with no checkable factual content.)
- ✕ `unsupported` — “To help us investigate this for you, please get in touch with our support team through the official help channels.” (confidence 0.12; Best evidence match reached only 8% word overlap (below the 60% threshold for this claim, longest verbatim span 1 words) — no retrieved historical cas)
- ✓ `supported` — “If it's been more than 36 hours since the delivery status was updated, let us know so we can look into the available options for you.” (confidence 0.54; 53% word overlap with the retrieved evidence (best case: AmazonHelp_881062))

**Why ESCALATE:** The drafted response contains claims not supported by retrieved historical evidence.

### Probe 2: refund

> “My refund hasn't arrived yet.”

| Stage | Result |
|---|---|
| Intent | `cancellation_or_refund_request` @ 1.00 (alternatives: delivery_delay 0.00, return_or_replacement_request 0.00) |
| Retrieval | 5 cases, top similarity 0.614, intent agreement 0.40, resolution agreement 1.00 (hybrid: True) |
| Evidence score | 0.520 |
| Generation | PASS via `groq/``openai/gpt-oss-120b` in 1985.3 ms |
| Grounding | score 0.0 |
| Claim verification | 1 supported / 1 unsupported / 2 neutral (score 0.5, passed: False) |
| OOD | score 0.3, flagged: False |
| **Decision** | **ESCALATE** (risk high) — codes: UNSUPPORTED_CLAIMS |
| Total latency | 2017 ms |

Top retrieved cases:

| # | sim | final | intent | historical customer message | recorded resolution |
|---|---|---|---|---|---|
| 1 | 0.614 | 0.83 | `general_other` | My Amazon order still hasn't arrived yet... @user | @user Sorry to hear this. Have we missed the delivery date advised? ^PK |
| 2 | 0.311 | 0.79 | `cancellation_or_refund_request` | why hasn't @user give me my refund lmaoooooo | @user Without personal info, what issue have you had? ^MC |
| 3 | 0.291 | 0.74 | `cancellation_or_refund_request` | When you are initiating my refund. @user @user. | @user We have already responded to your query here: Request you to check and reply there. ^NK |

**Generated response (actual):**

> I’m sorry to hear that your refund hasn’t arrived yet
> We’d like to look into this for you
> Could you please reach out to us through our official help channels and provide the order details so we can investigate further? Thank you for your patience.

**Claim verification (extracted from the actual draft):**

- · `greeting` — “I'm sorry to hear that your refund hasn't arrived yet.” (confidence 0.00; Social filler / empathy with no checkable factual content.)
- ✓ `supported` — “We'd like to look into this for you.” (confidence 0.75; 75% word overlap with the retrieved evidence (best case: AmazonHelp_2147991))
- ✕ `unsupported` — “Could you please reach out to us through our official help channels and provide the order details so we can investigate further?” (confidence 0.25; Best evidence match reached only 25% word overlap (below the 45% threshold for this suggested action, longest verbatim span 2 words) — no retrieved hi)
- · `greeting` — “Thank you for your patience.” (confidence 0.00; Social filler / empathy with no checkable factual content.)

**Why ESCALATE:** The drafted response contains claims not supported by retrieved historical evidence.

### Probe 3: technical issue

> “The app crashes every time I open my order history.”

| Stage | Result |
|---|---|
| Intent | `app_or_device_technical_issue` @ 1.00 (alternatives: delivery_not_received 0.00, general_other 0.00) |
| Retrieval | 5 cases, top similarity 0.187, intent agreement 1.00, resolution agreement 1.00 (hybrid: True) |
| Evidence score | 0.497 |
| Generation | PASS via `groq/``openai/gpt-oss-120b` in 1683.4 ms |
| Grounding | score 0.0 |
| Claim verification | 4 supported / 1 unsupported / 3 neutral (score 0.8, passed: False) |
| OOD | score 0.5296, flagged: True |
| **Decision** | **ESCALATE** (risk high) — codes: UNSUPPORTED_CLAIMS |
| Total latency | 1722 ms |

Top retrieved cases:

| # | sim | final | intent | historical customer message | recorded resolution |
|---|---|---|---|---|---|
| 1 | 0.187 | 0.761 | `app_or_device_technical_issue` | @user quick question... How am I supposed to connect the dot to the Wi-Fi if the app constantly crashes? | @user Have you tried these troubleshooting steps? Let us know! ^CN |
| 2 | 0.186 | 0.761 | `app_or_device_technical_issue` | When will the @user app offer in-app purchases?? | @user In-app purchases should be allowed. Have you had a chance to download the new Appstore: ^SJ |
| 3 | 0.255 | 0.731 | `app_or_device_technical_issue` | Is the @user app down? | @user The app seems to work fine, have you tried uninstalling and reinstalling the app? In case you haven't, please try and keep us posted. ^VN |

**Generated response (actual):**

> Hi there, I'm sorry to hear the app crashes whenever you open your order history
> Could you please try the following steps? 1
> Log out of the app, then log back in
> 2
> Uninstall the app and reinstall it from your device’s app store
> 3
> Make sure the app is updated to the latest version
> If the issue persists after these steps, please let us know and we’ll pass your feedback along to the team for further investigation
> Thank you for your patience!

**Claim verification (extracted from the actual draft):**

- · `greeting` — “Hi there,” (confidence 0.00; Social filler / empathy with no checkable factual content.)
- · `greeting` — “I'm sorry to hear the app crashes whenever you open your order history.” (confidence 0.00; Social filler / empathy with no checkable factual content.)
- ✓ `supported` — “Could you please try the following steps?” (confidence 0.50; 50% word overlap with the retrieved evidence (best case: AmazonHelp_2588259))
- ✓ `supported` — “Log out of the app, then log back in.” (confidence 0.50; 50% word overlap with the retrieved evidence (best case: AmazonHelp_227920))
- ✓ `supported` — “Uninstall the app and reinstall it from your device's app store.” (confidence 0.50; 50% word overlap with the retrieved evidence (best case: AmazonHelp_683094))
- ✕ `unsupported` — “Make sure the app is updated to the latest version.” (confidence 0.00; Best evidence match reached only 0% word overlap (below the 45% threshold for this suggested action, longest verbatim span 0 words) — no retrieved his)
- ✓ `supported` — “If the issue persists after these steps, please let us know and we'll pass your feedback along to the team for further investigation.” (confidence 0.62; 50% word overlap with the retrieved evidence (best case: AmazonHelp_2588259))
- · `greeting` — “Thank you for your patience!” (confidence 0.00; Social filler / empathy with no checkable factual content.)

**Why ESCALATE:** The drafted response contains claims not supported by retrieved historical evidence.

## Interpretation notes (honest)

- Probe 7 (OOD) may reach a decision via `RETRIEVAL_DISAGREEMENT` rather than `OOD_REQUEST`: the corpus genuinely contains generic "what is …" questions, so nearest-neighbor similarity is moderate and the OOD score lands below the threshold; scattered retrieval intents still force escalation. That is defense-in-depth working as designed, not a silent pass.
- Probes whose drafts contain unsupported claims escalate via `UNSUPPORTED_CLAIMS` — the claim verifier runs on the ACTUAL draft text, so the grounding shown here matches the response shown here.
- Latencies are single-shot measurements from a residential connection; treat them as indicative, not as benchmarks.
- The mock-mode banner (if present above) labels a non-live run. This file is regenerated by `scripts/validate_live_groq.py`; re-run it after any pipeline change that could alter behavior.
