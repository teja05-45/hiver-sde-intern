# Failure Analysis (from real golden-set misclassifications)

Golden set size: 200 | Failures: 161 (80.5%)

## out_of_distribution -- 39 cases (24.2% of failures)
**Hypothesis:** Message doesn't cleanly match any of the 12 taxonomy intents (promotional content, meta-questions about the support account itself, vague venting). This is partly by design (general_other exists to catch these and route to escalation) but the classifier sometimes assigns high confidence to a specific wrong intent instead of recognizing OOD. Fix: dedicated OOD/novelty detection layer, not just relying on general_other cluster membership.

- **[golden_007]** "@AmazonHelp hello,I have received my order but as you can see it is already opened and the cd "who built the moon" is missing, customer service have r"
  - Expected: `delivery_not_received` | Predicted: `customer_service_complaint` (confidence 0.534)
- **[golden_047]** "@AmazonHelp The first pic was the USPS status update. Very obviously unhelpful. You can see it went from being on a truck to being "someday"."
  - Expected: `general_other` | Predicted: `order_status_inquiry` (confidence 0.691)
- **[golden_051]** "Did not receive order no. 405-6021334-3005910. Received sms, item delivered to me &amp; signed by me! @118919"
  - Expected: `general_other` | Predicted: `delivery_not_received` (confidence 0.969)

## other_classifier_error -- 38 cases (23.6% of failures)
**Hypothesis:** Errors not explained by any of the above structural categories -- likely genuine TF-IDF feature-overlap between semantically related intents.

- **[golden_002]** "the one day I have a day off work and I "missed" my package from Amazon 😭"
  - Expected: `return_or_replacement_request` | Predicted: `delivery_delay` (confidence 0.698)
- **[golden_008]** "I never actually receive my “same day” orders from amazon on the same day. Maybe they’ll show up tomorrow? They say delivered. Who knows. It’s always "
  - Expected: `customer_service_complaint` | Predicted: `delivery_delay` (confidence 0.84)
- **[golden_009]** "@115850 seems they have got a huge population of prime members in india and now they don't want more customers in india https://t.co/Cngg7ux6Kr"
  - Expected: `delivery_delay` | Predicted: `content_availability_inquiry` (confidence 0.997)

## ambiguous_intent -- 33 cases (20.5% of failures)
**Hypothesis:** Two or more intents genuinely fit the message about equally well (classifier's own top-2 margin was <0.15); TF-IDF surface features can't resolve this the way semantic context could. Fix: real sentence embeddings, or a second-pass LLM classification specifically for low-margin cases.

- **[golden_025]** ".@115830 Just spoken to your team in a chat to discuss a Prime order of mine due yesterday that was not arriving until today. Without discussing it wi"
  - Expected: `cancellation_or_refund_request` | Predicted: `delivery_delay` (confidence 0.456)
- **[golden_026]** "@115821 I️ don’t understand that I️ ordered something that was to be delivered today only to find out there’s no delivery update. I️ need it by Monday"
  - Expected: `app_or_device_technical_issue` | Predicted: `delivery_not_received` (confidence 0.527)
- **[golden_064]** "@115850 @115821 @AmazonHelp @115821 Order # 402-0726587-9868359. It's been 18 days, haven't recvd item so far. Why so late? When will I receive produc"
  - Expected: `general_other` | Predicted: `delivery_not_received` (confidence 0.535)

## multi_intent_message -- 24 cases (14.9% of failures)
**Hypothesis:** Message raises multiple issues at once (e.g. a late delivery AND a billing complaint); the taxonomy assumes one intent per message. Fix: multi-label classification, or explicit 'primary vs secondary intent' extraction.

- **[golden_015]** "@115850 @AmazonHelp I'm a prime member and placed order for prime fulfilled product on 21st Nov. Forget quick delivery, it is not yet dispatched. ORDE"
  - Expected: `delivery_delay` | Predicted: `prime_membership_issue` (confidence 0.732)
- **[golden_057]** "Enough of I am sorry @115821, a classic trademark of perverse organizational dynamics @115821. FULL REFUND IS THE ONLY ANSWER WHEN A SELLER DELIBERATE"
  - Expected: `prime_membership_issue` | Predicted: `cancellation_or_refund_request` (confidence 0.493)
- **[golden_102]** "@115830 hey, whats happening with your Prime delivery dates ? used to be next day, now been quoted Thursday Delivery ???"
  - Expected: `return_or_replacement_request` | Predicted: `delivery_delay` (confidence 0.877)

## noisy_short_message -- 17 cases (10.6% of failures)
**Hypothesis:** Very short or heavily-punctuated messages lack enough token signal for TF-IDF to disambiguate. Fix: use conversation context (later customer turns), not just the root message, for classification.

- **[golden_001]** "It REALLY annoys me that each and every time I order something from Amazon prime and USPS is the carrier, they NEVER deliver my items on time!"
  - Expected: `cancellation_or_refund_request` | Predicted: `delivery_delay` (confidence 0.825)
- **[golden_021]** "I pay $100+/year for @115821 Prime and in 5+ years I’ve never had any issues. Until yesterday, when a Prime package was “Guaranteed” to arrive by the "
  - Expected: `content_availability_inquiry` | Predicted: `delivery_delay` (confidence 0.877)
- **[golden_071]** "@115830 (2/2) I’ve contacted the seller and was told to wait 2 days for a response. if I’m paying for prime then I shouldn’t have to wait for my deliv"
  - Expected: `prime_membership_issue` | Predicted: `delivery_delay` (confidence 0.937)
