# Baseline Results

Brand: AmazonHelp | Test set size: 7560 (silver/cluster-derived labels, temporal test split)

| Model | Accuracy | Macro P | Macro R | Macro F1 | Weighted F1 | Rare-intent avg F1 (support<500) |
|---|---|---|---|---|---|---|
| majority | 0.473 | 0.043 | 0.0909 | 0.0584 | 0.3038 | 0.0 |
| tfidf_logreg | 0.9357 | 0.8684 | 0.8704 | 0.8689 | 0.936 | 0.845 |

## Per-intent F1 (TF-IDF + LogisticRegression)

| Intent | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| general_other | 0.9783 | 0.9477 | 0.9628 | 3576 |
| delivery_delay | 0.8942 | 0.9449 | 0.9188 | 1306 |
| delivery_not_received | 0.9078 | 0.9265 | 0.917 | 1211 |
| customer_service_complaint | 0.9423 | 0.9692 | 0.9556 | 455 |
| cancellation_or_refund_request | 0.8644 | 0.8954 | 0.8796 | 306 |
| return_or_replacement_request | 0.8902 | 0.828 | 0.8579 | 186 |
| prime_membership_issue | 0.8225 | 0.858 | 0.8399 | 162 |
| app_or_device_technical_issue | 0.9441 | 0.9712 | 0.9574 | 139 |
| content_availability_inquiry | 0.9391 | 0.9076 | 0.9231 | 119 |
| order_status_inquiry | 0.92 | 0.8519 | 0.8846 | 81 |
| payment_or_billing_issue | 0.45 | 0.4737 | 0.4615 | 19 |