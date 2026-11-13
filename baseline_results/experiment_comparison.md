# NimbusPay Experiment Comparison

| Experiment | Train min | Inference min | JSON valid | Exact match | Mean field accuracy |
|---|---:|---:|---:|---:|---:|
| E1 - Raw QLoRA | 28.82 | Pending | 100.00% | 56.00% | 92.56% |
| E1 - Cleaned QLoRA | 24.98 | Pending | Pending | 90.00% | 97.00% |
| E2 - FP16 LoRA | 19.84 | Pending | 100.00% | 94.50% | 98.88% |
| E2 - QLoRA | 24.98 | Pending | Pending | 90.00% | 97.00% |
| E3 - Rank 8 FP16 LoRA | 18.56 | Pending | Pending | Pending | Pending |
| E4 - LR 1e-4 FP16 LoRA | Pending | Pending | Pending | Pending | Pending |

## Per-field accuracy

### E1 - Raw QLoRA

| Field | Accuracy |
|---|---:|
| category | 95.50% |
| priority | 61.50% |
| amount_paise | 100.00% |
| txn_date | 98.50% |
| txn_id | 100.00% |
| channel | 99.50% |
| language | 100.00% |
| needs_human | 85.50% |

### E1 - Cleaned QLoRA

| Field | Accuracy |
|---|---:|
| Pending | Pending |

### E2 - FP16 LoRA

| Field | Accuracy |
|---|---:|
| category | 97.00% |
| priority | 96.00% |
| amount_paise | 100.00% |
| txn_date | 99.50% |
| txn_id | 99.50% |
| channel | 100.00% |
| language | 100.00% |
| needs_human | 99.00% |

### E2 - QLoRA

| Field | Accuracy |
|---|---:|
| Pending | Pending |

### E3 - Rank 8 FP16 LoRA

| Field | Accuracy |
|---|---:|
| Pending | Pending |

### E4 - LR 1e-4 FP16 LoRA

| Field | Accuracy |
|---|---:|
| Pending | Pending |
