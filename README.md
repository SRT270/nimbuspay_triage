# NimbusPay Support Ticket Triage — LLM Fine-Tuning

**Base Model ID:** `Qwen/Qwen3-1.7B`  
**Precision / 4-bit Loading:** Evaluated both 4-bit NormalFloat (NF4 QLoRA) and full-precision FP16 base model with FP32 LoRA adapters. The **best model runs on unquantized FP16** (not loaded in 4-bit).  
**Best Result Summary:** **E2 FP16 LoRA** achieves **94.5% Exact Match**, **98.88% Mean Field Accuracy**, and **100% JSON Validity** on the 200-ticket gold development set with 7.90 GB peak training memory and 7.89 min inference time.

---

## 1. Project Overview

NimbusPay is a digital payments platform requiring automated, high-precision customer ticket triage. This repository provides the complete data auditing pipeline, baseline benchmarks, parameter-efficient fine-tuning (PEFT) configurations, error analyses, and evaluation harnesses to transform raw support tickets into strict RFC-8259 JSON objects containing exactly 8 keys:

1. `category` (`payment_failed`, `refund`, `fraud`, `account_access`, `kyc`, `offers`, `other`)
2. `priority` (`P1`, `P2`, `P3`)
3. `amount_paise` (integer paise or `null`)
4. `txn_date` (`YYYY-MM-DD` or `null`)
5. `txn_id` (`NP` + 10 digits or `null`)
6. `channel` (`upi`, `card`, `netbanking`, `wallet` or `null`)
7. `language` (`en`, `hi-en`)
8. `needs_human` (`true`, `false`)

---

## 2. Dataset Hygiene & Partitioning

| Split | Records | Description | Hygiene Status |
|---|---:|---|---|
| **Raw Training** | 1,445 | Initial crowdsourced / pipeline tickets | Contains 25 truncated rows, 506 schema defects, and contradictory labels |
| **Clean Training (`v9`)** | **1,293** | Audited, deduplicated, verified training set | Cleaned, zero label-spec contradictions (`final_dataset/train_final_corrected_v9.jsonl`) |
| **Development** | 200 | Gold-standard progress measurement set | Fixed-format, unquoted headers (`final_dataset/dev_final_quoted_history_cleaned_v3.jsonl`) |
| **Test Inputs** | 400 | Blind evaluation inputs | Strictly hold-out, unique IDs, no overlaps (`final_dataset/test_inputs_final.jsonl`) |

Dataset audit logs are archived in `audit/` and `cleaning_log.csv`.

---

## 3. Experimental Results Summary

Evaluated on 200 development tickets using greedy decoding (`max_new_tokens = 220`, `max_input_length = 768`) scored with the official evaluator `candidate_pack/score.py`:

| Experiment | Setup | Train Time | Infer Time | Peak VRAM | JSON Valid | Exact Match | Mean Field Acc | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| **Baseline 1** | Base Model + Fixed Prompt | — | 3.17 min | 5.54 GB | 4.5% | 0.0% | 0.06% | Completed |
| **Baseline 2** | Base Model + In-Context Rules | — | 2.48 min | 6.05 GB | 100.0% | 0.0% | 54.81% | Completed |
| **E1 (Raw)** | 4-bit QLoRA on Raw Data | 28.82 min | 9.36 min | 6.18 GB | 100.0% | 56.0% | 92.56% | Completed |
| **E1 (Clean)**| 4-bit QLoRA on Clean Data | 24.98 min | 7.58 min | ~6.20 GB | 100.0% | 90.0% | 97.00% | Completed |
| **E2 (FP16)** | **FP16 LoRA (r=16, Clean Data)** | **19.84 min** | **7.89 min** | **7.90 GB** | **100.0%** | **94.5%** | **98.88%** | **Best Completed** |
| **E3 (Rank 8)**| FP16 LoRA (r=8, 8.7M params) | 18.56 min | Pending | 8.11 GB | Pending | Pending | Pending | Train Done; Eval Pending |
| **E4 (LR 1e-4)**| FP16 LoRA (r=16, LR=1e-4) | Ready | Pending | Pending | Pending | Pending | Pending | Planned / Script Ready |

### Best Model Field-Level Accuracy (E2 FP16 LoRA)
```text
  category        97.0%
  priority        96.0%
  amount_paise   100.0%
  txn_date        99.5%
  txn_id          99.5%
  channel        100.0%
  language       100.0%
  needs_human     99.0%
```

---

## 4. Repository Structure

```text
NimbusPay_Project/
├── adapter/                                 # Final PEFT adapter submission folder (E2 FP16 LoRA)
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── audit/                                   # Data auditing & cleaning records
│   ├── business_rule_issues.csv
│   ├── corrected_business_rule_issues.csv
│   ├── final_business_rule_audit.csv
│   ├── malformed_records.csv
│   └── schema_issues.csv
├── baseline_results/                        # Official evaluation outputs & metrics
│   ├── baseline_summary.json
│   ├── experiment_comparison.csv
│   ├── experiment_comparison.md
│   ├── experiment_E1_results.json
│   ├── experiment_E2_results.json
│   ├── experiment_E3_training.json
│   ├── fp16_lora_e2_predictions.jsonl
│   └── raw_qlora_e1_predictions.jsonl
├── candidate_pack/                          # Assignment specification & scorer
│   ├── data/
│   ├── ASSIGNMENT.md
│   ├── SCHEMA.md
│   └── score.py
├── final_dataset/                           # Audited & partitioned datasets
│   ├── train_final_corrected_v9.jsonl
│   ├── dev_final_quoted_history_cleaned_v3.jsonl
│   └── test_inputs_final.jsonl
├── qlora_checkpoints/                       # Trained LoRA adapters
│   ├── qwen3_1.7b_fp16_lora_e2/best_adapter/
│   ├── qwen3_1.7b_nimbuspay/
│   ├── qwen3_1.7b_rank8_e3/best_adapter/
│   └── qwen3_1.7b_raw_e1/
├── reports/                                 # Detailed engineering documentation
│   ├── error_analysis.md                    # Deep dive into all 11 development set mismatches
│   └── FINAL_REPORT.md                      # Comprehensive 16-section technical report
├── scripts/                                 # Automated execution pipeline
│   ├── evaluate_adapter.py                  # Evaluates adapter against dev set using score.py
│   ├── compare_experiments.py               # Generates experiment_comparison CSV and Markdown
│   ├── train_e4.py                          # Trains learning-rate experiment E4
│   ├── predict_test.py                      # Generates 400 test predictions
│   └── validate_predictions.py              # Validates schema, IDs, and prediction structure
├── cleaning_log.csv                         # Mandatory log of dropped / modified training rows
├── notebook.ipynb                           # Sequential execution notebook for Colab T4
├── report.md                                # 3-page submission report
└── README.md                                # Repository guide & documentation
```

---

## 5. Execution & Reproducibility Guide

### Environment Requirements
- GPU: NVIDIA Tesla T4 (or equivalent with >= 8 GB VRAM)
- Python: 3.10+
- Packages: `pip install torch transformers peft trl accelerate`

### Step 1: Run E3 Adapter Evaluation (GPU)
```bash
python scripts/evaluate_adapter.py \
    --adapter qlora_checkpoints/qwen3_1.7b_rank8_e3/best_adapter \
    --gold final_dataset/dev_final_quoted_history_cleaned_v3.jsonl \
    --predictions baseline_results/rank8_e3_predictions.jsonl \
    --score-script candidate_pack/score.py \
    --result-json baseline_results/experiment_E3_results.json \
    --experiment "E3 - Rank 8 FP16 LoRA"
```

### Step 2: Train & Evaluate E4 (GPU)
```bash
python scripts/train_e4.py
python scripts/evaluate_adapter.py \
    --adapter qlora_checkpoints/qwen3_1.7b_lr1e4_e4/best_adapter \
    --gold final_dataset/dev_final_quoted_history_cleaned_v3.jsonl \
    --predictions baseline_results/lr1e4_e4_predictions.jsonl \
    --score-script candidate_pack/score.py \
    --result-json baseline_results/experiment_E4_results.json \
    --experiment "E4 - LR 1e-4 FP16 LoRA"
```

### Step 3: Generate and Validate Test Predictions
```bash
# Generate 400 predictions on test_inputs_final.jsonl
python scripts/predict_test.py \
    --adapter adapter \
    --test final_dataset/test_inputs_final.jsonl \
    --output predictions_test.jsonl

# Validate test predictions
python scripts/validate_predictions.py \
    --test final_dataset/test_inputs_final.jsonl \
    --predictions predictions_test.jsonl \
    --report baseline_results/test_validation_report.json
```

### Step 4: Regenerate Experiment Comparison
```bash
python scripts/compare_experiments.py
```
