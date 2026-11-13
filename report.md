# NimbusPay Ticket Triage — Machine Learning Project Report

**Role:** Machine Learning Intern – LLM Fine-Tuning  
**Base Model:** `Qwen/Qwen3-1.7B`  
**Best Result:** FP16 LoRA (Rank 16, Alpha 32) — **94.5% Exact Match**, **98.88% Mean Field Accuracy**, **100% JSON Validity** on the 200-ticket development set.

---

## 1. Data Audit Findings (`train.jsonl`)

The raw training dataset (1,445 records) was systematically audited against the labelling specification and cleaned down to 1,293 verified records:

1. **Truncated Records (25 rows dropped):** 25 records contained assistant JSON strings that were cut off mid-sentence (such as `tr-00060`, `tr-00125`, and `tr-00497`). These truncated records had unrecoverable syntax errors and were dropped (logged in `cleaning_log.csv`).
2. **Schema Inconsistencies (506 rows reviewed and normalized):** We audited and standardized non-standard keys (such as `prio` instead of `priority`), European date formatting (`DD/MM/YYYY`), and hyphenated transaction numbers (such as `NP-XXXX`).
3. **Severe Rule Contradictions (127 rows dropped):** We removed records where the ground-truth label directly contradicted the written rules (for example, cases where an unauthorized fraud transaction of 150,000 rupees was labelled P3 with `needs_human: false`). Training on such contradictions impairs model convergence.
4. **Data Isolation:** There is zero ID overlap between the training, development (200 records), and test (400 records) sets.

---

## 2. Controlled Experiment Comparison

Fine-tuning was conducted on a single Tesla T4 GPU using PEFT and TRL SFTTrainer with greedy decoding (`max_new_tokens = 220`, `max_input_length = 768`). Models were evaluated against the 200 development tickets using the official scorer (`candidate_pack/score.py`):

| Experiment | Configuration | Train Time | Inference Time | Peak VRAM | JSON Valid | Exact Match | Mean Field Accuracy | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| **Baseline 1** | Base Model + Fixed Prompt | — | 3.17 min | 5.54 GB | 4.5% | 0.0% | 0.06% | Completed |
| **Baseline 2** | Base Model + Rules in Prompt | — | 2.48 min | 6.05 GB | 100.0% | 0.0% | 54.81% | Completed |
| **E1 (Raw)** | 4-bit QLoRA on raw noisy data | 28.82 min | 9.36 min | 6.18 GB | 100.0% | 56.0% | 92.56% | Completed |
| **E1 (Clean)**| 4-bit QLoRA on clean audited data | 24.98 min | 7.58 min | ~6.20 GB | 100.0% | 90.0% | 97.00% | Completed |
| **E2 (FP16)** | **FP16 LoRA (Rank 16, Clean Data)** | **19.84 min** | **7.89 min** | **7.90 GB** | **100.0%** | **94.5%** | **98.88%** | **Best Completed** |
| **E3 (Rank 8)**| FP16 LoRA (Rank 8, 8.7M params) | 18.56 min | Pending | 8.11 GB | Pending | Pending | Pending | Train Done; Eval Pending |
| **E4 (LR 1e-4)**| FP16 LoRA (Rank 16, LR 1e-4) | Ready | Pending | Pending | Pending | Pending | Pending | Planned / Script Ready |

### Key Experimental Insights
- **Data Quality is the Decisive Factor:** Cleaning the training data increased Exact Match from **56.0% to 90.0%** (a 34.0% absolute improvement), proving that clean annotations matter far more than prompt engineering.
- **Unquantized FP16 Weights Outperform 4-bit QLoRA:** Training unquantized FP16 weights with LoRA delivered an additional **4.5% Exact Match gain** (94.5% vs 90.0%) and trained 5.14 minutes faster than 4-bit QLoRA while operating well within the 16 GB GPU limit (7.90 GB peak).

---

## 3. Error Analysis (Best Model: E2 FP16 LoRA)

On the 200 development tickets, the E2 FP16 LoRA model achieved 189 exact matches (94.5%). The remaining 11 mismatches break down into four clear patterns:

| Ticket ID | Prediction vs Ground Truth | Root Cause Analysis |
|---|---|---|
| **dv-00006** | `txn_date`: `"2026-06-31"` vs `"2026-06-30"` | **Month-Boundary Arithmetic:** Two days before July 2nd was calculated as June 31st (June has only 30 days). The model understood the relative time offset but lacked internal calendar tables. |
| **dv-00010** | `category`: `fraud` (P1) vs `payment_failed` (P3) | **Urgency Keyword Bias:** The customer wrote "URGENT!!" for a missing top-up; the model overreacted and classified it as unauthorized fraud instead of a failed payment. |
| **dv-00019** | `category`: `other` vs `refund` | **Implicit Complaint Intent:** The customer complained of a faulty product without explicitly writing the word "refund". The model defaulted to `other`. |
| **dv-00037** | `priority`: `P2` vs `P1` | **Numerical Threshold Comparison:** A duplicate charge of 284,669 rupees was correctly extracted, but the model assigned P2 instead of P1 (rule: amounts of 50,000 rupees or more belong to P1). |
| **dv-00044** | `priority`: `P3` vs `P2` | **P2 Threshold Slip:** An 8,377 rupee refund was assigned P3 instead of P2 (rule: refunds of 5,000 rupees or more belong to P2). |
| **dv-00084** | `category`: `account_access` vs `other` | **Gold-Label Inconsistency:** The customer asked for a missing PIN. The written specification lists PIN issues under `account_access`, but the gold label assigned `other`. |
| **dv-00124** | `txn_id`: `null` vs `"NP7026767936"` | **Colloquial Hinglish Phrasing:** The model missed the transaction identifier when embedded in colloquial phrasing ("check kar lo"). |
| **dv-00157** | `category`: `account_access` vs `fraud` | **Keyword Collision:** The customer reported an OTP scam ("OTP maang ke paise nikal liye"). The model attended to "OTP" instead of prioritizing the unauthorized theft (fraud). |
| **dv-00177** | `category`: `fraud` vs `payment_failed` | **Ambiguous Phrasing:** "Red flag on my top up" was interpreted as suspicious fraud instead of a failed top-up. |
| **dv-00180** | `priority`: `P1` vs `P3` | **Account Balance Distraction:** The customer stated their account balance (241,670 rupees) alongside a small transaction (3,130 rupees). The model extracted the right amount but escalated priority using the large balance. |
| **dv-00194** | `category`: `account_access` vs `other` | **Card Lifecycle Boundary:** Inquiring how to reactivate a lost card was classified as account access rather than a general inquiry (`other`). |

---

## 4. One-Week Improvement Plan

1. **Calendar Reference Header:** Pre-inject calendar reference points into the header (such as `Yesterday: YYYY-MM-DD`) to eliminate date arithmetic errors.
2. **Account Balance Distractor Data Augmentation:** Generate 150 synthetic tickets pairing high account balances with small transaction amounts to teach the attention layers to decouple balances from priority escalation.
3. **Targeted Loss Weighting:** Apply double cross-entropy loss weight to `category` and `priority` tokens during training to sharpen decision boundaries.
4. **Specification Alignment:** Harmonize ambiguous labels (such as PIN inquiries) with annotators to ensure complete agreement between data and guidelines.

---

## 5. AI Tools Used

- **Antigravity Assistant:** Used for dataset syntax auditing, comparative metric consolidation, error categorization, and Markdown report drafting. All models were trained and scored using open-source PyTorch, Transformers, PEFT, TRL, and the official `candidate_pack/score.py` validator.
