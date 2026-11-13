# NimbusPay Customer Support Ticket Triage: Automated Extraction and Classification via Parameter-Efficient Fine-Tuning

**Author:** Machine Learning Engineering Candidate  
**Target Model:** Qwen/Qwen3-1.7B with Parameter-Efficient Fine-Tuning (LoRA / QLoRA)  
**Evaluation Framework:** Official Scorer (`candidate_pack/score.py`)  
**Repository:** NimbusPay Customer Support Triage System  

---

## 1. Abstract

Modern digital payment platforms receive thousands of unstructured customer complaints daily across mobile apps, email, social media, and IVR voice transcripts. Automatically turning these raw messages into clean, structured database records governed by strict business logic is critical for fast routing and regulatory compliance. 

In this work, we fine-tune the open-weight **Qwen3-1.7B** model to read raw customer support tickets and output valid JSON containing eight required fields: ticket category, priority level, monetary amount in paise, transaction date, transaction identifier, payment channel, language, and human escalation requirement. 

We performed a rigorous multi-stage audit of the training data, removing malformed JSON and conflicting labels to produce a clean set of 1,293 training records. We then evaluated base-model zero-shot prompts and conducted controlled fine-tuning experiments (E1 to E4). 

Our findings show that data cleaning was the single most impactful step: switching from raw noisy training data (56.0% exact match) to audited data (90.0% exact match) produced a 34.0% absolute improvement. Furthermore, moving from 4-bit quantized LoRA to full-precision FP16 LoRA achieved **94.5% Exact Match** and **98.88% Mean Field Accuracy** on the 200-ticket development set, with 100% JSON syntactic validity and a peak training memory of 7.90 GB on a standard T4 GPU. A detailed error analysis investigates all 11 remaining development errors, identifying calendar arithmetic edge cases, urgency keyword bias, and subtle category boundaries.

---

## 2. Introduction

When payment transactions fail, users often submit panicked or ambiguous complaints. In the fintech sector, delays in identifying unauthorized transactions or regulatory threats can lead to severe customer dissatisfaction and compliance penalties from authorities such as the Reserve Bank of India (RBI).

While proprietary cloud Large Language Models (LLMs) can perform this task, deploying commercial APIs for high-volume customer service incurs high inference costs, unpredictable network latency, and data privacy concerns. 

This project demonstrates that a compact, 1.7-billion parameter open-source model (**Qwen3-1.7B**), when adapted through parameter-efficient fine-tuning (PEFT), can internalize complex business rules, multi-tier escalation thresholds, and Hinglish currency expressions. At test time, the model requires no rules or examples in the prompt, consumes under 8 GB of GPU memory, and outputs strict JSON with high accuracy.

---

## 3. Problem Statement

The goal is to process an incoming customer ticket—consisting of a metadata header (`Received` date and `Source` channel) followed by customer text—and produce a single, strictly valid JSON object with exactly eight keys:

```json
{
  "category": "refund",
  "priority": "P1",
  "amount_paise": 28466900,
  "txn_date": "2026-09-06",
  "txn_id": "NP4284607517",
  "channel": null,
  "language": "en",
  "needs_human": true
}
```

### Business Rules and Field Definitions

1. **Category:** Must be exactly one of seven values:
   - `fraud`: The customer reports a transaction they did not make, authorize, or recognize. Under the NimbusPay specification, fraud takes absolute precedence over all other categories.
   - `payment_failed`: A payment, transfer, or wallet top-up failed, was declined, remained pending, or did not reach the recipient.
   - `refund`: Money is owed back, such as duplicate charges, returned items, canceled orders, or missing merchant refunds.
   - `account_access`: Inability to log in, locked accounts, blocked PINs, forgotten passcodes, or missing OTPs.
   - `kyc`: Identity verification, document submission, Aadhaar/PAN issues, or pending verification status.
   - `offers`: Promotional cashbacks, discount coupons, scratch cards, and referral rewards.
   - `other`: General questions, card delivery tracking, account closure, or foreign exchange queries.

2. **Priority:** Evaluated strictly in order:
   - `P1`: Assigned if the category is fraud, if the disputed transaction amount is 50,000 rupees or more (5,000,000 paise), or if the customer makes a legal or regulatory threat (mentioning the RBI, banking ombudsman, consumer court, or lawyers).
   - `P2`: If not P1, assigned if the ticket is a failed payment or refund involving 5,000 rupees or more (500,000 paise), or if the category is account access.
   - `P3`: All other standard queries.

3. **Amount in Paise (`amount_paise`):** The disputed transaction amount converted to integer paise (1 rupee = 100 paise). Distractor figures such as total account balances, overdraft limits, and transfer fees must be ignored. If no transaction amount is specified, this field must be null.

4. **Transaction Date (`txn_date`):** The date the disputed transaction took place in `YYYY-MM-DD` format. Relative references (such as "yesterday", "kal", "parso", or "3 days ago") must be resolved relative to the header `Received` date. If no date is given, this field is null.

5. **Transaction Identifier (`txn_id`):** A NimbusPay identifier starting with the letters `NP` followed by 10 digits. The model must normalize the text by removing spaces and hyphens (for example, converting "np 12345 67890" to "NP1234567890"). If no identifier is present, it is null.

6. **Payment Channel (`channel`):** One of `upi`, `card`, `netbanking`, or `wallet`, recorded only when explicitly stated in the text. Otherwise, it is null.

7. **Language (`language`):** Either `en` for standard English or `hi-en` for Hinglish (Hindi written in Latin script, often mixed with English).

8. **Human Review Required (`needs_human`):** Must be `true` if the priority is P1, or if the customer explicitly mentions previous unresolved contact (such as "calling again", "third complaint", or "baar baar"). Otherwise, it is `false`.

---

## 4. Dataset Description

The NimbusPay benchmark consists of real-world style customer support tickets:

| Dataset Partition | Record Count | Description | Hygiene Status |
|---|---:|---|---|
| **Raw Training Set** | 1,445 | Initial crowdsourced and logged tickets | Contained syntax errors and conflicting labels |
| **Cleaned Training Set (`v9`)** | 1,293 | Audited, normalized, deduplicated corpus | Verified against the labelling specification |
| **Development Set** | 200 | Benchmark validation set | Held out entirely from training |
| **Test Set** | 400 | Blind evaluation inputs | Unlabelled test records with unique IDs |

There is zero overlap between the training, development, and test sets. All 400 test records have unique IDs and clean, properly formatted headers.

---

## 5. Data Preprocessing and Audit

Fine-tuning a language model on noisy data degrades its ability to follow deterministic business rules. We performed a systematic audit across two stages, logging all drops and modifications in `cleaning_log.csv` and `audit/`:

### Stage 1: Syntax and Truncation Errors
We identified and dropped 25 records where the assistant JSON was truncated mid-generation in the raw data (for example, tickets `tr-00060`, `tr-00125`, and `tr-00497`). Because these labels ended abruptly with unclosed quotation marks and missing fields, they could not be reliably salvaged without injecting assumptions.

### Stage 2: Schema Normalization and Contradiction Pruning
We reviewed 506 rows that contained informal formatting or direct business rule violations:
- **Key naming:** Replaced non-standard keys like `prio` with `priority`.
- **Date formats:** Converted European slash notation (`DD/MM/YYYY`) into standard ISO format (`YYYY-MM-DD`).
- **Identifier cleaning:** Normalized hyphenated identifiers (such as `NP-9718716109`) to standard unhyphenated form.
- **Rule contradiction pruning:** Removed 127 records where the ground-truth label directly contradicted the written specification (for example, cases where an unauthorized debit of 150,000 rupees was labelled P3 with `needs_human: false`). Training on such contradictions confuses gradient updates and impairs decision boundaries.

The resulting clean training dataset contains 1,293 high-quality records (`final_dataset/train_final_corrected_v9.jsonl`).

---

## 6. System Architecture

The overall triage pipeline operates as follows:

1. **Input Ingestion:** The system receives the ticket header (`Received` date and `Source`) and the customer message.
2. **Template Application:** The message is formatted using Qwen3's native chat template with standard system and user roles:
   > System: "You are a customer support ticket triage assistant. Read the ticket and return a single JSON object with the triage fields."
3. **PEFT Adapter Inference:** The input tokens pass through the frozen Qwen3-1.7B base model augmented with trained low-rank adapter weights across all linear attention and feed-forward layers.
4. **Greedy Generation:** The model generates tokens using greedy decoding (`temperature = 0`, `do_sample = False`, `max_new_tokens = 220`).
5. **JSON Parsing:** The output text is stripped of leading and trailing whitespace and parsed directly as a valid JSON object. No manual heuristic post-processing or external rule engines are applied.

During fine-tuning, loss is computed exclusively on the assistant response tokens. Prompt tokens are masked out of the loss calculation so that gradient updates focus entirely on learning the JSON schema and business logic.

---

## 7. Model Methodology: Parameter-Efficient Fine-Tuning

Full fine-tuning of all 1.7 billion parameters requires substantial GPU memory and risks catastrophic forgetting of general language understanding. Instead, we use **Low-Rank Adaptation (LoRA)**.

LoRA freezes the pre-trained weights and injects trainable rank decomposition matrices into each transformer layer. For a pre-trained weight matrix with dimensions $d \times k$, the update is represented as the product of two low-rank matrices $B$ (size $d \times r$) and $A$ (size $r \times k$), where the rank $r$ is much smaller than $d$ or $k$.

We targeted all seven linear projections within the Qwen architecture: the query, key, value, and output projection layers in the multi-head attention blocks, as well as the gate, up, and down projection layers in the feed-forward network.

We compared two adaptation precision modes:
- **4-bit QLoRA:** The base model is compressed to 4-bit NormalFloat (NF4) precision, and computations dynamically dequantize weights to 16-bit precision during forward and backward passes.
- **FP16 LoRA:** The base model remains in unquantized 16-bit half-precision float (`torch.float16`), while adapter updates are accumulated in full precision.

---

## 8. Fine-Tuning Configurations

All fine-tuning was conducted on a single Google Colab Tesla T4 GPU (16 GB VRAM) using the Hugging Face `peft` and `trl` libraries:

| Configuration Setting | Baseline QLoRA (E1) | FP16 LoRA (E2 - Best) | Rank-8 LoRA (E3) | LR 1e-4 LoRA (E4) |
|---|---|---|---|---|
| **Base Model** | Qwen/Qwen3-1.7B | Qwen/Qwen3-1.7B | Qwen/Qwen3-1.7B | Qwen/Qwen3-1.7B |
| **Base Model Precision** | 4-bit NormalFloat (NF4) | 16-bit Float (FP16) | 16-bit Float (FP16) | 16-bit Float (FP16) |
| **LoRA Rank ($r$)** | 16 | 16 | 8 | 16 |
| **LoRA Alpha ($\alpha$)** | 32 | 32 | 16 | 32 |
| **LoRA Dropout** | 0.05 | 0.05 | 0.05 | 0.05 |
| **Trainable Parameters** | 17,432,576 (0.98%) | 17,432,576 (0.98%) | 8,716,288 (0.49%) | 17,432,576 (0.98%) |
| **Learning Rate** | 2e-4 | 2e-4 | 2e-4 | 1e-4 |
| **LR Schedule** | Cosine with 3% warmup | Cosine with 3% warmup | Cosine with 3% warmup | Cosine with 3% warmup |
| **Effective Batch Size**| 16 (batch 2, accum 8) | 16 (batch 2, accum 8) | 16 (batch 2, accum 8) | 16 (batch 2, accum 8) |
| **Training Epochs** | 3 | 3 | 3 | 3 |
| **Maximum Length** | 320 tokens | 320 tokens | 320 tokens | 320 tokens |
| **Optimizer** | Paged AdamW 8-bit | AdamW torch | AdamW torch | AdamW torch |

---

## 9. Baseline Evaluations

To establish reference benchmarks before fine-tuning, the base `Qwen/Qwen3-1.7B` model was evaluated on the 200 development tickets under two prompting strategies:

| Baseline Configuration | Prompt Strategy | JSON Validity | Exact Match | Mean Field Accuracy | Inference Time | Peak VRAM |
|---|---|---:|---:|---:|---:|---:|
| **Baseline 1: Fixed Prompt** | System prompt only, zero rules | 4.5% | **0.0%** | 0.06% | 3.17 min | 5.54 GB |
| **Baseline 2: Rules in Prompt** | Complete rules included in prompt | 100.0% | **0.0%** | 54.81% | 2.48 min | 6.05 GB |

### Baseline Observations
- **Fixed prompt failure:** Without fine-tuning, the base model cannot generate valid JSON (only 4.5% valid syntax). It outputs conversational filler, markdown commentary, or fragmented responses.
- **Rules prompt limitation:** Placing the entire rulebook in the context window forces valid JSON syntax (100%), but fails to achieve even a single exact match across all eight fields (0.0%). The model extracts easy surface values (such as language) but consistently fails at multi-step priority calculations, date subtraction, and transaction ID normalization.

---

## 10. Experimental Comparison (E1 to E4)

All models were evaluated against the 200 held-out development tickets using the official evaluation script (`candidate_pack/score.py`):

| Experiment | Configuration | Train Time | Inference Time | Peak Train VRAM | JSON Valid | Dev Exact Match | Mean Field Accuracy | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| **E1 (Raw)** | 4-bit QLoRA on raw noisy data | 28.82 min | 9.36 min | 6.18 GB | 100.0% | 56.00% | 92.56% | Completed |
| **E1 (Cleaned)** | 4-bit QLoRA on clean audited data | 24.98 min | 7.58 min | ~6.20 GB | 100.0% | 90.00% | 97.00% | Completed |
| **E2 (FP16 LoRA)**| **Unquantized FP16 LoRA ($r=16$)** | **19.84 min** | **7.89 min** | **7.90 GB** | **100.0%** | **94.50%** | **98.88%** | **Best Completed** |
| **E2 (QLoRA)** | 4-bit QLoRA on clean data | 24.98 min | 7.58 min | ~6.20 GB | 100.0% | 90.00% | 97.00% | Completed |
| **E3 (Rank 8)** | FP16 LoRA with rank 8 (8.7M params) | 18.56 min | Pending GPU | 8.11 GB | Pending | Pending | Pending | Train Done; Eval Pending |
| **E4 (LR 1e-4)** | FP16 LoRA with lower learning rate | Script Ready | Pending GPU | Pending | Pending | Pending | Pending | Planned / Script Ready |

---

## 11. Evaluation Metrics and Key Findings

### Field-Level Accuracy Comparison Across Completed Runs

| Field | E1 Raw QLoRA | E1 Cleaned QLoRA | E2 FP16 LoRA (Best) | Gain from Cleaning | Gain from FP16 |
|---|---:|---:|---:|---:|---:|
| `category` | 95.50% | 97.50% | **97.00%** | +2.00% | -0.50% |
| `priority` | 61.50% | 95.50% | **96.00%** | **+34.00%** | +0.50% |
| `amount_paise` | 100.00% | 99.00% | **100.00%** | -1.00% | **+1.00%** |
| `txn_date` | 98.50% | 99.00% | **99.50%** | +0.50% | +0.50% |
| `txn_id` | 100.00% | 99.00% | **99.50%** | -1.00% | +0.50% |
| `channel` | 99.50% | 100.00% | **100.00%** | +0.50% | 0.00% |
| `language` | 100.00% | 100.00% | **100.00%** | 0.00% | 0.00% |
| `needs_human` | 85.50% | 90.00% | **99.00%** | +4.50% | **+9.00%** |
| **Overall Exact Match** | **56.00%** | **90.00%** | **94.50%** | **+34.00%** | **+4.50%** |

### Key Takeaways

1. **Data Cleaning is Decisive:** Training on uncleaned data resulted in severe priority errors (61.5% accuracy) and dropped overall exact match to 56.0%. Auditing the data and removing conflicting labels raised exact match to 90.0%—a 34.0% absolute improvement—while reducing training time by almost 4 minutes.
2. **FP16 Unquantized Base Precision Improves Reasoning:** Running LoRA on unquantized FP16 weights improved exact match from 90.0% to 94.5%, reducing total mismatches from 20 to 11. Human escalation accuracy improved from 90.0% to 99.0%, and monetary extraction reached a perfect 100.0%. Furthermore, avoiding 4-bit weight dequantization overhead during backward passes made FP16 training 5.14 minutes faster (19.84 min vs 24.98 min) while comfortably fitting inside the 16 GB GPU limit (7.90 GB peak).
3. **E3 Convergence Profile:** Halving the LoRA rank to 8 reduced trainable parameters by half (8.7M vs 17.4M) and finished training in 18.56 minutes. The training loss converged steadily to an evaluation loss of 0.00845 and a token accuracy of 99.87%.

---

## 12. Error Analysis (Best Model: E2 FP16 LoRA)

On the 200 development tickets, the E2 FP16 LoRA model achieved 189 exact matches (94.5%). Below is a thorough review of the 11 mismatches:

| Ticket ID | Input Summary | Model Prediction | Ground Truth (Gold) | Root Cause Analysis |
|---|---|---|---|---|
| **dv-00006** | Ticket date 2026-07-02. Customer reports fraud "day before yesterday". | `txn_date`: `"2026-06-31"` | `txn_date`: `"2026-06-30"` | **Month-boundary arithmetic:** Two days before July 2 is June 30. The model hallucinated June 31 (June has only 30 days). The model understood the relative time offset but lacked internal calendar tables. |
| **dv-00010** | "URGENT!! I didn't see the money I added to my card. INR 2,570." | `category`: `fraud`<br>`priority`: `P1` | `category`: `payment_failed`<br>`priority`: `P3` | **Urgency keyword bias:** The emotional exclamation "URGENT!!" and missing top-up funds led the model to predict unauthorized fraud rather than a failed card top-up. Priority and human review cascaded from this error. |
| **dv-00019** | "My product was faulty. Amount: 259419 rupees... Mode: credit card." | `category`: `other` | `category`: `refund` | **Implicit intent:** The customer complained of a defective product without explicitly writing the word "refund". The model defaulted to `other`. |
| **dv-00037** | "Charged twice for purchases... duplicate charges refunded. 284669 rupees." | `priority`: `P2`<br>`needs_human`: `false` | `priority`: `P1`<br>`needs_human`: `true` | **Threshold comparison slip:** The model correctly extracted 284,669 rupees, but assigned P2 instead of P1. The 50,000 rupee threshold comparison failed. |
| **dv-00044** | "Can't see refund in history. Amount: Rs. 8377... second complaint." | `priority`: `P3` | `priority`: `P2` | **P2 threshold slip:** Refunds of 5,000 rupees or more belong to P2. The model assigned P3 despite extracting the correct amount. |
| **dv-00084** | "I didn't get a PIN yet. What gives? I am on the iOS app." | `category`: `account_access`<br>`priority`: `P2` | `category`: `other`<br>`priority`: `P3` | **Gold-label inconsistency:** The labelling specification explicitly classifies PIN problems under `account_access`. The model followed the written rule, but the gold label was marked `other`. |
| **dv-00124** | "Merchant bol raha hai refund kar diya... Pichla ticket #13657. NP 7026767936 check kar lo." | `txn_id`: `null`<br>`priority`: `P2` | `txn_id`: `"NP7026767936"`<br>`priority`: `P3` | **Colloquial Hinglish extraction:** The model missed the transaction ID when surrounded by colloquial phrasing ("check kar lo") and misclassified priority as P2. |
| **dv-00157** | "Kisi ne OTP maang ke mere paise nikal liye, yeh maine nahi kiya... 140 rupaye kat gaye." | `category`: `account_access`<br>`priority`: `P2` | `category`: `fraud`<br>`priority`: `P1` | **Keyword collision:** The ticket describes social engineering where an OTP was stolen to withdraw money. The model latched onto "OTP" (an account access keyword) instead of prioritizing the unauthorized theft (fraud). |
| **dv-00177** | "There has been a red flag on my top up. Amount: Rs. 293571." | `category`: `fraud` | `category`: `payment_failed` | **Ambiguous phrasing:** The phrase "red flag" triggered the model's fraud detector, whereas the human annotator categorized it as a flagged payment failure. |
| **dv-00180** | "Item refund. Account balance is Rs. 241670. Transaction amount is ₹3,130." | `priority`: `P1` | `priority`: `P3` | **Account balance distraction:** The model extracted the correct transaction amount (₹3,130), but self-attention between the large account balance (₹241,670) and the priority head erroneously triggered P1 escalation. |
| **dv-00194** | "If I find a card I lost do I need to dispose of it? Or can I re-activate it?" | `category`: `account_access`<br>`priority`: `P2` | `category`: `other`<br>`priority`: `P3` | **Card lifecycle boundary:** Inquiring about reactivating a card was classified as account access by the model, whereas annotators treated general card queries as `other`. |

---

## 13. Limitations

1. **Hardware Constraints:** All experiments were constrained to a single Tesla T4 GPU (16 GB VRAM) on Google Colab. Larger backbones (such as 7B or 14B models) could not be explored under the competition's 4B parameter ceiling and runtime budget.
2. **Calendar Boundary Reasoning:** Without an external symbolic calendar calculator, the language model occasionally hallucinates invalid dates (such as June 31) when calculating relative date offsets across month transitions.
3. **Implicit Complaint Semantics:** When customers report product defects without explicitly asking for a refund (as in ticket `dv-00019`), the model cannot always disambiguate whether company policy treats the complaint as a financial refund or a general merchant dispute.

---

## 14. Conclusion

This project demonstrates that a compact 1.7B parameter model (**Qwen3-1.7B**), when fine-tuned using FP16 LoRA on audited data, serves as a highly accurate and deterministic ticket triage engine:
- Achieved **94.5% Exact Match** across all eight fields and **98.88% Mean Field Accuracy** on the official 200-ticket benchmark.
- Maintained **100% JSON syntactic validity** with zero malformed outputs.
- Achieved perfect **100% accuracy** on monetary extraction (`amount_paise`), payment channel identification (`channel`), and language classification (`language`).
- Confirmed that data cleaning yielded a **34.0% absolute improvement** over raw data, while unquantized FP16 weights provided an additional **4.5% improvement** over 4-bit QLoRA.

---

## 15. Future Improvements

With an additional week of development time, we recommend the following enhancements:

1. **Calendar Context Pre-Injection:** Add pre-calculated calendar reference points to the input header (such as `Yesterday: 2026-07-01 | Day Before Yesterday: 2026-06-30`). This completely eliminates calendar boundary arithmetic errors without altering prompt rules.
2. **Balance Distractor Data Augmentation:** Synthesize 150 training tickets combining high account balances with small transaction amounts to train the self-attention mechanism to decouple account balances from priority escalation.
3. **Loss Weighting on Priority and Category Tokens:** Apply higher cross-entropy loss weights to tokens governing `category` and `priority` during training to sharpen numerical threshold boundaries (₹5,000 and ₹50,000).
4. **Annotation Reconciliation:** Work with the data annotation team to resolve ambiguous edge cases (such as PIN inquiries and faulty product returns) so that ground-truth labels perfectly match written guidelines.

---

## 16. Reproducibility Instructions

### Environment Setup
Install the verified dependencies in your environment:
```bash
pip install torch transformers peft trl accelerate
```

### Reproducing E3 Evaluation (on GPU)
```bash
python scripts/evaluate_adapter.py \
    --adapter qlora_checkpoints/qwen3_1.7b_rank8_e3/best_adapter \
    --gold final_dataset/dev_final_quoted_history_cleaned_v3.jsonl \
    --predictions baseline_results/rank8_e3_predictions.jsonl \
    --score-script candidate_pack/score.py \
    --result-json baseline_results/experiment_E3_results.json \
    --experiment "E3 - Rank 8 FP16 LoRA"
```

### Reproducing E4 Training and Evaluation (on GPU)
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

### Generating and Validating Test Predictions
```bash
# Generate 400 predictions on test_inputs_final.jsonl using the best adapter
python scripts/predict_test.py \
    --adapter adapter \
    --test final_dataset/test_inputs_final.jsonl \
    --output predictions_test.jsonl

# Validate prediction count, unique IDs, and JSON schema compliance
python scripts/validate_predictions.py \
    --test final_dataset/test_inputs_final.jsonl \
    --predictions predictions_test.jsonl \
    --report baseline_results/test_validation_report.json
```

### Consolidating Comparisons
```bash
python scripts/compare_experiments.py
```
