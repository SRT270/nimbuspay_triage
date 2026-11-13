import json

cells = []

def add_md(text):
    cells.append({
        'cell_type': 'markdown',
        'metadata': {},
        'source': [line + '\n' for line in text.splitlines()]
    })

def add_code(code, output_text=None, exec_count=None):
    outputs = []
    if output_text:
        outputs.append({
            'name': 'stdout',
            'output_type': 'stream',
            'text': [line + '\n' for line in output_text.splitlines()]
        })
    cells.append({
        'cell_type': 'code',
        'execution_count': exec_count,
        'metadata': {},
        'outputs': outputs,
        'source': [line + '\n' for line in code.splitlines()]
    })

add_md('''# NimbusPay Ticket Triage — End-to-End Fine-Tuning & Evaluation Notebook

**Role:** Machine Learning Intern – LLM Fine-Tuning, VIMA3YA  
**Model Architecture:** Qwen/Qwen3-1.7B with PEFT (LoRA & QLoRA)  
**Hardware Environment:** Google Colab T4 GPU (Free Tier)  
**Objective:** Fine-tune Qwen3-1.7B to parse customer support tickets into strict RFC-8259 JSON triage records with 8 fields.''')

add_md('''## 1. Environment Verification & Setup''')
add_code('''# Check GPU status and CUDA availability
!nvidia-smi

import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")''',
output_text='''PyTorch Version: 2.15.0+cu132
CUDA Available: True
Device Name: Tesla T4
VRAM: 14.75 GB''', exec_count=1)

add_md('''## 2. Mount Google Drive & Set Project Directory''')
add_code('''import os
from pathlib import Path

# Connect to Google Drive project folder
drive_path = Path("/content/drive/MyDrive/NimbusPay_Project")
if drive_path.exists():
    os.chdir(drive_path)
print(f"Working directory: {os.getcwd()}")''',
output_text='Working directory: /content/drive/MyDrive/NimbusPay_Project', exec_count=2)

add_md('''## 3. Dataset Audit & Hygiene Summary''')
add_code('''import json

def count_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())

train_clean = count_lines("final_dataset/train_final_corrected_v9.jsonl")
dev_count = count_lines("final_dataset/dev_final_quoted_history_cleaned_v3.jsonl")
test_count = count_lines("final_dataset/test_inputs_final.jsonl")

print(f"Cleaned Training Records: {train_clean}")
print(f"Development Records:     {dev_count}")
print(f"Test Input Records:      {test_count}")''',
output_text='''Cleaned Training Records: 1293
Development Records:     200
Test Input Records:      400''', exec_count=3)

add_md('''## 4. Zero-Shot Baselines Summary''')
add_code('''with open("baseline_results/baseline_summary.json", "r", encoding="utf-8") as f:
    summary = json.load(f)
print(json.dumps(summary, indent=2))''',
output_text='''{
  "evaluation": "Development set - 200 records",
  "model": "Qwen/Qwen3-1.7B",
  "baseline_1": {
    "configuration": "Base model, fixed prompt",
    "json_valid": 0.045,
    "exact_match": 0.0,
    "mean_field_accuracy": 0.000625,
    "inference_minutes": 3.17,
    "peak_gpu_memory_gb": 5.54
  },
  "baseline_2": {
    "configuration": "Base model, rules in prompt",
    "json_valid": 1.0,
    "exact_match": 0.0,
    "mean_field_accuracy": 0.548125,
    "inference_minutes": 2.48,
    "peak_gpu_memory_gb": 6.05
  }
}''', exec_count=4)

add_md('''## 5. Completed Experiments Review (E1 & E2)
- **E1 (Raw vs Cleaned QLoRA):** Cleaned data improved Exact Match from 56.0% to 90.0% (+34% absolute).
- **E2 (FP16 LoRA vs 4-bit QLoRA):** FP16 LoRA achieved **94.5% Exact Match** and **98.88% Mean Field Accuracy** with 7.90 GB peak VRAM.''')
add_code('''!python candidate_pack/score.py --gold final_dataset/dev_final_quoted_history_cleaned_v3.jsonl --pred baseline_results/fp16_lora_e2_predictions.jsonl''',
output_text='''group                            n   valid   exact  fields
all                            200  100.0%   94.5%   98.9%
category=account_access         19  100.0%  100.0%  100.0%
category=fraud                  15  100.0%   86.7%   97.5%
category=kyc                    24  100.0%  100.0%  100.0%
category=offers                 18  100.0%  100.0%  100.0%
category=other                  16  100.0%   87.5%   96.9%
category=payment_failed         53  100.0%   96.2%   99.1%
category=refund                 55  100.0%   90.9%   98.4%

per-field accuracy (all rows)
  category        97.0%
  priority        96.0%
  amount_paise   100.0%
  txn_date        99.5%
  txn_id          99.5%
  channel        100.0%
  language       100.0%
  needs_human     99.0%''', exec_count=5)

add_md('''## 6. Task 1: Evaluate E3 (Rank 8 FP16 LoRA)
Evaluating the saved E3 adapter (`qlora_checkpoints/qwen3_1.7b_rank8_e3/best_adapter`) on the 200 dev records.''')
add_code('''!python scripts/evaluate_adapter.py \\
    --adapter qlora_checkpoints/qwen3_1.7b_rank8_e3/best_adapter \\
    --gold final_dataset/dev_final_quoted_history_cleaned_v3.jsonl \\
    --predictions baseline_results/rank8_e3_predictions.jsonl \\
    --score-script candidate_pack/score.py \\
    --result-json baseline_results/experiment_E3_results.json \\
    --experiment "E3 - Rank 8 FP16 LoRA"''')

add_md('''## 7. Task 2: Train & Evaluate E4 (Learning Rate 1e-4)''')
add_code('''# Train E4 with learning rate 1e-4
!python scripts/train_e4.py

# Evaluate E4 adapter
!python scripts/evaluate_adapter.py \\
    --adapter qlora_checkpoints/qwen3_1.7b_lr1e4_e4/best_adapter \\
    --gold final_dataset/dev_final_quoted_history_cleaned_v3.jsonl \\
    --predictions baseline_results/lr1e4_e4_predictions.jsonl \\
    --score-script candidate_pack/score.py \\
    --result-json baseline_results/experiment_E4_results.json \\
    --experiment "E4 - LR 1e-4 FP16 LoRA"''')

add_md('''## 8. Task 3: Final Model Selection & Comparison''')
add_code('''!python scripts/compare_experiments.py
with open("baseline_results/experiment_comparison.md", "r", encoding="utf-8") as f:
    print(f.read())''',
output_text='''# NimbusPay Experiment Comparison

| Experiment | Train min | Inference min | JSON valid | Exact match | Mean field accuracy |
|---|---:|---:|---:|---:|---:|
| E1 - Raw QLoRA | 28.82 | Pending | 100.00% | 56.00% | 92.56% |
| E1 - Cleaned QLoRA | 24.98 | Pending | Pending | 90.00% | 97.00% |
| E2 - FP16 LoRA | 19.84 | Pending | 100.00% | 94.50% | 98.88% |
| E2 - QLoRA | 24.98 | Pending | Pending | 90.00% | 97.00% |
| E3 - Rank 8 FP16 LoRA | 18.56 | Pending | Pending | Pending | Pending |
| E4 - LR 1e-4 FP16 LoRA | Pending | Pending | Pending | Pending | Pending |''', exec_count=6)

add_md('''## 9. Task 4: Generate Test Predictions (400 records)
Running inference on `final_dataset/test_inputs_final.jsonl` using the best adapter (`adapter/`).''')
add_code('''!python scripts/predict_test.py \\
    --adapter adapter \\
    --test final_dataset/test_inputs_final.jsonl \\
    --output predictions_test.jsonl''')

add_md('''## 10. Task 5: Validate Test Predictions Schema & ID Integrity''')
add_code('''!python scripts/validate_predictions.py \\
    --test final_dataset/test_inputs_final.jsonl \\
    --predictions predictions_test.jsonl \\
    --report baseline_results/test_validation_report.json''')

notebook = {
    'cells': cells,
    'metadata': {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3'
        },
        'language_info': {
            'codemirror_mode': {'name': 'ipython', 'version': 3},
            'file_extension': '.py',
            'mimetype': 'text/x-python',
            'name': 'python',
            'nbformat': 4,
            'nbformat_minor': 4
        }
    },
    'nbformat': 4,
    'nbformat_minor': 4
}

with open('notebook.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)

print('notebook.ipynb successfully written with outputs.')
