
import json
import time
import torch

from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer


PROJECT_DIR = Path("/content/drive/MyDrive/NimbusPay_Project")
if not PROJECT_DIR.exists():
    PROJECT_DIR = Path(__file__).resolve().parent.parent

TRAIN_PATH = PROJECT_DIR / "final_dataset/train_final_corrected_v9.jsonl"
DEV_PATH = PROJECT_DIR / "final_dataset/dev_final_quoted_history_cleaned_v3.jsonl"

OUTPUT_DIR = PROJECT_DIR / "qlora_checkpoints/qwen3_1.7b_lr1e4_e4"
BEST_ADAPTER = OUTPUT_DIR / "best_adapter"

BASE_MODEL = "Qwen/Qwen3-1.7B"

# E4 controlled experiment: only learning rate changes from E2.
LEARNING_RATE = 1e-4

SEED = 42
MAX_LENGTH = 320


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def prepare_dataset(records, tokenizer):
    rows = []

    for record in records:
        messages = record["messages"]

        prompt = tokenizer.apply_chat_template(
            messages[:2],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        completion = messages[2]["content"] + tokenizer.eos_token

        rows.append({
            "prompt": prompt,
            "completion": completion
        })

    return Dataset.from_list(rows)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable. Enable a GPU runtime before E4 training."
        )

    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)
    print("Learning rate:", LEARNING_RATE)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_records = read_jsonl(TRAIN_PATH)
    dev_records = read_jsonl(DEV_PATH)

    print("Training records:", len(train_records))
    print("Development records:", len(dev_records))

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True
    )
    tokenizer.padding_side = "right"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = prepare_dataset(train_records, tokenizer)
    eval_dataset = prepare_dataset(dev_records, tokenizer)

    print("Loading FP16 base model...")

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    model.config.use_cache = False

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ]
    )

    model = get_peft_model(model, lora_config)

    # Keep trainable LoRA parameters in FP32.
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.data = param.data.float()

    model.print_trainable_parameters()

    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),

        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,

        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_steps=12,
        weight_decay=0.01,
        max_grad_norm=0.3,

        optim="paged_adamw_8bit",

        fp16=False,
        bf16=False,

        max_length=MAX_LENGTH,
        packing=False,
        completion_only_loss=True,

        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={
            "use_reentrant": False
        },

        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        logging_steps=10,
        report_to="none",

        seed=SEED,
        data_seed=SEED
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer
    )

    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    train_output = trainer.train()

    training_minutes = (time.time() - start_time) / 60
    peak_memory_gb = (
        torch.cuda.max_memory_allocated() / (1024 ** 3)
    )

    print("Training minutes:", round(training_minutes, 2))
    print("Peak GPU memory GB:", round(peak_memory_gb, 2))
    print("Best checkpoint:", trainer.state.best_model_checkpoint)

    # load_best_model_at_end restores the best checkpoint.
    trainer.save_model(str(BEST_ADAPTER))
    tokenizer.save_pretrained(str(BEST_ADAPTER))

    results = {
        "experiment": "E4 - FP16 LoRA learning rate comparison",
        "base_model": BASE_MODEL,
        "dataset": "cleaned",
        "precision": "FP16 base model with FP32 LoRA parameters",
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "learning_rate": LEARNING_RATE,
        "epochs": 3,
        "effective_batch_size": 16,
        "max_length": MAX_LENGTH,
        "training_records": len(train_records),
        "development_records": len(dev_records),
        "training_minutes": round(training_minutes, 2),
        "peak_training_memory_gb": round(peak_memory_gb, 2),
        "training_loss": train_output.training_loss,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_eval_loss": trainer.state.best_metric,
        "adapter_path": str(BEST_ADAPTER)
    }

    result_path = PROJECT_DIR / "baseline_results/experiment_E4_training.json"

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nE4 adapter saved:", BEST_ADAPTER)
    print("Training results saved:", result_path)


if __name__ == "__main__":
    main()
