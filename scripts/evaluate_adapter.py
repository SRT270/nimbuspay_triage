
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate(args):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Enable a GPU runtime before evaluation."
        )

    adapter_path = Path(args.adapter)
    output_path = Path(args.predictions)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(args.gold)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(adapter_path),
        trust_remote_code=True
    )
    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    print("Loading adapter...")
    model = PeftModel.from_pretrained(
        base_model,
        str(adapter_path)
    )
    model.eval()
    model.config.use_cache = True

    prompts = []
    for record in records:
        messages = record["messages"][:2]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        prompts.append(prompt)

    predictions = []

    torch.cuda.reset_peak_memory_stats()
    start = time.time()

    for start_idx in range(0, len(records), args.batch_size):
        batch_records = records[start_idx:start_idx + args.batch_size]
        batch_prompts = prompts[start_idx:start_idx + args.batch_size]

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_input_length
        )

        device = model.get_input_embeddings().weight.device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        prompt_width = inputs["input_ids"].shape[1]

        for i, record in enumerate(batch_records):
            output_tokens = generated[i, prompt_width:]
            raw_output = tokenizer.decode(
                output_tokens,
                skip_special_tokens=True
            ).strip()

            predictions.append({
                "id": record["id"],
                "output": raw_output
            })

        completed = min(start_idx + args.batch_size, len(records))
        print(f"Evaluated {completed}/{len(records)}")

        del inputs, generated

    inference_minutes = (time.time() - start) / 60
    peak_memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    with open(output_path, "w", encoding="utf-8") as f:
        for row in predictions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\nPredictions saved:", output_path)
    print("Inference minutes:", round(inference_minutes, 2))
    print("Peak inference memory GB:", round(peak_memory_gb, 2))

    score_command = [
        sys.executable,
        args.score_script,
        "--gold", args.gold,
        "--pred", str(output_path),
        "--json"
    ]

    completed = subprocess.run(
        score_command,
        capture_output=True,
        text=True,
        check=True
    )

    score_result = json.loads(completed.stdout)
    all_metrics = score_result["all"]

    print("\nOFFICIAL EVALUATION")
    print(json.dumps(all_metrics, indent=2))

    result = {
        "experiment": args.experiment,
        "adapter": str(adapter_path),
        "base_model": args.base_model,
        "evaluation_records": len(records),
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "max_input_length": args.max_input_length,
        "decoding": "greedy",
        "inference_minutes": round(inference_minutes, 2),
        "peak_inference_memory_gb": round(peak_memory_gb, 2),
        "predictions_file": str(output_path),
        "evaluation": score_result
    }

    result_path = Path(args.result_json)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\nExperiment results saved:", result_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--adapter", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--score-script", required=True)
    parser.add_argument("--result-json", required=True)

    parser.add_argument("--experiment", default="Evaluation")
    parser.add_argument("--base-model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--max-input-length", type=int, default=768)

    evaluate(parser.parse_args())
