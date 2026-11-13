
import argparse
import json
import torch

from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for test inference.")

    adapter_path = Path(args.adapter)
    records = read_jsonl(args.test)

    if len(records) != 400:
        raise ValueError(f"Expected 400 test records, found {len(records)}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(adapter_path),
        trust_remote_code=True
    )
    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.eval()
    model.config.use_cache = True

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seen_ids = set()
    count = 0

    with open(output_path, "w", encoding="utf-8") as out:
        for start in range(0, len(records), args.batch_size):
            batch = records[start:start + args.batch_size]

            prompts = [
                tokenizer.apply_chat_template(
                    record["messages"][:2],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
                for record in batch
            ]

            inputs = tokenizer(
                prompts,
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

            for i, record in enumerate(batch):
                record_id = record["id"]

                if record_id in seen_ids:
                    raise ValueError(f"Duplicate test ID: {record_id}")

                seen_ids.add(record_id)

                raw_output = tokenizer.decode(
                    generated[i, prompt_width:],
                    skip_special_tokens=True
                ).strip()

                out.write(json.dumps({
                    "id": record_id,
                    "output": raw_output
                }, ensure_ascii=False) + "\n")

                count += 1

            print(f"Generated {count}/{len(records)}")

    print("\nPredictions saved:", output_path)
    print("Total predictions:", count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--max-input-length", type=int, default=768)

    main(parser.parse_args())
