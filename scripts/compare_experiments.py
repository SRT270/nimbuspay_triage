
import csv
import json
from pathlib import Path

ROOT = Path("/content/drive/MyDrive/NimbusPay_Project")
if not ROOT.exists():
    ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "baseline_results"


def load_json(filename):
    path = RESULTS / filename
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_metrics(data):
    evaluation = data.get("evaluation", {})
    metrics = evaluation.get("all", {})

    if not metrics:
        metrics = data.get("metrics", data)

    fields = metrics.get("fields", metrics.get("field_accuracy", {}))

    return {
        "json_validity": metrics.get("json_valid", metrics.get("json_validity")),
        "exact_match": metrics.get("exact_match"),
        "mean_field_accuracy": metrics.get(
            "mean_field_acc", metrics.get("mean_field_accuracy")
        ),
        "field_accuracy": fields
    }


rows = []


def add_row(name, training=None, evaluation=None, extra=None):
    training = training or {}
    evaluation = evaluation or {}
    extra = extra or {}

    metrics = get_metrics(evaluation) if evaluation else get_metrics(training)

    rows.append({
        "experiment": name,
        "training_minutes": training.get("training_minutes"),
        "inference_minutes": evaluation.get("inference_minutes"),
        "peak_training_memory_gb": training.get("peak_training_memory_gb"),
        "peak_inference_memory_gb": evaluation.get("peak_inference_memory_gb"),
        "json_validity": metrics["json_validity"],
        "exact_match": metrics["exact_match"],
        "mean_field_accuracy": metrics["mean_field_accuracy"],
        "field_accuracy": metrics["field_accuracy"],
        **extra
    })


# E1: raw and cleaned QLoRA
e1 = load_json("experiment_E1_results.json")
add_row("E1 - Raw QLoRA", e1.get("raw"))
add_row("E1 - Cleaned QLoRA", e1.get("cleaned"))

# E2: FP16 LoRA and QLoRA
e2 = load_json("experiment_E2_results.json")
add_row("E2 - FP16 LoRA", e2.get("fp16_lora"))
add_row("E2 - QLoRA", e2.get("qlora"))

# E3: rank-8 FP16 LoRA
e3_training = load_json("experiment_E3_training.json")
e3_evaluation = load_json("experiment_E3_results.json")
add_row("E3 - Rank 8 FP16 LoRA", e3_training, e3_evaluation)

# E4: learning-rate experiment
e4_training = load_json("experiment_E4_training.json")
e4_evaluation = load_json("experiment_E4_results.json")
add_row("E4 - LR 1e-4 FP16 LoRA", e4_training, e4_evaluation)

# Write CSV
columns = [
    "experiment", "training_minutes", "inference_minutes",
    "peak_training_memory_gb", "peak_inference_memory_gb",
    "json_validity", "exact_match", "mean_field_accuracy"
]

csv_path = RESULTS / "experiment_comparison.csv"

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key) for key in columns})

# Write detailed Markdown report
md = [
    "# NimbusPay Experiment Comparison",
    "",
    "| Experiment | Train min | Inference min | JSON valid | Exact match | Mean field accuracy |",
    "|---|---:|---:|---:|---:|---:|"
]

def fmt(value, percentage=False):
    if value is None:
        return "Pending"
    return f"{value:.2%}" if percentage else f"{value:.2f}"

for row in rows:
    md.append(
        f"| {row['experiment']} "
        f"| {fmt(row['training_minutes'])} "
        f"| {fmt(row['inference_minutes'])} "
        f"| {fmt(row['json_validity'], True)} "
        f"| {fmt(row['exact_match'], True)} "
        f"| {fmt(row['mean_field_accuracy'], True)} |"
    )

md += ["", "## Per-field accuracy", ""]

for row in rows:
    md.append(f"### {row['experiment']}")
    md.append("")
    md.append("| Field | Accuracy |")
    md.append("|---|---:|")

    for field, value in row["field_accuracy"].items():
        md.append(f"| {field} | {value:.2%} |")

    if not row["field_accuracy"]:
        md.append("| Pending | Pending |")

    md.append("")

md_path = RESULTS / "experiment_comparison.md"
md_path.write_text("\n".join(md), encoding="utf-8")

print("Comparison generated:")
print(csv_path)
print(md_path)

print("\nCurrent experiment rows:")
for row in rows:
    print(
        row["experiment"],
        "| Exact match:",
        fmt(row["exact_match"], True)
    )
