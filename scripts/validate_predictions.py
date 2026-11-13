
import json
import re
import argparse
from pathlib import Path
from datetime import date

ROOT = Path("/content/drive/MyDrive/NimbusPay_Project")
if not ROOT.exists():
    ROOT = Path(__file__).resolve().parent.parent

EXPECTED_KEYS = {
    "category", "priority", "amount_paise", "txn_date",
    "txn_id", "channel", "language", "needs_human"
}

CATEGORIES = {
    "payment_failed", "refund", "fraud",
    "account_access", "kyc", "offers", "other"
}
PRIORITIES = {"P1", "P2", "P3"}
CHANNELS = {"upi", "card", "netbanking", "wallet"}


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def validate_output(raw):
    errors = []

    try:
        obj = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return ["invalid_json"]

    if not isinstance(obj, dict):
        return ["output_not_object"]

    if set(obj.keys()) != EXPECTED_KEYS:
        errors.append("incorrect_keys")

    if obj.get("category") not in CATEGORIES:
        errors.append("invalid_category")

    if obj.get("priority") not in PRIORITIES:
        errors.append("invalid_priority")

    amount = obj.get("amount_paise")
    if amount is not None and (
        type(amount) is not int or amount < 0
    ):
        errors.append("invalid_amount_paise")

    txn_date = obj.get("txn_date")
    if txn_date is not None:
        try:
            parsed = date.fromisoformat(txn_date)
            if parsed.isoformat() != txn_date:
                errors.append("invalid_txn_date")
        except (ValueError, TypeError):
            errors.append("invalid_txn_date")

    txn_id = obj.get("txn_id")
    if txn_id is not None and not re.fullmatch(r"NP\d{10}", str(txn_id)):
        errors.append("invalid_txn_id")

    if obj.get("channel") is not None and obj["channel"] not in CHANNELS:
        errors.append("invalid_channel")

    if obj.get("language") not in {"en", "hi-en"}:
        errors.append("invalid_language")

    if type(obj.get("needs_human")) is not bool:
        errors.append("invalid_needs_human")

    return errors


def main(args):
    gold_inputs = read_jsonl(args.test)
    predictions = read_jsonl(args.predictions)

    expected_ids = [r["id"] for r in gold_inputs]
    predicted_ids = [r.get("id") for r in predictions]

    expected_set = set(expected_ids)
    predicted_set = set(predicted_ids)

    duplicates = sorted({
        x for x in predicted_ids if predicted_ids.count(x) > 1
    })

    missing = sorted(expected_set - predicted_set)
    unexpected = sorted(predicted_set - expected_set)

    schema_errors = []

    for row in predictions:
        errors = validate_output(row.get("output"))
        if errors:
            schema_errors.append({
                "id": row.get("id"),
                "errors": errors
            })

    summary = {
        "expected_count": len(expected_ids),
        "prediction_count": len(predictions),
        "unique_prediction_ids": len(predicted_set),
        "missing_ids": missing,
        "unexpected_ids": unexpected,
        "duplicate_ids": duplicates,
        "invalid_outputs": len(schema_errors),
        "schema_errors": schema_errors,
        "submission_structure_valid": (
            len(expected_ids) == 400
            and len(predictions) == 400
            and len(predicted_set) == 400
            and not missing
            and not unexpected
            and not duplicates
            and not schema_errors
        )
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print("\nValidation report:", report_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--report", required=True)
    main(parser.parse_args())
