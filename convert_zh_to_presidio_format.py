import json
import re
from pathlib import Path

SRC = Path("data/data_person_1000_zh.json")
OUT = Path("data/data_person_1000_zh_presidio_format.json")

FIELD_TO_ENTITY = {
    "name": "PERSON",
    "gender": "GENDER",
    "age": "AGE",
    "location": "LOCATION",
    "occupation": "PROFESSION",
    "idCardNumbers": "ID_NUMBER",
    "emailAddress": "EMAIL_ADDRESS",
    "phoneNumbers": "PHONE_NUMBER",
    "symptoms": "MEDICAL_SYMPTOM",
    "diagnosticOutcome": "DIAGNOSIS",
    "medicationDetails": "MEDICATION",
    "doctor": "DOCTOR",
    "transactionDetails": "TRANSACTION",
    "creditScore": "CREDIT_SCORE",
    "income": "INCOME",
}


def candidate_values(key, value):
    if value is None:
        return []

    candidates = []
    if isinstance(value, (int, float)):
        candidates.append(str(value))
        if key == "income":
            n = float(value)
            candidates.extend(
                [
                    f"{n:.2f}",
                    f"{n:,.2f}",
                    f"{n:.1f}",
                    f"{n:,.1f}",
                    f"{int(n)}",
                    f"{int(n):,}",
                ]
            )
    else:
        s = str(value).strip()
        if s:
            candidates.extend(
                [
                    s,
                    s.replace("/", " / "),
                    s.replace(" / ", "/"),
                    s.replace("，", ","),
                    s.replace(",", "，"),
                ]
            )

    deduped = []
    seen = set()
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            deduped.append(c)
    return deduped


def convert_record(record, template_id, miss_counter):
    text = record.get("naturalParagraph", "")
    used = [False] * len(text)
    spans = []

    for key, entity_type in FIELD_TO_ENTITY.items():
        raw_value = record.get(key)
        found = False

        for cand in sorted(candidate_values(key, raw_value), key=len, reverse=True):
            for match in re.finditer(re.escape(cand), text):
                start, end = match.start(), match.end()
                if start < end and not any(used[start:end]):
                    spans.append(
                        {
                            "entity_type": entity_type,
                            "entity_value": text[start:end],
                            "start_position": start,
                            "end_position": end,
                        }
                    )
                    for i in range(start, end):
                        used[i] = True
                    found = True

        if not found:
            miss_counter[key] += 1

    spans.sort(key=lambda x: (x["start_position"], x["end_position"]))

    masked_parts = []
    cursor = 0
    for span in spans:
        start = span["start_position"]
        end = span["end_position"]
        if start < cursor:
            continue
        masked_parts.append(text[cursor:start])
        masked_parts.append(f"{{{{{span['entity_type']}}}}}")
        cursor = end
    masked_parts.append(text[cursor:])

    return {
        "full_text": text,
        "masked": "".join(masked_parts),
        "spans": spans,
        "template_id": template_id,
        "metadata": None,
    }


def main():
    with SRC.open("r", encoding="utf-8") as f:
        source_data = json.load(f)

    miss_counter = {k: 0 for k in FIELD_TO_ENTITY}
    output_data = [
        convert_record(record, i + 1, miss_counter) for i, record in enumerate(source_data)
    ]

    with OUT.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    print(f"Converted {len(output_data)} records -> {OUT}")
    print("Missing field match counts:")
    for key, misses in miss_counter.items():
        print(f"  {key}: {misses}")


if __name__ == "__main__":
    main()
