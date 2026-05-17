from transformers import pipeline

MODELS = {"TUNED": "../final_requirement_model", "RAW": "distilbert-base-uncased"}

# Label map only applies to YOUR trained model
# Raw model outputs generic LABEL_0, LABEL_1 etc. with no meaning
id_2_label = {
    "LABEL_0": "ADMIN_TRAP",
    "LABEL_1": "IGNORE",
    "LABEL_2": "OPTIONAL_BONUS",
    "LABEL_3": "PRE_REQUISITE",
    "LABEL_4": "TECHNICAL_TASK",
    "LABEL_5": "WEIGHTED_PRIORITY",
}

sentences = [
    "Build a Bank System using Python and implement abstraction, encapsulation and inheritance",
    "Note: Make sure to submit this before May 9, 2026 7:00PM",
    "College of Information and Communications Technology",
    "Make sure that the code you implement is yours, not AI Generated",
]

print("=" * 60)
print("RAW model (untrained baseline — expect garbage):")
print("=" * 60)
raw_pipe = pipeline("text-classification", model=MODELS["RAW"])
for s in sentences:
    r = raw_pipe(s)[0]
    print(f"  ⚠️  [{r['score']:.2f}] {r['label']:10s} → {s[:60]}")

print()
print("=" * 60)
print("TUNED model (your fine-tuned classifier):")
print("=" * 60)
tuned_pipe = pipeline("text-classification", model=MODELS["TUNED"])
for s in sentences:
    r = tuned_pipe(s)[0]
    label = id_2_label.get(r["label"], r["label"])
    flag = "✅" if r["score"] >= 0.75 else "⚠️"
    print(f"  {flag}  [{r['score']:.2f}] {label:20s} → {s[:60]}")
