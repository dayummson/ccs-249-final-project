from transformers import pipeline

pipe = pipeline("text-classification", model="./final_requirement_model")

# Test with sentences from an actual activity sheet, not synthetic data
test_sentences = [
    "Implement a Naive Bayes classifier using scikit-learn.",
    "Name: _______  Date: _______  Section: _______",
    "This item is worth 20 points.",
    "Make sure Python is installed before starting.",
    "Bonus: Add input validation for +5 extra points.",
    "Page 1 of 3",
]

for sentence in test_sentences:
    result = pipe(sentence)[0]
    print(f"[{result['score']:.2f}] {result['label']:20s} → {sentence[:60]}")
