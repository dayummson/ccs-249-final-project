from transformers import pipeline, AutoConfig

# Load the config to see the label mapping
config = AutoConfig.from_pretrained("./final_requirement_model")

# If the config doesn't have the names yet, we map them manually
# Replace these with the order they appeared in your 'wvsu_robust_requirements.csv'
id2label = {
    0: "ADMIN_TRAP",
    1: "IGNORE",
    2: "OPTIONAL_BONUS",
    3: "PRE_REQUISITE",
    4: "TECHNICAL_TASK",
    5: "WEIGHTED_PRIORITY",
}

analyzer = pipeline("text-classification", model="./final_requirement_model")

test_text = (
    "Students must submit their source code via a GitHub repository link by Friday 5PM."
)

result = analyzer(test_text)[0]

# Map the result
readable_label = id2label[int(result["label"].split("_")[1])]
print(f"Prediction: {readable_label} (Confidence: {result['score']:.2%})")
