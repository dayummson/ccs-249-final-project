# generate_data.py  —  run once, save to datasets/synthetic_data.csv
import random
import csv

TEMPLATES = {
    "TECHNICAL_TASK": [
        "Implement {thing} using {lib} and ensure it handles {edge_case}.",
        "Write a function that {action} the {data_structure}.",
        "Create a {artifact} that accepts {input} and returns {output}.",
        "Extract {feature} from the corpus using {technique}.",
        "Modify the {component} to support {requirement}.",
        "Code a {task} and test it with the provided {dataset}.",
        "Using {lib}, train a model that classifies {target}.",
        "Submit your {artifact} as a {filetype} with proper naming convention.",
    ],
    "WEIGHTED_PRIORITY": [
        "This section is worth {n} points.",
        "{task} — {n} pts",
        "Rubric: {criterion} ({n}/{total} points)",
        "The {component} will be graded out of {n} points.",
        "Total: {n} points. Partial credit given.",
        "{deliverable} — {n}/{total}",
    ],
    "ADMIN_TRAP": [
        "Write your name and section on the document.",
        "Submit via the LMS by {date}.",
        "Name: ____  Date: ____  Section: ____",
        "Email your submission to {email}.",
        "Late submissions will not be accepted after {date}.",
        "Meeting at {time} with {person}.",
        "Attendance is required for the {event}.",
    ],
    "OPTIONAL_BONUS": [
        "Bonus: {task} for {n} extra points.",
        "Optional: Implement {feature} for additional credit.",
        "Extra credit: {task} (+{n} pts)",
        "Challenge: {task} — bonus points awarded.",
    ],
    "PRE_REQUISITE": [
        "Before starting, make sure {condition} is set up.",
        "You should have completed {prior_task} before attempting this.",
        "Prerequisite: Install {tool} and configure {config}.",
        "This activity assumes knowledge of {concept}.",
    ],
    "IGNORE": [
        "Page {n} of {total}",
        "© {year} West Visayas State University",
        "CCS {code} — {semester}",
        "This document is confidential.",
        "For inquiries, contact {email}",
        "{university} College of {college}",
    ],
}

SLOT_VALUES = {
    "thing": [
        "Naive Bayes classifier",
        "TF-IDF vectorizer",
        "LSTM model",
        "tokenizer",
        "stemmer",
        "sentiment analyzer",
        "NER tagger",
    ],
    "lib": [
        "scikit-learn",
        "NLTK",
        "spaCy",
        "HuggingFace Transformers",
        "PyTorch",
        "Keras",
        "Gensim",
    ],
    "edge_case": [
        "empty strings",
        "unseen tokens",
        "special characters",
        "long sequences",
    ],
    "action": ["tokenizes", "classifies", "preprocesses", "vectorizes", "embeds"],
    "data_structure": ["corpus", "dataset", "vocabulary", "feature matrix"],
    "artifact": ["Jupyter notebook", "Python script", "Flask endpoint", "REST API"],
    "input": ["raw text", "user query", "CSV file", "JSON payload"],
    "output": ["labeled class", "probability distribution", "embedding vector"],
    "feature": [
        "n-grams",
        "POS tags",
        "named entities",
        "TF-IDF scores",
        "word embeddings",
    ],
    "technique": ["bag of words", "TF-IDF", "word2vec", "BERT embeddings"],
    "component": ["preprocessing pipeline", "model", "tokenizer", "classifier"],
    "requirement": [
        "multi-label classification",
        "batch inference",
        "GPU acceleration",
    ],
    "task": ["text classification", "NER pipeline", "sentiment analysis module"],
    "dataset": ["provided training data", "sample corpus", "activity dataset"],
    "target": ["spam vs ham", "sentiment", "intent", "named entities"],
    "filetype": [".ipynb file", ".py file", ".zip archive"],
    "n": [str(i) for i in range(5, 30, 5)],
    "total": ["50", "100", "25"],
    "criterion": ["model accuracy", "code quality", "documentation", "presentation"],
    "deliverable": ["GitHub submission", "report", "source code"],
    "component": ["backend", "model", "frontend"],
    "date": ["Friday 11:59 PM", "next Monday", "the end of the week"],
    "email": ["cict@wvsu.edu.ph", "instructor@example.edu"],
    "time": ["2 PM", "9 AM", "3:30 PM"],
    "person": ["the instructor", "your group leader", "the department head"],
    "event": ["lab session", "final presentation", "peer evaluation"],
    "prior_task": ["Activity 1", "Unit 5", "the preprocessing lab"],
    "tool": ["Python 3.10", "Node.js", "Flask", "Git"],
    "config": ["your virtual environment", "your GitHub SSH key", ".env file"],
    "concept": ["linear algebra", "probability", "Python OOP", "REST APIs"],
    "condition": [
        "Python is installed",
        "the repository is cloned",
        "the dataset is downloaded",
    ],
    "year": ["2024", "2025"],
    "code": ["249", "235", "221"],
    "semester": ["1st Semester AY 2024-2025", "2nd Semester"],
    "university": ["West Visayas State University", "WVSU"],
    "college": ["Information and Communications Technology", "Computing"],
}


def fill(template):
    import re

    result = template
    for slot in re.findall(r"\{(\w+)\}", template):
        if slot in SLOT_VALUES:
            result = result.replace(f"{{{slot}}}", random.choice(SLOT_VALUES[slot]), 1)
    return result


rows = []
for label, templates in TEMPLATES.items():
    for _ in range(300):  # 300 per class = 1800 total
        tmpl = random.choice(templates)
        rows.append({"text": fill(tmpl), "label": label})

random.shuffle(rows)

with open("synthetic_data.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["text", "label"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Generated {len(rows)} rows.")
