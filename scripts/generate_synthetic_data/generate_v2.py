import random
import csv
import re

TECHNICAL_TASK_FRAMES = [
    "Implement {thing} using {lib}.",
    "Write a {thing} that handles {edge_case}.",
    "Build {thing} from scratch without using {lib}.",
    "Your objective is to develop a {thing} within the {lib} environment.",
    "The goal of this task is to construct a working {thing}.",
    "You are expected to produce a {thing} that correctly {action}s the {data}.",
    "A {thing} must be implemented using {lib} for this exercise.",
    "Using the features provided by {lib}, bootstrap a {thing} system.",
    "{thing} development is required for the following subtask.",
    "For this part, go ahead and code up a {thing}.",
    "Try to get the {thing} working using {lib} — test it on the sample data.",
    "Your task here is simple: make a {thing} that {action}s correctly.",
    "Create a {thing} using {lib} — this is worth {n} points.",
    "Develop a {thing} ({n} pts) and submit it as a {filetype}.",
    "Build a {thing} that correctly classifies {data}. ({n} points)",
]

WEIGHTED_PRIORITY_FRAMES = [
    "({n} points) {task}",
    "({n} pts) {task}",
    "{task} — worth {n} points total.",
    "{task} This item is graded out of {n}.",
    "{task} [{n} pts]",
    "{task} ({n} points)",
    "{task} — {n}/{total} points.",
    "{criterion}: {n} points",
    "{criterion} — {n} out of {total}",
    "Rubric: {criterion} ({n} points)",
    "Total score: {n} out of {total} points.",
    "This section accounts for {n} points of your final grade.",
    "Grading: {criterion} is worth {n} points.",
    "({n} points each)",
    "({n} points per item)",
    "— {n} pts",
]

ADMIN_TRAP_FRAMES = [
    "Note: Save your {artifact} as a {filetype} with the proper naming convention.",
    "Note: Submit your work to {platform} before {date}.",
    "Note: This is not a coding activity. Put your answers in the answer sheet.",
    "Note: Use this file as your answer sheet.",
    "Note: Write your answers on a one whole sheet of paper.",
    "Note: Note: Save your {artifact} as a {filetype} and upload to {platform}.",
    "Upload your {artifact} to your assigned folder in the GitHub organization.",
    "Submit via {platform} by {date}.",
    "Submit your {artifact} as a {filetype} following the naming convention.",
    "An assignment on the LMS will be created for submitting the necessary files.",
    "The team leaders will fork the repository and collaborate with their members.",
    "Once done, create a PR to merge your changes to the main repository.",
    "A GitHub repository will be created within our organization.",
    "Name your file following the convention: {filetype}.",
    "Make sure your {artifact} follows the naming convention on the README.",
    "Make sure that the code you submit is your own work.",
    "Plagiarism and AI-generated submissions are not allowed.",
    "Name: _______ Date: _______ Year and Section: _______",
    "Name:",
    "Date:",
    "Year and Section:",
    "A Self and Peer Assessment template will be distributed after your presentation.",
    "Peer evaluation scores will factor into your final project grade.",
]

OPTIONAL_BONUS_FRAMES = [
    "Bonus: {task} for {n} extra points.",
    "BONUS! ({n} points) {task}",
    "Optional: {task} for additional credit.",
    "Extra credit: {task} (+{n} pts)",
    "Challenge: {task} — bonus points awarded.",
    "For extra credit, {task}.",
    "This item is optional: {task}",
    "If you finish early, attempt this bonus task: {task}",
    "({n} points, optional) {task}",
]

PRE_REQUISITE_FRAMES = [
    "Given a small dataset of documents with class labels, do the following tasks.",
    "Given the corpus below, perform the required preprocessing steps.",
    "Given the following sentence, determine its class.",
    "Using the provided dataset, complete the following:",
    "Based on the discussions in the provided resource materials, answer the following.",
    "Refer to the sample output on the repository before starting.",
    "Using the sample source codes shared on the LMS, update it to your needs.",
    "Before starting, make sure {tool} is installed in your environment.",
    "Download {tool} into your local environment before running the script.",
    "Clone the repository and install all dependencies first.",
    "Ensure your virtual environment is activated before proceeding.",
    "Take note of the following configuration before training:",
    "Make sure to follow the steps outlined below →",
    "The article should be at least a few thousand words for good results.",
    "Select a Wikipedia article of your choice as the corpus for this exercise.",
]

IGNORE_FRAMES = [
    "Exercise for Unit {n}",
    "Activity for Unit {n}",
    "Activity for Unit {n} Part {part}",
    "Pre-Activity for Unit {n}",
    "Assignment for Unit {n}",
    "**Exercise for Unit {n}**",
    "**Activity for Unit {n} Part {part}**",
    "**Assignment for Unit {n}**",
    "Lab {n}: {topic}",
    "Unit {n} — {topic}",
    "Exercise for Unit {n}.{sub} {topic}",
    "West Visayas State University",
    "College of Information and Communications Technology",
    "CCS {code} — {semester}",
    "Luna St., La Paz, Iloilo City 5000",
    # URLs and contact
    "https://github.com/{org}/{repo}",
    "www.wvsu.edu.ph",
    "cict@wvsu.edu.ph",
    "Trunkline: (063) (033) 320-0870 loc 1403",
    "Page {n} of {total}",
    "— {n} —",
    "Answer:",
    "Article Name:",
    "Article Link:",
    "Paste the graph below.",
    "Once done, paste your results below:",
    "PCA:",
    "Output:",
    "**Documents**",
    "**Query:** {topic}",
    "References:",
]


SLOTS = {
    "thing": [
        "Naive Bayes classifier",
        "TF-IDF vectorizer",
        "LSTM model",
        "tokenizer",
        "stemmer",
        "sentiment analyzer",
        "NER tagger",
        "word embedding model",
        "logistic regression classifier",
        "bag-of-words model",
        "skip-gram model",
        "decision tree",
        "text preprocessing pipeline",
        "cosine similarity function",
        "confusion matrix generator",
        "evaluation script",
    ],
    "lib": [
        "scikit-learn",
        "NLTK",
        "spaCy",
        "HuggingFace Transformers",
        "PyTorch",
        "Keras",
        "Gensim",
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "tensorflow",
    ],
    "edge_case": [
        "empty strings",
        "unseen tokens",
        "special characters",
        "long sequences",
        "mixed language input",
        "duplicate entries",
        "missing values",
        "out-of-vocabulary words",
    ],
    "action": [
        "tokenizes",
        "classifies",
        "preprocesses",
        "vectorizes",
        "embeds",
        "evaluates",
        "normalizes",
        "filters",
        "extracts",
    ],
    "data": [
        "corpus",
        "dataset",
        "vocabulary",
        "feature matrix",
        "training set",
        "test sentences",
        "document collection",
    ],
    "artifact": [
        "Python source code",
        "Jupyter notebook",
        "script",
        "output file",
        "answer sheet",
        "submission",
    ],
    "filetype": [
        ".ipynb file",
        ".py file",
        ".zip archive",
        ".pdf file",
        "a single .ipynb OR multiple .py files",
    ],
    "platform": [
        "the LMS",
        "GitHub",
        "your assigned folder",
        "the Google Classroom",
        "Canvas",
    ],
    "date": [
        "Friday 11:59 PM",
        "next Monday",
        "the end of the week",
        "May 9, 2026 7:00 PM",
        "before the next session",
    ],
    "task": [
        "implement input validation",
        "add a confusion matrix visualization",
        "extend the classifier to handle multi-label inputs",
        "deploy the Flask app using Docker",
        "write unit tests for all preprocessing functions",
    ],
    "criterion": [
        "code correctness",
        "model accuracy",
        "documentation quality",
        "preprocessing implementation",
        "evaluation metrics",
        "code readability",
        "output formatting",
    ],
    "n": [str(i) for i in [5, 10, 15, 20, 25, 30, 50]],
    "total": ["50", "100", "25", "30"],
    "tool": [
        "Python 3.10",
        "NLTK",
        "Gensim",
        "Flask",
        "Git",
        "the required packages",
        "your virtual environment",
    ],
    "topic": [
        "Naive Bayes",
        "Logistic Regression",
        "Word Embeddings",
        "Text Classification",
        "NER",
        "Sentiment Analysis",
        "Language Models",
        "Text Preprocessing",
    ],
    "org": ["CS-3RD-YEAR-25-26", "wvsu-cict", "your-org"],
    "repo": ["CCS-249-Sample-Codes", "final-project", "activity-repo"],
    "code": ["249", "235", "221"],
    "semester": ["1st Semester AY 2025-2026", "2nd Semester AY 2024-2025"],
    "part": ["1", "2", "3"],
    "sub": ["1", "2", "3"],
}


def fill(template):
    result = template
    for slot in re.findall(r"\{(\w+)\}", template):
        if slot in SLOTS:
            result = result.replace(f"{{{slot}}}", random.choice(SLOTS[slot]), 1)
    return result


FRAME_MAP = {
    "TECHNICAL_TASK": TECHNICAL_TASK_FRAMES,
    "WEIGHTED_PRIORITY": WEIGHTED_PRIORITY_FRAMES,
    "ADMIN_TRAP": ADMIN_TRAP_FRAMES,
    "OPTIONAL_BONUS": OPTIONAL_BONUS_FRAMES,
    "PRE_REQUISITE": PRE_REQUISITE_FRAMES,
    "IGNORE": IGNORE_FRAMES,
}

CLASS_COUNTS = {
    "TECHNICAL_TASK": 500,
    "ADMIN_TRAP": 400,
    "IGNORE": 400,
    "WEIGHTED_PRIORITY": 300,
    "PRE_REQUISITE": 250,
    "OPTIONAL_BONUS": 150,
}

rows = []
for label, count in CLASS_COUNTS.items():
    frames = FRAME_MAP[label]
    for _ in range(count):
        tmpl = random.choice(frames)
        rows.append({"text": fill(tmpl), "label": label})

random.shuffle(rows)

with open("synthetic_data_v2.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["text", "label"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Generated {len(rows)} rows.")
print("Distribution:")
from collections import Counter

counts = Counter(r["label"] for r in rows)
for label, n in sorted(counts.items()):
    print(f"  {label:20s}: {n}")
