import pandas as pd
import random


def generate_infinite_admin_traps(target_count=1000):
    data = []

    papers = [
        "yellow pad",
        "A4 bond paper",
        "long brown folder",
        "intermediate pad",
        "1/4 sheet",
        "legal size",
        "short white folder",
    ]
    verbs = [
        "Submit",
        "Pass",
        "Upload",
        "Turn in",
        "Hand over",
        "Forward",
        "Attach",
        "Direct",
    ]
    platforms = [
        "LMS",
        "Google Classroom",
        "faculty room",
        "guard house",
        "Messenger",
        "Class President",
        "Admin Office",
    ]

    # Adding 'Noise' variables to ensure every sentence is mathematically unique
    room_numbers = [f"Room {n}" for n in range(101, 505)]
    times = [
        f"{h}:{m:02d} {ap}"
        for h in range(1, 13)
        for m in [0, 15, 30, 45]
        for ap in ["AM", "PM"]
    ]
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

    templates = [
        "Note: {verb} your output in a {paper} to {platform} at {room}.",
        "Strictly no late submissions. {verb} it before {time} on {day}.",
        "Use {paper} for calculations; otherwise, you will lose {pts} points.",
        "Staple the {paper} at the {corner} and bring it to {room}.",
        "Filenames must be: {surname}_{task}_{rand_id}.pdf",
        "Warning: Submissions in {fmt} will be failed by {time}.",
        "Please {verb} the {paper} to the {platform} by {day} {time}.",
        "Note: Only {color} ink is allowed for the {paper} submission.",
    ]

    while len(data) < target_count:
        tpl = random.choice(templates)
        sentence = tpl.format(
            verb=random.choice(verbs),
            paper=random.choice(papers),
            platform=random.choice(platforms),
            room=random.choice(room_numbers),
            time=random.choice(times),
            day=random.choice(days),
            pts=random.randint(5, 50),
            corner=random.choice(["top-left", "top-right", "bottom-left"]),
            surname=random.choice(["Dom", "Santiago", "Perez", "Cruz", "Reyes"]),
            task=random.choice(["ACT", "LAB", "PROJ", "QUIZ"]),
            rand_id=random.randint(100, 9999),
            fmt=random.choice([".rar", ".zip", ".png", ".docx"]),
            color=random.choice(["black", "blue"]),
        )
        data.append({"text": sentence, "label": "ADMIN_TRAP"})

        if len(data) % 100 == 0:
            temp_df = pd.DataFrame(data).drop_duplicates(subset=["text"])
            data = temp_df.to_dict("records")

    return pd.DataFrame(data).iloc[:target_count]


df_traps = generate_infinite_admin_traps(1000)
df_traps.to_csv("admin_trap.csv", index=False)
print(f"Final Count: {len(df_traps)} Unique Admin Traps.")
