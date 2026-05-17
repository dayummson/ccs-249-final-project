import fitz
import re


def extract_blocks_from_pdf(file_path):
    doc = fitz.open(file_path)
    raw_lines = []

    for page in doc:
        page_dict = page.get_text("dict")
        for block in page_dict["blocks"]:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                spans = line["spans"]
                if not spans:
                    continue

                text = " ".join(s["text"] for s in spans).strip()
                if not text:
                    continue

                avg_size = sum(s["size"] for s in spans) / len(spans)
                is_bold = any("Bold" in s["font"] for s in spans)
                x_origin = spans[0]["origin"][0]

                raw_lines.append(
                    {
                        "text": text,
                        "font_size": avg_size,
                        "is_bold": is_bold,
                        "x_origin": x_origin,
                        "is_heading": avg_size > 13 or is_bold,
                    }
                )

    # MERGE FRAGMENTED CONTINUATION LINES
    def is_new_block(prev, curr):
        if prev is None:
            return True

        prev_text = prev["text"]
        curr_text = curr["text"]

        if prev_text.endswith((".", ":", "?", "!")):
            return True

        if curr["is_heading"] and not prev["is_heading"]:
            return True

        if re.match(r"^(\d+\.|[a-zA-Z]\.|•|-|–|\*|[ivxIVX]+\.)", curr_text.strip()):
            return True

        if abs(curr["x_origin"] - prev["x_origin"]) > 20:
            return True

        return False

    merged = []
    current = None

    for line in raw_lines:
        if len(line["text"]) < 4:
            continue

        if is_new_block(current, line):
            if current:
                merged.append(current)
            current = dict(line)
        else:
            current["text"] = current["text"].rstrip() + " " + line["text"].lstrip()

    if current:
        merged.append(current)

    # FINAL CLEANUP
    blocks = []
    for b in merged:
        text = b["text"].strip()
        if len(text) < 12:
            continue

        is_list = bool(re.match(r"^(\d+\.|[a-zA-Z]\.|•|-|–|\*|[ivxIVX]+\.)", text))

        blocks.append(
            {
                "text": re.sub(r"\s+", " ", text),
                "is_heading": b["is_heading"],
                "is_list_item": is_list,
                "is_bold": b["is_bold"],
                "indent_level": max(0, b["x_origin"] - 72),
            }
        )

    return blocks
