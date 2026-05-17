import pymupdf


def extract_blocks_from_pdf(file_path):
    """PDF extraction using PyMuPDF — preserves font size, bold, bullets."""
    doc = pymupdf.open(file_path)
    blocks = []

    for page in doc:
        page_dict = page.get_text("dict")
        for block in page_dict["blocks"]:
            if block["type"] != 0:  # 0 = text block
                continue
            for line in block["lines"]:
                spans = line["spans"]
                if not spans:
                    continue

                text = " ".join(s["text"] for s in spans).strip()
                if len(text) < 10:
                    continue

                avg_size = sum(s["size"] for s in spans) / len(spans)
                is_bold = any("Bold" in s["font"] for s in spans)
                x_origin = spans[0]["origin"][0]

                # Rough bullet detection
                is_list_item = text.startswith(("•", "-", "–", "*")) or (
                    len(text) > 2 and text[0].isdigit() and text[1] in ".)"
                )

                blocks.append(
                    {
                        "text": text.lstrip("•-–* "),
                        "is_heading": avg_size > 13,
                        "heading_level": (
                            1 if avg_size > 16 else (2 if avg_size > 13 else 0)
                        ),
                        "is_list_item": is_list_item,
                        "is_bold": is_bold,
                        "indent_level": max(
                            0, x_origin - 72
                        ),  # 72pt = 1 inch left margin
                    }
                )

    return blocks
