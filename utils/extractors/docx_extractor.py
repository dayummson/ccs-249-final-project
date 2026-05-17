import docx


def extract_blocks_from_docx(file_path):
    """Returns list of dicts with text + structural metadata."""
    doc = docx.Document(file_path)
    blocks = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if len(text) < 10:
            continue

        style = para.style.name.lower()
        # Detect numbering/bullet from XML
        is_list = (
            para._element.find(
                ".//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}numPr"
            )
            is not None
        )

        blocks.append(
            {
                "text": text,
                "is_heading": "heading" in style,
                "heading_level": (
                    int(style.replace("heading ", "")) if "heading" in style else 0
                ),
                "is_list_item": is_list,
                "is_bold": any(run.bold for run in para.runs if run.text.strip()),
                "indent_level": para.paragraph_format.left_indent or 0,
            }
        )

    return blocks


# some random utils
# import pymupdf
# def loop_inside(array):

#     for something in array:
#         if isinstance(something, list):

#             loop_inside(something)

#         else:

#             print(something)
