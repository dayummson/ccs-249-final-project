import sys

sys.path.append("../")

from utils.extractors.docx_extractor import extract_blocks_from_docx
from utils.extractors.pdf_extractor import extract_blocks_from_pdf

extracted_pdf = extract_blocks_from_pdf("../samples/final-project.pdf")
extracted_docx = extract_blocks_from_docx("../samples/unit-2-activity.docx")


def loop_inside(array):

    for something in array:
        if isinstance(something, list):

            loop_inside(something)

        else:

            print(something)


loop_inside(array=extracted_pdf)
loop_inside(array=extracted_docx)
