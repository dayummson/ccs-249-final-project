import os
import fitz
import docx

MAX_FILE_SIZE_MB = 5
MAX_PAGES = 20


def validate_upload(file_path, filename):
    """
    Returns (is_valid, error_message, warnings)
    """
    warnings = []
    ext = filename.lower().rsplit(".", 1)[-1]

    # FILE TYPE
    if ext not in ("pdf", "docx"):
        return False, f"Unsupported file type '.{ext}'. Upload a .docx or .pdf.", []

    # File SIZE
    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        return (
            False,
            f"File too large ({size_mb:.1f}MB). Max is {MAX_FILE_SIZE_MB}MB.",
            [],
        )

    # PDF CHECKS
    if ext == "pdf":
        try:
            doc = fitz.open(file_path)
        except Exception:
            return False, "Could not open PDF. File may be corrupted.", []

        if doc.is_encrypted:
            return False, "Password-protected PDFs are not supported.", []

        if len(doc) > MAX_PAGES:
            warnings.append(f"Only the first {MAX_PAGES} pages will be analyzed.")

        total_text = "".join(page.get_text().strip() for page in doc)
        if len(total_text) < 50:
            return (
                False,
                (
                    "This looks like a scanned or image-based PDF. "
                    "Please upload the original typed document."
                ),
                [],
            )

        lines = [l for l in total_text.split("\n") if l.strip()]
        avg_words = sum(len(l.split()) for l in lines) / max(len(lines), 1)
        if avg_words < 4:
            warnings.append(
                "Complex layout detected (multi-column or heavy formatting). "
                "Some blocks may be extracted incorrectly."
            )

        return True, None, warnings

    # DOCX CHECKS
    if ext == "docx":
        try:
            doc = docx.Document(file_path)
        except Exception:
            return False, "Could not open DOCX. File may be corrupted.", []

        text = " ".join(p.text.strip() for p in doc.paragraphs if p.text.strip())
        if len(text) < 50:
            return False, "Document appears empty or contains only images.", []

        return True, None, warnings
