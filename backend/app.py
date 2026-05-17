import sys

sys.path.append("../")

import os
import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS
from utils.validators.validator import validate_upload
from utils.extractors.docx_extractor import extract_blocks_from_docx
from utils.extractors.pdf_extractor import extract_blocks_from_pdf
from services.classifier import classify_blocks, group_to_briefing

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "/tmp/activitybrief"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/health", methods=["GET"])
def health():
    """Quick check that the server is running."""
    return jsonify({"status": "ok"})


@app.route("/analyze", methods=["POST"])
def analyze():
    # FIRST: we check if the file is present
    if "file" not in request.files:
        return jsonify({"error": "No file provided. Send a .docx or .pdf."}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename."}), 400

    # NEXT: Save to temp path with unique name (avoids collisions)
    ext = file.filename.rsplit(".", 1)[-1].lower()
    tmp_name = f"{uuid.uuid4().hex}.{ext}"
    tmp_path = os.path.join(UPLOAD_FOLDER, tmp_name)
    file.save(tmp_path)

    try:
        # Next: Validate the file (type, size, etc.)
        is_valid, error, warnings = validate_upload(tmp_path, file.filename)
        if not is_valid:
            return jsonify({"error": error}), 422

        # Next: Extract blocks
        if ext == "pdf":
            blocks = extract_blocks_from_pdf(tmp_path)
        else:
            blocks = extract_blocks_from_docx(tmp_path)

        if not blocks:
            return (
                jsonify(
                    {"error": "No meaningful text could be extracted from this file."}
                ),
                422,
            )

        # Next: Classify blocks
        classified = classify_blocks(blocks)
        briefing = group_to_briefing(classified)

        # Next: Calculate quality metrics for the frontend
        total = len(classified)
        low_conf = sum(1 for b in classified if b["source"] == "low_confidence")
        heuristic = sum(1 for b in classified if b["source"] == "heuristic")
        model_hit = sum(1 for b in classified if b["source"] == "model")

        quality = "good" if (low_conf / max(total, 1)) < 0.3 else "degraded"

        return jsonify(
            {
                "briefing": briefing,
                "warnings": warnings,
                "quality": quality,
                "stats": {
                    "total_blocks": total,
                    "model_hits": model_hit,
                    "heuristic_hits": heuristic,
                    "low_confidence": low_conf,
                },
            }
        )

    finally:
        # Always clean up the temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
