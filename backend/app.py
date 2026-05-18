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
from flask import Flask, render_template
from dotenv import load_dotenv

load_dotenv(dotenv_path="env.local")

# Connecting flask to hmlt
# since we have our frontend ( ui )
# on a seperate folder we have to do this
backend_dir = os.path.dirname(os.path.abspath(__file__))

# Go up one level and into the frontend folder
frontend_dir = os.path.abspath(os.path.join(backend_dir, "..", "frontend"))


app = Flask(
    __name__,
    template_folder=frontend_dir,
    # the static folder is not prefixed
    # on the html ( index.html ) side, since we
    # already done it here
    static_folder=os.path.join(frontend_dir, "static"),
)

# we need to do this to force the reload everytime
# we did change on the index.html and tailwindcss
# but make sure to remove this on PROD
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

CORS(app)

UPLOAD_FOLDER = "/tmp/activitybrief"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def home():
    return render_template("index.html", name="User")


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
    from livereload import Server

    HOST = os.getenv("HOST")
    PORT = os.getenv("PORT")

    server = Server(app.wsgi_app)
    server.watch(frontend_dir)

    print(f"LiveReload server running on {HOST}:{PORT}")
    server.serve(host=HOST, port=PORT, debug=True)

    # app.run(debug=True, host="0.0.0.0", port=5000)
