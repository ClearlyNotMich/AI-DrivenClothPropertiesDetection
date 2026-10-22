"""Local fabric metadata viewer; this demo does not run a trained model."""

import csv
from pathlib import Path

from flask import Flask, jsonify, render_template, send_file

CODE_DIR = Path(__file__).resolve().parent
DATA_PATH = CODE_DIR / "P2_Fabric_data_csv.csv"

app = Flask(__name__, template_folder=str(CODE_DIR / "templates"))

@app.get("/")
def index():
    return render_template("interface.html")


@app.get("/api/fabrics")
def fabrics():
    """Return the existing CSV, retaining quoted values and their field names."""
    try:
        # Match the encoding used by the historical research notebooks.
        with DATA_PATH.open(encoding="ISO-8859-1", newline="") as source:
            reader = csv.DictReader(source, skipinitialspace=True)
            fields = [field.strip() for field in reader.fieldnames or []]
            if "General Identifier" not in fields:
                raise ValueError("Missing fabric identifier column")
            records = {}
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError("Inconsistent CSV columns")
                record = {key.strip(): value.strip() for key, value in row.items()}
                records[record["General Identifier"]] = record
    except (OSError, csv.Error, ValueError):
        app.logger.exception("Could not load fabric metadata")
        return jsonify(error="Fabric metadata is unavailable. Please try again later."), 503

    return jsonify(fields=fields, fabrics=records)


@app.get("/placeholder.jpg")
def placeholder():
    return send_file(CODE_DIR / "templates" / "placeholder.jpg", mimetype="image/jpeg")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
