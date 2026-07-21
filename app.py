

from pathlib import Path


import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

from utils import FORM_FIELD_MAP

# ---------------------------------------------------------------------------
# App Setup
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PIPELINE_PATH = BASE_DIR / "pipeline.pkl"

app = Flask(__name__)
pipeline = joblib.load(PIPELINE_PATH)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def home():
    """Render the prediction form."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Read form inputs, build a single-row DataFrame with raw feature values,
    pass it through the pipeline, and display the prediction result.
    """
    # Build a dict of {DataFrame_column_name: typed_value} from the form
    row = {}
    for form_field, (col_name, dtype) in FORM_FIELD_MAP.items():
        raw_value = request.form.get(form_field)
        if raw_value is not None:
            row[col_name] = dtype(raw_value)

    # Create a single-row DataFrame matching the training data format
    df = pd.DataFrame([row])

    # The pipeline handles ALL preprocessing internally
    # Get binary prediction and probability
    prediction = pipeline.predict(df)[0]
    
    try:
        # Get probability of the predicted class
        proba = pipeline.predict_proba(df)[0]
        confidence = proba[prediction]
        confidence_text = f" (Confidence: {confidence:.1%})"
    except (AttributeError, IndexError):
        confidence_text = ""

    if prediction == 0:
        result = "Employee Might Not Leave The Job" + confidence_text
    else:
        result = "Employee Might Leave The Job" + confidence_text

    return render_template("index.html", prediction_text=result)


if __name__ == "__main__":
    app.run(debug=True)