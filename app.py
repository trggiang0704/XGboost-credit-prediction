from flask import Flask, render_template, request
from predict import predict_single
import pandas as pd
import os

app = Flask(__name__)

# -----------------------
# Load feature names
# -----------------------
OUT_DIR = "output_model"
feature_names = pd.read_csv(
    os.path.join(OUT_DIR, "feature_names.csv")
).values.flatten().tolist()

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        form_data = {k: float(v) for k, v in request.form.items()}
        result = predict_single(form_data)
    return render_template("index.html", result=result, feature_names=feature_names)

if __name__ == "__main__":
    app.run(debug=True)
