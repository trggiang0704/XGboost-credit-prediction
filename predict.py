import joblib
import numpy as np
import pandas as pd
import os

OUT_DIR = "output_model"

# Load artifacts
model = joblib.load(os.path.join(OUT_DIR, "xgb_calibrated_model.joblib"))
preprocessor = joblib.load(os.path.join(OUT_DIR, "preprocessor.joblib"))
feature_names = pd.read_csv(os.path.join(OUT_DIR, "feature_names.csv")).values.flatten().tolist()

def predict_single(input_dict):
    """
    input_dict: dict chứa feature -> value
    """
    x_df = pd.DataFrame([input_dict], columns=feature_names)
    x_proc = preprocessor.transform(x_df)
    prob = model.predict_proba(x_proc)[0, 1]
    label = int(prob >= 0.5)
    return {"probability": float(prob), "prediction": label}
