from flask import Flask, render_template, request
from predict import predict_single

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        form_data = {k: float(v) for k, v in request.form.items()}
        result = predict_single(form_data)
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)