from flask import Flask, render_template
import json

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/model')
def model():
    with open("model_performance.json", "r") as f:
        metrics = json.load(f)
    return render_template("model.html", metrics=metrics)

@app.route('/insights')
def insights():
    return render_template("insights.html")

@app.route('/limitations')
def limitations():
    return render_template("limitations.html")

if __name__ == '__main__':
    app.run(debug=True)
