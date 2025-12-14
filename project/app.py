from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
model = joblib.load("diabetes_model.pkl")

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/submit", methods=["POST"])
def submit():
    fields = ["preg","plas","pres","skin","test","mass","pedi","age"]
    values = {f: request.form.get(f) for f in fields}

    features = [float(values[f]) for f in fields]
    prediction = model.predict([features])
    result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"

    return render_template("home.html", result=result, values=values)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
