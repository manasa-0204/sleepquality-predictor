from flask import Flask, render_template, request
from sleep import predict_sleep, accuracy

app = Flask(__name__)

@app.route("/")
def home():
    return render_template(
        "index.html",
        accuracy=round(accuracy * 100, 2)
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        sleep = request.form["sleep"]
        screen = request.form["screen"]
        caffeine = request.form["caffeine"]
        stress = request.form["stress"]

        result = predict_sleep(
            sleep,
            screen,
            caffeine,
            stress
        )

        return render_template(
            "index.html",
            prediction=result,
            accuracy=round(accuracy * 100, 2)
        )

    except (KeyError, ValueError) as e:
        return render_template(
            "index.html",
            error="Invalid input. Please select all options.",
            accuracy=round(accuracy * 100, 2)
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)