import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "sleep_data.csv")

df = pd.read_csv(csv_path)

# Convert text values into numbers
df["sleep"] = df["sleep"].map({
    "Poor": 0,
    "Average": 1,
    "Good": 2
})

df["screen"] = df["screen"].map({
    "Low": 0,
    "Medium": 1,
    "High": 2
})

df["caffeine"] = df["caffeine"].map({
    "None": 0,
    "Low": 1,
    "Medium": 2,
    "High": 3
})

df["stress"] = df["stress"].map({
    "Low": 0,
    "Medium": 1,
    "High": 2
})

quality_mapping = {
    "Very Poor": 0,
    "Poor": 1,
    "Medium": 2,
    "Good": 3,
    "Very Good": 4,
    "Excellent": 5
}

df["sleep_quality"] = df["sleep_quality"].map(quality_mapping)

df = df.dropna()

# Features and target
X = df[["sleep", "screen", "caffeine", "stress"]]
y = df["sleep_quality"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Calculate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy * 100, "%")


def predict_sleep(sleep, screen, caffeine, stress):

    input_data = pd.DataFrame([{
        "sleep": {
            "Poor": 0,
            "Average": 1,
            "Good": 2
        }[sleep],

        "screen": {
            "Low": 0,
            "Medium": 1,
            "High": 2
        }[screen],

        "caffeine": {
            "None": 0,
            "Low": 1,
            "Medium": 2,
            "High": 3
        }[caffeine],

        "stress": {
            "Low": 0,
            "Medium": 1,
            "High": 2
        }[stress]
    }])

    prediction = model.predict(input_data)[0]

    reverse_mapping = {
        0: "Very Poor",
        1: "Poor",
        2: "Medium",
        3: "Good",
        4: "Very Good",
        5: "Excellent"
    }

    return reverse_mapping[prediction]