import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# print('Import done')
url = r"https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

df = pd.read_csv(url)

# print(df.head())

X = df.iloc[:, :-1]     # all rows, all columns except last (features)
y = df.iloc[:, -1]      # all rows, last column (target)
print(y.head())

# ---------------------------
# Train–test split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# Model training
# ---------------------------
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# ---------------------------
# Prediction and accuracy
# ---------------------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("Accuracy:", acc)

# ---------------------------
# Save the model using joblib
# ---------------------------
joblib.dump(model, "project/diabetes_model.pkl")

print("Model saved as diabetes_model.pkl")

