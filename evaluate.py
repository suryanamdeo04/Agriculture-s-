import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Load dataset
data = pd.read_csv("data/crop_recommendation.csv")

X = data.drop("label", axis=1)
y = data["label"]

# Recreate the same test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Load trained model
with open("models/crop_model.pkl", "rb") as file:
    model = pickle.load(file)

# Predictions
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("AGRIVISION AI - MODEL EVALUATION")
print("--------------------------------")
print("Test Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Save classification report
report = classification_report(
    y_test, predictions, output_dict=True
)

pd.DataFrame(report).transpose().to_csv(
    "models/classification_report.csv"
)

# Confusion matrix
cm = confusion_matrix(
    y_test,
    predictions,
    labels=model.classes_
)

plt.figure(figsize=(15, 12))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.title("AgriVision AI - Confusion Matrix")
plt.xlabel("Predicted Crop")
plt.ylabel("Actual Crop")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()

plt.savefig("models/confusion_matrix.png", dpi=200)

print("\nEvaluation files saved in models folder!")
