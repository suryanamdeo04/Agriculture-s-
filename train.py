import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
)

# Load 10K dataset
data = pd.read_csv("data/crop_recommendation_10k.csv")

# Separate original real and synthetic records
real = data[data["data_source"] == "real"].copy()
synthetic = data[data["data_source"] == "synthetic"].copy()

# Hold out 20% of original real data for honest testing
real_train, real_test = train_test_split(
    real,
    test_size=0.2,
    random_state=42,
    stratify=real["label"]
)

# Train on real training data + synthetic data
train_data = pd.concat([real_train, synthetic], ignore_index=True)

X_train = train_data.drop(columns=["label", "data_source"])
y_train = train_data["label"]

X_test = real_test.drop(columns=["label", "data_source"])
y_test = real_test["label"]

# Train Random Forest
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Test only on original unseen real records
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# Create models folder
os.makedirs("models", exist_ok=True)

# Save trained model
with open("models/crop_model.pkl", "wb") as file:
    pickle.dump(model, file)

# Save classification report
report = classification_report(
    y_test, predictions, output_dict=True, zero_division=0
)
pd.DataFrame(report).transpose().to_csv(
    "models/classification_report.csv"
)

# Save confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions,
    labels=model.classes_,
    xticks_rotation=90,
    cmap="Greens",
    colorbar=False
)
plt.title("AgriVision AI - Confusion Matrix")
plt.tight_layout()
plt.savefig("models/confusion_matrix.png", dpi=250)
plt.close()

# Results
print("\n========== AGRIVISION AI TRAINING ==========")
print("Total dataset records:", len(data))
print("Real training records:", len(real_train))
print("Synthetic training records:", len(synthetic))
print("Total training records:", len(train_data))
print("Real testing records:", len(real_test))
print("--------------------------------------------")
print("Real-data test accuracy:", round(accuracy * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))
print("============================================")
print("Model saved: models/crop_model.pkl")
print("Report saved: models/classification_report.csv")
print("Confusion matrix saved: models/confusion_matrix.png")
