import pickle
import pandas as pd

# Load trained model
with open("models/crop_model.pkl", "rb") as f:
    model = pickle.load(f)

def predict_crop(N, P, K, temperature, humidity, ph, rainfall):

    data = pd.DataFrame([[
        N, P, K, temperature,
        humidity, ph, rainfall
    ]], columns=[
        "N", "P", "K", "temperature",
        "humidity", "ph", "rainfall"
    ])

    prediction = model.predict(data)[0]

    probabilities = model.predict_proba(data)[0]
    confidence = max(probabilities) * 100

    return prediction, confidence


if __name__ == "__main__":
    crop, confidence = predict_crop(
        90, 42, 43, 20.8, 82, 6.5, 202
    )

    print("Recommended Crop:", crop)
    print("Model Confidence:", round(confidence, 2), "%")
