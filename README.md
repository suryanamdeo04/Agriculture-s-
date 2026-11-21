# 🌿 AgriVision AI
### AI-Powered Smart Crop Recommendation System

AgriVision AI is a machine learning-based agricultural decision support system that recommends suitable crops based on soil nutrients and environmental conditions.

🌐 **Live Demo:** https://agrivision-ai-demo.streamlit.app/

## 🚀 Key Features

- **Smart Crop Recommendation:** Predicts suitable crops using soil and climate parameters.
- **Top-5 Predictions:** Displays the five most probable crop recommendations.
- **Crop Profiles:** Provides detailed crop information and growing conditions.
- **Dataset Insights:** Visualizes dataset statistics and crop distributions.
- **Model Performance:** Displays accuracy, precision, recall, F1-score and confusion matrix.
- **Interactive Dashboard:** Provides a user-friendly Streamlit interface.

## 🧠 Machine Learning

- Algorithm: Random Forest Classifier
- Dataset: Crop Recommendation Dataset
- Features: Nitrogen (N), Phosphorus (P), Potassium (K), Temperature, Humidity, pH and Rainfall
- Crop Classes: 22
- Evaluation Samples: 440
- Test Accuracy: **99.32%**

*Note: Reported performance is based on the saved evaluation report. Results depend on dataset quality and evaluation methodology.*

## 🛠️ Tech Stack

- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Matplotlib / Plotly

## 📂 Project Structure

```text
AgriVisionAI/
├── app.py
├── crop_profiles.py
├── data/
│   ├── crop_recommendation.csv
│   └── crop_recommendation_10k.csv
├── models/
│   ├── crop_model.pkl
│   ├── classification_report.csv
│   └── confusion_matrix.png
├── requirements.txt
└── README.md
```

## ⚙️ Installation & Setup

1. Clone the repository:

```bash
git clone https://github.com/suryanamdeo04/Agriculture-s-.git
cd Agriculture-s-
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the application:

```bash
streamlit run app.py
```

## 🎯 Project Objective

The objective of AgriVision AI is to demonstrate how machine learning can support data-driven agricultural decisions by analyzing soil and environmental parameters to recommend suitable crops.

## 👨‍💻 Developer

**Surya Namdeo**

B.Tech – Artificial Intelligence & Machine Learning

---

⭐ If you find this project useful, consider giving it a star!
