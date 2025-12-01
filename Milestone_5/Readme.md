Milestone 5 – Visualization & Dashboard Development
Project: PrognosAI — Remaining Useful Life (RUL) Prediction
Author: Nithin G J
Date: November 28, 2025
📌 Objective

The objective of Milestone 5 is to build an interactive visualization dashboard to monitor Remaining Useful Life (RUL) trends, predictions, alerts, and model performance.
A Streamlit-based dashboard was developed to provide clear insights into equipment health, supporting real-time and proactive maintenance decisions.

📊 Summary of Implementation

A complete Streamlit Prognostics Dashboard was created with:

Model and scaler loading

Preprocessing of input sequences

Real-time RUL predictions

Interactive Plotly charts (line, scatter, summary table)

KPI cards for RMSE, MAE, R²

Automatic alert classification based on RUL thresholds

Clean, organized UI for easy interpretation

The dashboard integrates seamlessly with the trained model and offers visual clarity through dynamic interactions.

📁 Key Files & Paths
Component	Path
Trained Model	models_m2/optimized_fd1_milestone4.h5
Scaler File	models_m2/scaler_fd1_milestone4.save
Processed Test Data	processed/fd1_test_ws30.npz
Streamlit Dashboard Script	app_streamlit_prognosai.py
🎨 Dashboard UX & Visual Components

The dashboard layout includes:

A header section describing the project

KPI cards summarizing model evaluation metrics

A dual-line (Actual vs Predicted RUL) interactive Plotly chart

RUL-based alert scatter plot (Normal, Warning, Critical)

Recent alert table with conditional highlighting for quick review

The interface is designed to be intuitive and efficient for monitoring equipment health.

🚨 Alert Thresholds & Logic

RUL predictions are mapped to alert categories:

Normal: RUL > 70

Warning: 30 < RUL ≤ 70

Critical: RUL ≤ 30

Recommended improvements:

Adaptive thresholds

Historical calibration

Confidence-based filtering

▶️ Deployment & Instructions
1️⃣ Install Required Libraries
pip install streamlit plotly joblib tensorflow scikit-learn pandas numpy

2️⃣ Ensure Correct File Paths

Verify that:

Model path

Scaler path

Test data path
match the locations defined in the app script.

3️⃣ Run the Dashboard
streamlit run app_streamlit_prognosai.py

4️⃣ Open in Browser

Visit:

http://localhost:8501

⚡ Performance & Usability Tips

Use st.cache_resource to cache model and scaler loading

Precompute predictions where possible for smoother performance

Use Docker & Nginx for production deployment

🚀 Extensions & Next Steps

Future enhancements may include:

Engine-level drilldown and playback

Email/SMS alert notifications

User authentication and role-based access

Real-time inference through API endpoints
