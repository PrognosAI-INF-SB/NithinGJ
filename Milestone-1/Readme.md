Nithin G J-Milestone1
PrognosAI: AI-Driven Predictive Maintenance System
📌 Project Objective
To design and develop an AI-based predictive maintenance system capable of estimating the Remaining Useful Life (RUL) of industrial machinery using multivariate time-series sensor data.
The system uses the NASA CMAPSS dataset and is designed to generalize across domains like turbines, pumps, and motors.
This will help in:

Enabling timely maintenance decisions
Minimizing unplanned downtime
Optimizing asset utilization
⚙️ Project Workflow
Data Ingestion

Load and preprocess the CMAPSS dataset (cycle-wise engine data).
Feature Engineering

Create rolling window sequences
Compute Remaining Useful Life (RUL) targets
Model Training

Train time-series models (LSTM/GRU) to predict RUL
Model Evaluation

Evaluate using RMSE and compare predicted vs actual RUL
Risk Thresholding

Define thresholds to trigger maintenance alerts
Visualization & Output

Dashboards showing RUL trends and alert zones
🏗️ Architecture
Data Processing → pandas, numpy
Visualization → matplotlib, seaborn
Model Training → TensorFlow/Keras (LSTM, GRU)
Evaluation → scikit-learn metrics
Deployment (optional) → Flask/Streamlit, Docker
📊 Dataset
Source: NASA CMAPSS dataset
Files: train_FD001.txt, train_FD002.txt, train_FD003.txt, train_FD004.txt
Each row contains:
Engine ID
Cycle number
21 sensor measurements
📈 Future Improvements
Experiment with Transformers for time-series
Hyperparameter tuning (learning rate, sequence length)
Deploy via Streamlit dashboard
Integrate real-time streaming data
🧑‍💻 Contributors
Nithin G J – Project Developer
