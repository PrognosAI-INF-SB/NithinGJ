# 🧩 Milestone 2 — Model Development & Training  
### Project: **PrognosAI — AI-Driven Predictive Maintenance System**  
**Dataset:** NASA CMAPSS Multivariate Time-Series Sensor Data  

---

## 🎯 Objective  
The goal of this milestone is to develop and train a deep learning model capable of predicting the **Remaining Useful Life (RUL)** of industrial machinery.  
Using sequential sensor data, the model learns degradation patterns over time to support **predictive maintenance** and minimize unplanned downtime.

---

## ⚙️ Process Summary

### 1. Data Utilization  
- Used the preprocessed CMAPSS dataset from Milestone 1.  
- Split data into **training**, **validation**, and **test** sets to ensure reliable evaluation.

### 2. Data Normalization  
- Scaled sensor values using normalization techniques to improve training stability and model performance.

### 3. Sequence Preparation  
- Created **time-windowed sequences** to capture historical trends in sensor data.  
- Each sequence represented a fixed-length history of engine operation cycles.

### 4. Model Development  
- Implemented a **deep learning model** (LSTM/GRU-based) designed to process sequential time-series data.  
- The model outputs a single **RUL value** representing the estimated remaining life of the equipment.

### 5. Training Strategy  
- Trained the model for multiple epochs with **early stopping** to prevent overfitting.  
- Used **model checkpointing** to save the best-performing version during training.  
- Monitored **training and validation loss** for convergence.

---

## 📊 Results & Observations  

- **Loss Curves:**  
  Training and validation losses consistently decreased, confirming stable learning behavior.

- **Prediction Patterns:**  
  Predicted RUL values aligned well with actual RUL trends, showing the model effectively captured degradation patterns.

- **Performance Metrics:**  
  Achieved a **reasonable RMSE** on the validation set, suitable as a baseline for further optimization.

---

## 🧠 Insights  

- The model effectively learned **temporal dependencies** among multiple sensor readings.  
- Proper **regularization and early stopping** minimized overfitting risks.  
- Some deviations between actual and predicted RUL indicate opportunities for:  
  - Hyperparameter tuning (e.g., sequence length, learning rate).  
  - Feature refinement (selecting the most informative sensors).  

---

## ✅ Deliverables Completed  

- LSTM/GRU-based predictive model implemented.  
- Model trained and saved with optimal weights.  
- Visual results for training and validation loss generated.  
- Preliminary RUL predictions verified against validation data.

---

## 🔜 Next Steps (Milestone 3 — Model Evaluation & Performance Assessment)  

- Evaluate the trained model on **unseen test data**.  
- Compute **RMSE** and **MAE** metrics for full performance analysis.  
- Plot **Predicted vs Actual RUL** graphs for deeper insight.  
- Investigate **model biases** and identify opportunities for improvement.

---

### 📚 References  
- **Dataset:** NASA CMAPSS (Commercial Modular Aero-Propulsion System Simulation)  
- **Frameworks:** TensorFlow, Keras, Scikit-learn, Pandas, NumPy  
- **Visualization:** Matplotlib, Seaborn  

---

**© 2025 PrognosAI Team — AI-Driven Predictive Maintenance**

