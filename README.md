# AI-Based Network Traffic Analysis

An unsupervised machine learning system for detecting anomalous network traffic flows and potential security threats using **Isolation Forest**.

## 📌 Features
- **Unsupervised Anomaly Detection:** Utilizes Isolation Forest to flag outlier network flows without requiring labeled attack signatures during inference.
- **Interactive Web Interface:** Streamlit dashboard (`app.py`) for uploading network captures, inspecting traffic patterns, and visualizing detected anomalies in real time.
- **RESTful API:** FastAPI service (`src/api.py`) exposing health checks and `/predict` endpoints for automated network flow evaluations.
- **Model Training Pipeline:** Automated data preprocessing, imputation, feature alignment, and evaluation pipeline (`src/train.py`).
- **Comprehensive Testing:** Automated unit and regression tests (`tests/test_predict.py`) verifying model artifact integrity and feature compatibility.

---

## 📁 Project Structure

```text
ai-network-traffic-analysis2/
├── app.py                      # Streamlit interactive web dashboard
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules for environments & large data
├── data/
│   ├── README.md               # Dataset details and download instructions
│   ├── raw/                    # Raw CIC-IDS2017 capture CSV files (git-ignored)
│   └── processed/              # Processed samples and features (git-ignored)
├── models/
│   └── isolation_forest.joblib # Serialized trained Isolation Forest model & metadata
├── reports/
│   └── model_metrics.json      # Model performance and evaluation metrics
├── src/
│   ├── api.py                  # FastAPI REST service
│   ├── predict.py              # Flow anomaly prediction engine
│   ├── preprocess.py           # Feature extraction & data cleaning
│   └── train.py                # Model training and validation pipeline
└── tests/
    └── test_predict.py         # Unit tests for prediction and model artifacts
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- `pip`

### 2. Environment Setup
Clone the repository and install the dependencies:
```bash
git clone https://github.com/Himanshpapnai/ai-network-traffic-analysis2.git
cd ai-network-traffic-analysis2
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Running the Streamlit Web Application
Launch the interactive dashboard:
```bash
streamlit run app.py
```

### 4. Running the REST API Backend
Start the FastAPI server:
```bash
uvicorn src.api:app --reload
```
Once started, the API docs are available at `http://127.0.0.1:8000/docs`.

### 5. Running Tests
Execute unit tests to ensure model integrity:
```bash
pytest tests/
```

### 6. Training the Model
To re-train the Isolation Forest model on the CIC-IDS2017 dataset:
1. Place the dataset CSVs inside `data/raw/` (see `data/README.md`).
2. Run:
```bash
python src/train.py
```
Evaluation metrics will be output to `reports/model_metrics.json`.
