# Deep Learning ANN Model with MLflow Integration (Wine Quality Prediction)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00)](https://www.tensorflow.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking%20%26%20Registry-0194E2)](https://mlflow.org/)

A production-minded Machine Learning / Deep Learning project demonstrating how to build an **Artificial Neural Network (ANN)** with **TensorFlow/Keras** to predict **wine quality**, while integrating **MLflow** for end-to-end experiment tracking and model lifecycle management. It also includes **hyperparameter optimization using Hyperopt**, run comparison, best-model selection, and model logging/registration for deployment.

---

## 1. Project Overview

This repository walks through a practical workflow for training and improving a Keras ANN model for wine quality prediction:

- Load and preprocess the Wine Quality dataset
- Split data into train / validation / test sets
- Train an ANN model using TensorFlow/Keras
- Optimize hyperparameters with Hyperopt
- Track metrics, parameters, and artifacts with MLflow
- Compare experiments in MLflow UI
- Select and register the best-performing model
- Prepare the final model for REST API deployment and containerization

---

## 2. Project Architecture / Workflow

High-level workflow:

1. **Data Loading**
2. **Preprocessing & Feature Engineering**
3. **Dataset Split** (Train / Validation / Test)
4. **Model Definition** (Keras ANN)
5. **Training & Evaluation**
6. **Hyperparameter Optimization** (Hyperopt)
7. **Experiment Tracking** (MLflow: params, metrics, artifacts)
8. **Model Comparison** (MLflow UI)
9. **Best Model Selection**
10. **Model Logging & (Optional) Registration**
11. **Deployment Prep**
    - REST API serving
    - Docker containerization

---

## 3. Features

- ANN model training using **TensorFlow/Keras**
- **Train/validation/test** split with reproducibility best practices
- Hyperparameter tuning with **Hyperopt**
- Centralized experiment tracking with **MLflow**:
  - parameters (e.g., layers, neurons, learning rate)
  - metrics (e.g., loss, MAE/MSE/RMSE, accuracy if applicable)
  - artifacts (model files, plots, preprocessing assets)
- Run comparison and model selection in **MLflow UI**
- Model logging and optional registration for deployment
- Deployment-ready guidance (REST API + Docker)

---

## 4. Tech Stack

- **Python**
- **TensorFlow / Keras**
- **MLflow**
- **Hyperopt**
- **Scikit-learn**
- **Pandas**
- **NumPy**

---

## 5. Installation

### 5.1 Clone the repository

```bash
git clone https://github.com/yashjajoria/-Deep-Learning-ANN-Model-with-MLflow-Integration-Wine-Quality-Prediction-.git
cd -Deep-Learning-ANN-Model-with-MLflow-Integration-Wine-Quality-Prediction-
```

### 5.2 Create & activate a virtual environment (recommended)

**macOS/Linux**
```bash
python -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 5.3 Install dependencies

If you have a `requirements.txt`:
```bash
pip install -r requirements.txt
```

If you don’t have one yet, a typical baseline is:
```bash
pip install tensorflow mlflow hyperopt scikit-learn pandas numpy
```

---

## 6. Quickstart Guide

> The exact script names may vary depending on your repository files. Common entrypoints are `train.py`, `main.py`, or notebooks in `notebooks/`.

Example flow:

1) Start MLflow tracking server/UI (local)  
2) Run a baseline training run  
3) Run hyperparameter tuning  
4) Compare runs & pick best model  
5) Log/register best model  
6) Deploy via REST API and optionally Docker  

---

## 7. Running Hyperparameter Tuning

Hyperopt is used to search across hyperparameter combinations (e.g., number of layers, units, dropout, batch size, learning rate).

Example command (adjust to your entrypoint):
```bash
python src/hyperopt_tuning.py --max-evals 50
```

Suggested Hyperopt tuning knobs:
- `--max-evals` : number of trials
- `--seed` : reproducibility
- `--experiment-name` : MLflow experiment grouping

---

## 8. Viewing Experiments in MLflow UI

### 8.1 Start MLflow UI (local)

From the repository root:
```bash
mlflow ui --host 0.0.0.0 --port 5000
```

Then open in your browser:
- `http://localhost:5000`

### 8.2 (Optional) Set tracking URI

If your code uses MLflow explicitly, you may set:
```bash
export MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
```

Windows (PowerShell):
```bash
$env:MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
```

---

## 9. Selecting the Best Model

Typical strategies to select the best model:

- Lowest validation loss (or lowest RMSE/MSE)
- Best generalization (validation metric + test metric verification)
- Simpler model if results are comparable (avoid overfitting)

In MLflow UI you can:
- Sort/filter runs by a metric (e.g., `val_loss`)
- Compare params across runs
- Open artifacts to review saved models/plots
- Promote the best run for model logging/registration

---

## 10. Deploying Model as REST API

A practical deployment approach is:

- Save/export the model (MLflow artifact)
- Load it inside a lightweight API (FastAPI/Flask)
- Provide a `/predict` endpoint
- Validate inputs with a schema
- Return predictions + metadata

### Example (FastAPI) pseudo-implementation

```python
from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd

app = FastAPI(title="Wine Quality Predictor")

class WineInput(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

model = mlflow.pyfunc.load_model("models:/WineQualityANN/Production")  # Example model registry path

@app.post("/predict")
def predict(payload: WineInput):
    df = pd.DataFrame([payload.model_dump()])
    pred = model.predict(df)
    return {"prediction": float(pred[0])}
```

> Notes:
> - Your actual feature set must match the training pipeline exactly.
> - If you used preprocessing (scaling/encoding), log it with MLflow and load it consistently.

---

## 11. Building Docker Container for Deployment

Below is a **template** Docker setup (adjust file names/paths to your repo).

### Example `Dockerfile`

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Build and run

```bash
docker build -t wine-quality-mlflow-ann:latest .
docker run -p 8000:8000 wine-quality-mlflow-ann:latest
```

Then:
- `http://localhost:8000/docs` (FastAPI Swagger UI)

---

## 12. Example Code Usage

### Training a baseline model (example)

```bash
python src/train.py --experiment-name "wine-quality-ann" --epochs 50 --batch-size 32
```

### Running evaluation (example)

```bash
python src/evaluate.py --run-id "<mlflow_run_id>"
```

### Making a prediction (example snippet)

```python
import requests

payload = {
  "fixed_acidity": 7.4,
  "volatile_acidity": 0.70,
  "citric_acid": 0.00,
  "residual_sugar": 1.9,
  "chlorides": 0.076,
  "free_sulfur_dioxide": 11.0,
  "total_sulfur_dioxide": 34.0,
  "density": 0.9978,
  "pH": 3.51,
  "sulphates": 0.56,
  "alcohol": 9.4
}

resp = requests.post("http://localhost:8000/predict", json=payload)
print(resp.json())
```

---

## 13. Project Structure

A commonly used structure for this kind of project:

```text
.
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
│   ├── train.py
│   ├── hyperopt_tuning.py
│   ├── evaluate.py
│   ├── api.py
│   └── utils.py
├── mlruns/                 # created by MLflow (local tracking)
├── models/                 # optional local exports
├── requirements.txt
├── Dockerfile
└── README.md
```

> Your repository may differ—update this section to reflect the actual layout.

---

## 14. Future Improvements

- Add a robust **data validation** layer (e.g., Great Expectations)
- Add **feature scaling / pipelines** persisted with MLflow artifacts
- Add **cross-validation** and more systematic evaluation
- Add **model explainability** (e.g., SHAP) and log explanations as artifacts
- Add CI checks (linting, unit tests, formatting) with GitHub Actions
- Deploy to a managed service (e.g., AWS/GCP/Azure) with full MLOps workflow

---

## 15. License

Specify your license here. Common choices:
- MIT
- Apache-2.0
- GPL-3.0

If you already have a `LICENSE` file in the repository, ensure this section matches it.

---

### Acknowledgements

- Wine Quality dataset (commonly used from UCI / Kaggle mirrors)
- TensorFlow/Keras, MLflow, Hyperopt open-source communities
