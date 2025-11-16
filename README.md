# <b>Heart Disease Prediction using XGBoost</b>

<b>Heart Disease Prediction</b> is an end-to-end machine learning project that predicts the presence of heart disease using clinical and demographic features. Built with <b>XGBoost</b> and deployed via <b>FastAPI</b>, it demonstrates a complete ML pipeline from data exploration to cloud deployment.

---

## <b>Live API Demo</b>

**Render Deployment:**
- API Endpoint: [https://heartdisease-pred.onrender.com](https://heartdisease-pred.onrender.com)
- Interactive Docs: [https://heartdisease-pred.onrender.com/docs](https://heartdisease-pred.onrender.com/docs)

---

## <b>Project Overview</b>

This project builds a machine learning model to predict heart disease using the public **Heart Failure Prediction Dataset** from Kaggle.

### <b>Medical Features Used:</b>
- Age
- Gender
- Chest pain type
- Resting blood pressure
- Cholesterol level
- ECG results
- Maximum heart rate achieved
- Exercise-induced angina
- ST depression
- ST slope
- And more…

### <b>Objective</b>

Develop an end-to-end ML pipeline that:

1. Loads and cleans the heart disease dataset
2. Performs extensive exploratory data analysis
3. Trains multiple models (Logistic Regression, Decision Trees, Random Forest, Gradient Boosting, XGBoost)
4. Tunes hyperparameters and selects the best model
5. Exports model + DictVectorizer
6. Serves predictions through a FastAPI web service
7. Containerizes the service using Docker
8. Deploys the application on Render Cloud

### <b>API Response Format:</b>
```json
{
  "heartdisease": 0 or 1,
  "probability": float
}
```

---

## <b>Dataset</b>

**Dataset:** Heart Failure Prediction Dataset  
**Source:** [Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)

The dataset is included in this repository under `data/heart.csv`

**Download using Kaggle CLI:**
```bash
kaggle datasets download -d fedesoriano/heart-failure-prediction
```

---

## <b>Jupyter Notebook (notebook.ipynb)</b>

The notebook includes comprehensive analysis and model development:

### <b>Data Cleaning & Preparation</b>
- Lowercasing column names
- Imputing RestingBP and Cholesterol (zero → median)
- Train/val/test split (stratified)

### <b>EDA</b>
- Distribution plots for each feature
- Target imbalance analysis
- Crosstabs for categorical features
- Correlation matrix
- Visual feature-importance from models

### <b>Model Training & Hyperparameter Tuning</b>

**Models Evaluated:**
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- **XGBoost (final model)**

Hyperparameters tuned using GridSearchCV and manual experiments.

---

## <b>Getting Started</b>

### <b>1. Clone the repository</b>
```bash
git clone https://github.com/your-username/heart-disease-prediction.git
cd heart-disease-prediction
```

### <b>2. Install dependencies</b>

This project uses **uv** for dependency management.

```bash
uv sync
```

Dependencies are listed in `pyproject.toml` and `uv.lock`

---

## <b>Training Script – train.py</b>

The training script performs the complete ML pipeline:

**Features:**
- Loads the dataset
- Performs preprocessing
- Trains XGBoost model
- Evaluates AUC on validation/test sets
- Saves the pipeline to `models/heart_model.bin`

**Run training locally:**
```bash
uv run python train.py
```

---

## <b>Prediction Service – predict.py</b>

Built using **FastAPI** for real-time inference.

### <b>API Endpoint</b>
```
POST /predict
```

### <b>Example Request:</b>
```json
{
  "age": 54,
  "sex": "M",
  "chestpaintype": "NAP",
  "restingbp": 140,
  "cholesterol": 239,
  "fastingbs": 0,
  "restingecg": "Normal",
  "maxhr": 160,
  "exerciseangina": "N",
  "oldpeak": 1.0,
  "st_slope": "Flat"
}
```

### <b>Example Response:</b>
```json
{
  "heartdisease": 0,
  "probability": 0.33679
}
```

### <b>Run locally:</b>
```bash
uv run python predict.py
```

### <b>Open API documentation (Swagger):</b>
```
http://localhost:8000/docs
```

---

## <b>Docker Containerization</b>

A Dockerfile is included for easy containerization.

### <b>Build Docker image:</b>
```bash
docker build -t heart-failure-xgb .
```

### <b>Run container:</b>
```bash
docker run -p 8000:8000 heart-failure-xgb
```

### <b>Test container:</b>
- API: `http://localhost:8000`
- Docs: `http://localhost:8000/docs`

---

## <b>☁️ Cloud Deployment (Render)</b>

The Docker image is pushed to Docker Hub:
```
lanarkite99/heartdisease_pred:latest
```

### <b>Render Configuration:</b>
- **Runtime:** Docker
- **Port:** 8000 (auto-detected)
- **Start command** (handled by Dockerfile):
```bash
uv run uvicorn predict:app --host 0.0.0.0 --port 8000
```

---

## <b>Project Summary</b>

This project demonstrates:

- ✅ Comprehensive data cleaning + EDA
- ✅ Multiple model training & hyperparameter tuning
- ✅ Exporting trained model with DictVectorizer
- ✅ Serving real-time inference using FastAPI
- ✅ Full Docker containerization
- ✅ Cloud deployment using Render
- ✅ Reproducibility using uv + pyproject.toml

---
