# Tourism Demand Forecasting in Sri Lanka Using Deep Learning

This repository contains the implementation of a tourism demand forecasting system developed as part of a Master's Independent Study. The project focuses on predicting monthly tourist arrivals in Sri Lanka using advanced deep learning–based time-series forecasting models.

## 📌 Project Overview

Accurate tourism demand forecasting is essential for strategic planning, policy formulation, and resource allocation. However, tourism demand is highly volatile due to external shocks such as economic crises and pandemics.

This study applies two state-of-the-art deep learning approaches:

- **NeuralProphet**
- **Temporal Fusion Transformer (TFT)**

The best-performing model was integrated into a Python-based web forecasting application.

---

## 🎯 Objectives

- Forecast monthly tourism demand in Sri Lanka
- Compare NeuralProphet and TFT models using standard error metrics
- Identify the most accurate forecasting approach
- Deploy the selected model through a web-based decision-support system

---

## 📂 Dataset

- Historical tourist arrivals data from **January 2014 to December 2023**
- Additional economic indicators included:
  - CCPI Index value
  - Exchange rate

The dataset was preprocessed to capture trends, seasonality, and external influences.

---

## 🧠 Models Implemented

### 1. NeuralProphet
NeuralProphet integrates:

- Trend modeling
- Seasonality components
- External regressors
- Neural network-based forecasting

**Best performance achieved:**
- **MAPE = 14.08%**

---

### 2. Temporal Fusion Transformer (TFT)
TFT is a transformer-based forecasting model capable of:

- Capturing long-term dependencies
- Handling multivariate time-series inputs
- Producing interpretable attention-based forecasts

**Performance:**
- **MAPE = 20.35%**

---

## 📊 Evaluation Metrics

Models were evaluated using:

- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)

Each model was executed across **20 independent runs** to ensure robustness.

---

## 🌐 Web Application

The NeuralProphet model was deployed as a Python-based forecasting web application using the Flask framework.

### Features:
- Upload or select tourism datasets
- Generate future tourist arrival forecasts
- Visualize historical trends and predictions

The application follows the **Model–View–Controller (MVC)** architecture.

---

## ⚙️ Requirements

Install dependencies using:

```bash
pip install -r requirements.txt

Install requirements:

pip install -r requirements.txt


Run the Flask app:

python app.py


Open in browser:
