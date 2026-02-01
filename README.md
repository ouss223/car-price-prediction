# Car Price Prediction

A machine learning project that predicts car prices based on various features like brand, model, body type, and condition. This end-to-end solution includes data collection, exploratory analysis, model training, and a web-based deployment interface.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Data Pipeline](#data-pipeline)
- [Deployment](#deployment)
- [Technologies](#technologies)

---

## Overview

This project aims to build an accurate car price prediction model using machine learning techniques. It analyzes factors that influence car prices in the Tunisian market and provides predictions through an interactive web interface.

**Why this matters:** Understanding car pricing helps buyers, sellers, and dealers make informed decisions in the used car market.

---

## Features

- **Automated Data Scraping**: Collects car data from multiple online sources
- **Comprehensive EDA**: In-depth exploratory data analysis with visualizations
- **Data Preprocessing**: Handles missing values, encoding, and feature scaling
- **Multiple Models**: Experiments with various ML algorithms
- **Ensemble Methods**: Stacking ensemble for improved predictions
- **Interactive Web Interface**: User-friendly Flask app for real-time predictions
- **Evaluation Metrics**: Detailed performance analysis and comparisons


##  Data Pipeline

### 1. **Data Collection** (`data_scraper/`)
   - Scrapes car listings from Tunisian automotive websites
   - Collects features: brand, model, year, price, mileage, body type, condition
   - Sources: Automobile.tn (new & used), Tayara

### 2. **Exploratory Data Analysis** (`2_data_understanding/`)
   - Analyzes price distributions and trends
   - Identifies correlations between features
   - Visualizes patterns and outliers
   - Generates insights for preprocessing

### 3. **Data Preprocessing** (`3_data_preparation/`)
   - **Cleaning**: Removes duplicates and handles missing values
   - **Encoding**: Converts categorical variables (brand, model, body type) to numerical format
   - **Feature Engineering**: Creates derived features
   - **Scaling**: Normalizes numerical features for model training
   - **Output**: Clean dataset ready for modeling

### 4. **Model Development** (`4_modeling/`)
   - Experiments with multiple algorithms:
     - Linear Regression
     - Random Forest
     - Gradient Boosting (XGBoost, CatBoost)
     - **Stacking Ensemble** (best performer)
   - Hyperparameter tuning with GridSearch
   - Cross-validation for robust evaluation

### 5. **Model Evaluation** (`5_evaluation/`)
   - Performance metrics: R², RMSE, MAE
   - Comparison of different models
   - Error analysis and residual plots
   - Detailed evaluation report

### 6. **Deployment** (`6_deployment/`)
   - Flask web application for real-time predictions
   - Clean, interactive user interface
   - Integration with trained model and preprocessing pipelines

---

##  Deployment

### Running the Web App Locally

```bash
cd 6_deployment
python app.py
```

The app will be available at `http://localhost:5000`

### Features of the Web Interface

- **Input Form**: Enter car details (brand, model, body type, year, etc.)
- **Real-time Prediction**: Instant price prediction
- **Confidence Metrics**: See prediction reliability
- **Responsive Design**: Works on desktop and mobile devices

### Deployment Options

- **Local Development**: Run `python app.py`
- **Heroku**: Deploy with Procfile and requirements.txt
- **Docker**: Containerize for scalable deployment
- **Cloud Platforms**: AWS, Google Cloud, Azure



##  Technologies

### Data Processing
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Scikit-learn**: Machine learning preprocessing and models

### Visualization
- **Matplotlib**: Static plots
- **Seaborn**: Statistical visualizations

### Machine Learning
- **Scikit-learn**: ML algorithms and preprocessing
- **XGBoost/CatBoost**: Gradient boosting models

### Deployment
- **Flask**: Web framework
- **Pickle/Joblib**: Model serialization

### Development
- **Jupyter Lab**: Interactive notebooks
- **PyYAML**: Configuration management


