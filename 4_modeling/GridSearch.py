import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from sklearn.model_selection import learning_curve


## Old Data With leak
#file_path = r"C:\Users\charn\Downloads\car_prices_final_preprocessed (2).csv"
#data=pd.read_csv(file_path)
#X = data.drop("price", axis=1)
#y = data["price"]
#y_log = np.log1p(y)
#X_train, X_test, y_train, y_test, y_train_log, y_test_log = train_test_split(
#    X, y, y_log, test_size=0.2, random_state=42, stratify=y_bins
#)
#use_log_target = False
#y_train = y_train_log if use_log_target else y_train


file_path = r"C:\Users\charn\Downloads\Xtrain (2).csv"
X_train=pd.read_csv(file_path)
file_path = r"C:\Users\charn\Downloads\Xtest (2).csv"
X_test=pd.read_csv(file_path)

file_path = r"C:\Users\charn\Downloads\Ytrain (2).csv"
y_train=pd.read_csv(file_path)
file_path = r"C:\Users\charn\Downloads\Ytest (2).csv"
y_test=pd.read_csv(file_path)

y_train = y_train.values.ravel()  # Flatten the column vector into a 1D array
y_test = y_test.values.ravel()  
# Define Pipelines for Each Model
pipelines = {
    'RandomForest': Pipeline([('model', RandomForestRegressor(random_state=42))]),
    'GBR': Pipeline([('model', GradientBoostingRegressor(random_state=42))]),
    'XGBoost': Pipeline([('model', xgb.XGBRegressor(objective='reg:squarederror', random_state=42))])
}

# Define Parameter Grids for Hyperparameter Tuning
param_grids = {

    'RandomForest': {
        'model__n_estimators': [50, 100, 200, 300],
        'model__max_depth': [None, 10, 20, 30],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4]
    },
    'GBR': {
        'model__n_estimators': [50, 100, 200],
        'model__learning_rate': [0.01, 0.1, 0.2],
        'model__max_depth': [3, 5, 10]
    },
    'XGBoost': {
        'model__n_estimators': [300],
        'model__learning_rate': [0.06, 0.05],
        'model__max_depth': [4,5,6,7],
        'model__subsample': [0.8],
        'model__colsample_bytree': [0.8]
    }
}
# Dictionary to store results
results = {}

# Run GridSearchCV for each model and evaluate performance
for name, pipe in pipelines.items():
    print(f"Running: {name}")
    grid = GridSearchCV(pipe, param_grids[name], cv=5, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    
    # Get the best estimator (model) and its hyperparameters
    best = grid.best_estimator_
    params = grid.best_params_
    
    # Predict on the test set
    y_pred = best.predict(X_test)
    from sklearn.metrics import mean_squared_error

    # Calculate MAE, R2, and cross-validation score
    mae = mean_absolute_error(y_test, y_pred)
    mse= mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Store results
    results[name] = {
        'best_params': params,
        'mae': mae,
        'mse': mse,
        'rmae': rmse,
        'r2': r2,
        'cv_score': cross_val_score(best, X_train, y_train, cv=5, scoring='neg_mean_absolute_error').mean(),
        'model': best
    }
    
    print(f"{name} → MAE: {mae:.2f},MSE: {mse:.2f},RMSE: {rmse:.2f}, R²: {r2:.3f}, best_params: {params}, CV MAE: {-results[name]['cv_score']:.2f}")

# Display Results
print("\nModel Comparison Results:")
for model_name, metrics in results.items():
    print(f"{model_name}:")
    print(f"  Best Params: {metrics['best_params']}")
    print(f"  MAE: {metrics['mae']}")
    print(f"  R²: {metrics['r2']}")
    print(f"  Cross-Validation MAE: {-metrics['cv_score']}\n")

# Plot the results for MAE (Training vs Testing)
models = list(results.keys())
mae_values = [results[model]['mae'] for model in models]
cv_values = [results[model]['cv_score'] for model in models]

plt.figure(figsize=(10, 6))
plt.bar(models, mae_values, color='blue', alpha=0.6, label="Test MAE")
plt.bar(models, [-cv for cv in cv_values], color='red', alpha=0.6, label="CV MAE")
plt.xlabel('Models')
plt.ylabel('Mean Absolute Error')
plt.title('Test vs Cross-Validation MAE for Models')
plt.legend()
plt.show()

# Learning Curves for each model to check for overfitting or underfitting
for name, pipe in pipelines.items():
    print(f"\nPlotting Learning Curves for {name}...")
    train_sizes, train_scores, test_scores = learning_curve(pipe, X_train, y_train, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5))
    
    # Calculate mean and std
    train_mean = -train_scores.mean(axis=1)
    test_mean = -test_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    test_std = test_scores.std(axis=1)
    
    # Plot learning curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label=f'{name} Train MAE', color='blue')
    plt.plot(train_sizes, test_mean, label=f'{name} Test MAE', color='red')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color='blue', alpha=0.2)
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color='red', alpha=0.2)
    plt.xlabel('Training Set Size')
    plt.ylabel('Mean Absolute Error')
    plt.title(f'Learning Curves for {name}')
    plt.legend()
    plt.show()

# Feature Importances for RandomForest, GBR, and XGBoost
for name in ['RandomForest', 'GBR', 'XGBoost']:
    if name in results:
        model = results[name]['model']
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            plt.figure(figsize=(10, 6))
            plt.title(f"Feature Importances for {name}")
            plt.bar(range(X_train.shape[1]), importances[indices], align="center")
            plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation=90)
            plt.xlabel('Features')
            plt.ylabel('Importance')
            plt.show()

# Prediction vs Actual plot for the best model
best_model = min(results, key=lambda model: results[model]['mae'])
y_pred_best = results[best_model]['model'].predict(X_test)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_best, color='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2)
plt.xlabel('True Prices')
plt.ylabel('Predicted Prices')
plt.title(f'Prediction vs Actual for {best_model}')
plt.show()
