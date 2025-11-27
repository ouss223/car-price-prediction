# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import xgboost as xgb



"""
# Old Data With leak
file_path = r"C:\Users\charn\Downloads\car_prices_final_preprocessed (2).csv"
data=pd.read_csv(file_path)
X = data.drop("price", axis=1)
y = data["price"]
y_log = np.log1p(y)
X_train, X_test, y_train, y_test, y_train_log, y_test_log = train_test_split(
    X, y, y_log, test_size=0.2, random_state=42, stratify=y_bins
)
use_log_target = False
y_train = y_train_log if use_log_target else y_train
"""

# Load train/test splits
file_path = r"C:\Users\charn\Downloads\Xtrain.csv"
X_train = pd.read_csv(file_path)
file_path = r"C:\Users\charn\Downloads\Xtest.csv"
X_test = pd.read_csv(file_path)
file_path = r"C:\Users\charn\Downloads\Ytrain_log.csv"
y_train = pd.read_csv(file_path)
file_path = r"C:\Users\charn\Downloads\Ytest_log.csv"
y_test = pd.read_csv(file_path)

''' if you want to use non-log targets, uncomment below
file_path = r"C:\Users\charn\Downloads\Ytrain.csv"
y_train=pd.read_csv(file_path)
file_path = r"C:\Users\charn\Downloads\Ytest.csv"
y_test=pd.read_csv(file_path)
'''


y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

# Enhanced evaluation function with detailed statistics
def comprehensive_evaluation(model, X_train, X_test, y_train, y_test, model_name):
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE ANALYSIS FOR {model_name}")
    print(f"{'='*80}\n")
    
    # Fit the model
    model.fit(X_train, y_train)
    
    # Get predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Convert from log scale back to original prices
    y_train_original = np.exp(y_train)
    y_test_original = np.exp(y_test)
    y_train_pred_original = np.exp(y_train_pred)
    y_test_pred_original = np.exp(y_test_pred)
    
    # Calculate errors
    train_errors = y_train_pred - y_train
    test_errors = y_test_pred - y_test
    train_errors_original = y_train_pred_original - y_train_original
    test_errors_original = y_test_pred_original - y_test_original
    
    # Calculate performance metrics (LOG SCALE)
    mae_train_log = mean_absolute_error(y_train, y_train_pred)
    mae_test_log = mean_absolute_error(y_test, y_test_pred)
    rmse_train_log = np.sqrt(mean_squared_error(y_train, y_train_pred))
    rmse_test_log = np.sqrt(mean_squared_error(y_test, y_test_pred))
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    
    # Calculate performance metrics (ORIGINAL SCALE)
    mae_train_orig = mean_absolute_error(y_train_original, y_train_pred_original)
    mae_test_orig = mean_absolute_error(y_test_original, y_test_pred_original)
    rmse_train_orig = np.sqrt(mean_squared_error(y_train_original, y_train_pred_original))
    rmse_test_orig = np.sqrt(mean_squared_error(y_test_original, y_test_pred_original))
    r2_train_orig = r2_score(y_train_original, y_train_pred_original)
    r2_test_orig = r2_score(y_test_original, y_test_pred_original)
    
    # Calculate percentage errors (MAPE)
    mape_train = np.mean(np.abs(train_errors_original / y_train_original)) * 100
    mape_test = np.mean(np.abs(test_errors_original / y_test_original)) * 100
    
    # Print LOG SCALE metrics
    print("📊 METRICS IN LOG SCALE:")
    print(f"   Training   - MAE: {mae_train_log:.4f} | RMSE: {rmse_train_log:.4f} | R²: {r2_train:.4f}")
    print(f"   Testing    - MAE: {mae_test_log:.4f} | RMSE: {rmse_test_log:.4f} | R²: {r2_test:.4f}")
    
    # Print ORIGINAL SCALE metrics
    print(f"\n💰 METRICS IN ORIGINAL PRICE SCALE:")
    print(f"   Training   - MAE: {mae_train_orig:,.2f} | RMSE: {rmse_train_orig:,.2f} | R²: {r2_train_orig:.4f}")
    print(f"   Testing    - MAE: {mae_test_orig:,.2f} | RMSE: {rmse_test_orig:,.2f} | R²: {r2_test_orig:.4f}")
    print(f"\n   Mean Absolute Percentage Error (MAPE):")
    print(f"   Training: {mape_train:.2f}% | Testing: {mape_test:.2f}%")
    
    # Overfitting analysis
    print(f"\n🔍 OVERFITTING ANALYSIS:")
    mae_diff = ((mae_test_log - mae_train_log) / mae_train_log) * 100
    r2_diff = r2_train - r2_test
    print(f"   MAE Increase (Train→Test): {mae_diff:.1f}%")
    print(f"   R² Decrease (Train→Test): {r2_diff:.4f}")
    
    if mae_diff > 50 and r2_diff > 0.1:
        print(f"   ⚠️  SEVERE OVERFITTING DETECTED")
    elif mae_diff > 30 and r2_diff > 0.05:
        print(f"   ⚠️  MODERATE OVERFITTING")
    elif mae_diff > 15:
        print(f"   ⚡ MILD OVERFITTING")
    else:
        print(f"   ✅ GOOD GENERALIZATION")
    
    # Best and worst predictions
    print(f"\n🎯 BEST PREDICTIONS (Testing Set):")
    abs_errors_test = np.abs(test_errors_original)
    best_indices = np.argsort(abs_errors_test)[:5]
    
    for i, idx in enumerate(best_indices, 1):
        actual = y_test_original[idx]
        predicted = y_test_pred_original[idx]
        error = test_errors_original[idx]
        pct_error = (error / actual) * 100
        print(f"   {i}. Actual: ${actual:,.2f} | Predicted: {predicted:,.2f} | Error: {error:,.2f} ({pct_error:.2f}%)")
    
    print(f"\n❌ WORST PREDICTIONS (Testing Set):")
    worst_indices = np.argsort(abs_errors_test)[-5:][::-1]
    
    for i, idx in enumerate(worst_indices, 1):
        actual = y_test_original[idx]
        predicted = y_test_pred_original[idx]
        error = test_errors_original[idx]
        pct_error = (error / actual) * 100
        print(f"   {i}. Actual: ${actual:,.2f} | Predicted: {predicted:,.2f} | Error: {error:,.2f} ({pct_error:.2f}%)")
    
    # Error distribution statistics
    print(f"\n📈 ERROR DISTRIBUTION (Original Scale):")
    print(f"   Testing Errors - Mean: {np.mean(test_errors_original):,.2f}")
    print(f"   Testing Errors - Median: {np.median(test_errors_original):,.2f}")
    print(f"   Testing Errors - Std Dev: {np.std(test_errors_original):,.2f}")
    print(f"   Testing Errors - Min: {np.min(test_errors_original):,.2f}")
    print(f"   Testing Errors - Max: {np.max(test_errors_original):,.2f}")
    
    # Percentage of predictions within thresholds
    within_5_pct = np.sum(np.abs(test_errors_original / y_test_original) <= 0.05) / len(y_test_original) * 100
    within_10_pct = np.sum(np.abs(test_errors_original / y_test_original) <= 0.10) / len(y_test_original) * 100
    within_20_pct = np.sum(np.abs(test_errors_original / y_test_original) <= 0.20) / len(y_test_original) * 100
    
    print(f"\n🎲 PREDICTION ACCURACY THRESHOLDS:")
    print(f"   Within  5%: {within_5_pct:.1f}% of predictions")
    print(f"   Within 10%: {within_10_pct:.1f}% of predictions")
    print(f"   Within 20%: {within_20_pct:.1f}% of predictions")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Predicted vs Actual (Log Scale)
    axes[0, 0].scatter(y_test, y_test_pred, alpha=0.5, s=20)
    axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Log(Price)')
    axes[0, 0].set_ylabel('Predicted Log(Price)')
    axes[0, 0].set_title(f'{model_name} - Log Scale')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Predicted vs Actual (Original Scale)
    axes[0, 1].scatter(y_test_original, y_test_pred_original, alpha=0.5, s=20)
    axes[0, 1].plot([y_test_original.min(), y_test_original.max()], 
                    [y_test_original.min(), y_test_original.max()], 'r--', lw=2)
    axes[0, 1].set_xlabel('Actual Price ()')
    axes[0, 1].set_ylabel('Predicted Price ()')
    axes[0, 1].set_title(f'{model_name} - Original Scale')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Residual plot
    axes[1, 0].scatter(y_test_pred_original, test_errors_original, alpha=0.5, s=20)
    axes[1, 0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1, 0].set_xlabel('Predicted Price ()')
    axes[1, 0].set_ylabel('Residual Error ()')
    axes[1, 0].set_title('Residual Plot')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Error distribution
    axes[1, 1].hist(test_errors_original, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 1].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1, 1].set_xlabel('Prediction Error ()')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Error Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return model

# Evaluate all models
print("\n" + "="*80)
print("COMPARING ALL MODELS")
print("="*80)

rf_model = comprehensive_evaluation(
    RandomForestRegressor(n_estimators=100, random_state=42),
    X_train, X_test, y_train, y_test,
    "Random Forest"
)

xg_model = comprehensive_evaluation(
    xgb.XGBRegressor(n_estimators=100, random_state=42),
    X_train, X_test, y_train, y_test,
    "XGBoost"
)

ridge_model = comprehensive_evaluation(
    Ridge(alpha=1.0),
    X_train, X_test, y_train, y_test,
    "Ridge Regression"
)

lasso_model = comprehensive_evaluation(
    Lasso(alpha=0.1, max_iter=5000),
    X_train, X_test, y_train, y_test,
    "Lasso Regression"
)

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)