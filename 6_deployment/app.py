from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ============================================
# Load Model and Resources on Startup
# ============================================
print("Loading model and resources...")

# Load the trained model
model_file = r"C:\Users\ahmed\Downloads\car-price-prediction\4_modeling\test_best_model_Stacking_Ensemble.pkl"
with open(model_file, 'rb') as f:
    model = pickle.load(f)
print("✓ Model loaded")

# Load encoding mappings
brand_encoding = pd.read_csv(r"C:\Users\ahmed\Downloads\car-price-prediction\3_data_preparation\brand_encoding.csv")
body_type_encoding = pd.read_csv(r'C:\Users\ahmed\Downloads\car-price-prediction\3_data_preparation\body_type_encoding.csv')
model_encoding = pd.read_csv(r'C:\Users\ahmed\Downloads\car-price-prediction\3_data_preparation\model_encoding.csv')
print("✓ Encoding mappings loaded")

# Load the saved scaler
with open(r'C:\Users\ahmed\Downloads\car-price-prediction\3_data_preparation\scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
print("✓ Scaler loaded")

# Load training data for scaling reference
train_data = pd.read_csv(r'C:\Users\ahmed\Downloads\car-price-prediction\3_data_preparation\car_prices_final_preprocessed.csv')
print("✓ Training data loaded")

# Extract unique values for dropdowns
brands_list = sorted(brand_encoding['brand'].unique().tolist())
body_types_list = sorted(body_type_encoding['body_type'].unique().tolist())
models_list = sorted(model_encoding['model'].unique().tolist())

print("="*60)
print("Flask app ready to start!")
print("="*60)


def preprocess_and_predict(car_input):
    """
    Preprocess car input and make prediction
    """
    # Extract values from input
    brand = car_input['brand']
    model_name = car_input['model']
    body_type = car_input['body_type']
    kilometrage = float(car_input['kilometrage'])
    puissance_fiscale = float(car_input['puissance_fiscale'])
    year = int(car_input['year'])
    fuel_type = car_input['fuel_type'].lower()
    transmission = car_input['transmission']
    is_new_input = car_input['is_new'].lower()

    # Calculate derived features
    car_age = 2025 - year
    is_new = 1 if is_new_input == 'yes' else 0

    # Encode brand
    if brand in brand_encoding['brand'].values:
        brand_encoded = brand_encoding[brand_encoding['brand'] == brand]['brand_encoded'].values[0]
    else:
        brand_encoded = brand_encoding['brand_encoded'].mean()

    # Encode body type
    if body_type in body_type_encoding['body_type'].values:
        body_type_encoded = body_type_encoding[body_type_encoding['body_type'] == body_type]['body_type_encoded'].values[0]
    else:
        body_type_encoded = body_type_encoding['body_type_encoded'].mean()

    # Encode model
    if model_name in model_encoding['model'].values:
        model_encoded = model_encoding[model_encoding['model'] == model_name]['model_encoded'].values[0]
    else:
        model_encoded = model_encoding['model_encoded'].mean()

    # Encode transmission
    transmission_encoded = 1 if transmission.lower() in ['automatique', 'automatic'] else 0

    # One-hot encode fuel type
    fuel_diesel = 1 if fuel_type == 'diesel' else 0
    fuel_electrique = 1 if fuel_type == 'electrique' else 0
    fuel_essence = 1 if fuel_type == 'essence' else 0
    fuel_hybride = 1 if fuel_type == 'hybride' else 0

    # Build initial dataframe with UNSCALED values
    input_df = pd.DataFrame({
        'kilometrage': [kilometrage],
        'transmission': [transmission_encoded],
        'puissance_fiscale': [puissance_fiscale],
        'car_age': [car_age],
        'is_new': [is_new],
        'brand_encoded': [brand_encoded],
        'body_type_encoded': [body_type_encoded],
        'model_encoded': [model_encoded],
        'fuel_diesel': [fuel_diesel],
        'fuel_electrique': [fuel_electrique],
        'fuel_essence': [fuel_essence],
        'fuel_hybride': [fuel_hybride]
    })

    # Apply scaling to base features
    cols_to_scale = [
        'kilometrage',
        'puissance_fiscale',
        'car_age',
        'brand_encoded',
        'body_type_encoded',
        'model_encoded'
    ]

    train_data_no_price = train_data.drop('price', axis=1, errors='ignore')
    train_scale_cols = [col for col in cols_to_scale if col in train_data_no_price.columns]
    train_scale_block = train_data_no_price[train_scale_cols].replace([np.inf, -np.inf], np.nan)
    train_scale_block = train_scale_block.fillna(train_scale_block.median())

    input_scale_block = input_df[cols_to_scale].replace([np.inf, -np.inf], np.nan)
    input_scale_block = input_scale_block.fillna(train_scale_block.median())
    input_df[cols_to_scale] = scaler.transform(input_scale_block)

    # Create advanced features
    input_df['power_age_interaction'] = input_df['puissance_fiscale'] * input_df['car_age']
    input_df['km_per_year'] = input_df['kilometrage'] / (input_df['car_age'] + 1)
    input_df['puissance_fiscale_sq'] = input_df['puissance_fiscale'] ** 2
    input_df['car_age_sq'] = input_df['car_age'] ** 2
    input_df['km_log'] = np.log1p(kilometrage)
    input_df['luxury_indicator'] = 1 if puissance_fiscale > 10 else 0
    input_df['high_mileage'] = 1 if kilometrage > 150000 else 0
    input_df['depreciation_rate'] = input_df['car_age'] / (input_df['puissance_fiscale'] + 1)
    input_df['brand_fuel_diesel'] = input_df['brand_encoded'] * input_df['fuel_diesel']
    input_df['brand_fuel_essence'] = input_df['brand_encoded'] * input_df['fuel_essence']
    input_df['new_premium'] = input_df['is_new'] * input_df['puissance_fiscale']

    # Align with training features
    expected_features = [
        'kilometrage', 'transmission', 'puissance_fiscale', 'car_age', 'is_new',
        'brand_encoded', 'body_type_encoded', 'model_encoded',
        'fuel_diesel', 'fuel_electrique', 'fuel_essence', 'fuel_hybride',
        'power_age_interaction', 'km_per_year', 'puissance_fiscale_sq', 'car_age_sq',
        'km_log', 'luxury_indicator', 'high_mileage', 'depreciation_rate',
        'brand_fuel_diesel', 'brand_fuel_essence', 'new_premium'
    ]

    # Add missing columns
    missing_cols = set(expected_features) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0

    # Remove extra columns
    extra_cols = set(input_df.columns) - set(expected_features)
    if extra_cols:
        input_df = input_df.drop(columns=list(extra_cols))

    # Reorder to match expected order
    input_df = input_df[expected_features]

    # Make prediction
    prediction = model.predict(input_df)[0]
    
    return prediction


@app.route('/')
def index():
    """
    Render the main page with the form
    """
    return render_template('index.html', 
                         brands=brands_list,
                         body_types=body_types_list,
                         models=models_list)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle prediction request
    """
    try:
        # Get form data
        car_input = {
            'brand': request.form.get('brand'),
            'model': request.form.get('model'),
            'body_type': request.form.get('body_type'),
            'kilometrage': request.form.get('kilometrage'),
            'puissance_fiscale': request.form.get('puissance_fiscale'),
            'year': request.form.get('year'),
            'fuel_type': request.form.get('fuel_type'),
            'transmission': request.form.get('transmission'),
            'is_new': request.form.get('is_new')
        }
        
        # Make prediction
        predicted_price = preprocess_and_predict(car_input)
        
        # Adjust price (based on your original code)
        final_price = predicted_price + 8000
        
        return jsonify({
            'success': True,
            'predicted_price': f"{final_price:,.0f}",
            'car_details': car_input
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


if __name__ == '__main__':
    app.run(debug=True, port=5000)