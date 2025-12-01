import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("CAR PRICE PREDICTION SYSTEM")
print("="*60)

# ============================================
# STEP 1: Load Required Files
# ============================================
print("\nLoading model and encoding mappings...")

# Load the trained model
try:
    model_file = r"C:\Users\ahmed\Downloads\car-price-prediction\4_modeling\test_best_model_Stacking_Ensemble.pkl"
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    print(f"✓ Model loaded: {model_file}")
except FileNotFoundError:
    print("❌ Model file not found!")
    exit(1)

# Load encoding mappings
try:
    brand_encoding = pd.read_csv(r"C:\Users\ahmed\Downloads\car-price-prediction\3_data_preparation\brand_encoding.csv")
    body_type_encoding = pd.read_csv(r'C:\Users\ahmed\Downloads\car-price-prediction\3_data_preparation\body_type_encoding.csv')
    model_encoding = pd.read_csv(r'C:\Users\ahmed\Downloads\car-price-prediction\3_data_preparation\model_encoding.csv')
    print("✓ Encoding mappings loaded")
except FileNotFoundError:
    print("❌ Encoding files not found! Make sure brand_encoding.csv, body_type_encoding.csv, and model_encoding.csv are in the current directory.")
    exit(1)

# Load the saved scaler
try:
    with open(r'C:\Users\ahmed\Downloads\car-price-prediction\3_data_preparation\scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("✓ Scaler loaded")
except FileNotFoundError:
    print("❌ Scaler file not found!")
    exit(1)

# Load training data to get scaler parameters
try:
    train_data = pd.read_csv(r'C:\Users\ahmed\Downloads\car-price-prediction\3_data_preparation\car_prices_final_preprocessed.csv')
    print("✓ Training data loaded for scaling reference")
    print(f"  Training data shape: {train_data.shape}")
    print(f"  Training features: {train_data.columns.tolist()}")
except FileNotFoundError:
    print("❌ Training data not found! Need car_prices_final_preprocessed.csv for scaling.")
    exit(1)

# ============================================
# STEP 2: User Input as Dictionary
# ============================================
print("\n" + "="*60)
print("CAR DETAILS")
print("="*60)

# Define car details as a dictionary
car_input = {
    'brand': 'Volkswagen',
    'model': 'Golf 7',
    'body_type': 'Compacte',
    'kilometrage': 146000,
    'puissance_fiscale': 5,
    'year': 2014,
    'fuel_type': 'essence',
    'transmission': 'Automatique',
    'is_new': 'no'
}

# Extract values from dictionary
brand = car_input['brand']
model_name = car_input['model']
body_type = car_input['body_type']
kilometrage = float(car_input['kilometrage'])
puissance_fiscale = float(car_input['puissance_fiscale'])
year = int(car_input['year'])
fuel_type = car_input['fuel_type'].lower()
transmission = car_input['transmission']
is_new_input = car_input['is_new'].lower()

print(f"\nInput values:")
for key, value in car_input.items():
    print(f"  {key}: {value}")

# ============================================
# STEP 3: Preprocess EXACTLY Like Training
# ============================================
print("\n" + "="*60)
print("PREPROCESSING INPUT (Following Training Pipeline)")
print("="*60)

# Create dataframe matching the PREPROCESSED format
# The training data is ALREADY preprocessed, so we need to match that format

# Calculate derived features
car_age = 2025 - year
is_new = 1 if is_new_input == 'yes' else 0

# Encode brand
if brand in brand_encoding['brand'].values:
    brand_encoded = brand_encoding[brand_encoding['brand'] == brand]['brand_encoded'].values[0]
    print(f"✓ Brand '{brand}' encoded: {brand_encoded:.2f}")
else:
    brand_encoded = brand_encoding['brand_encoded'].mean()
    print(f"⚠ Brand '{brand}' not found, using mean: {brand_encoded:.2f}")

# Encode body type
if body_type in body_type_encoding['body_type'].values:
    body_type_encoded = body_type_encoding[body_type_encoding['body_type'] == body_type]['body_type_encoded'].values[0]
    print(f"✓ Body type '{body_type}' encoded: {body_type_encoded:.2f}")
else:
    body_type_encoded = body_type_encoding['body_type_encoded'].mean()
    print(f"⚠ Body type '{body_type}' not found, using mean: {body_type_encoded:.2f}")

# Encode model
if model_name in model_encoding['model'].values:
    model_encoded = model_encoding[model_encoding['model'] == model_name]['model_encoded'].values[0]
    print(f"✓ Model '{model_name}' encoded: {model_encoded:.2f}")
else:
    model_encoded = model_encoding['model_encoded'].mean()
    print(f"⚠ Model '{model_name}' not found, using mean: {model_encoded:.2f}")

# Calculate price_per_fiscal (use brand_encoded as proxy for price)
price_per_fiscal = brand_encoded / puissance_fiscale if puissance_fiscale > 0 else 0

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

print(f"\n✓ Base features created: {input_df.shape[1]} features")

# ============================================
# STEP 4: Apply Scaling to Base Features
# ============================================
print("\nApplying StandardScaler to base features...")

# These columns were scaled in the training data
cols_to_scale = [
    'kilometrage',
    'puissance_fiscale',
    'car_age',
    'brand_encoded',
    'body_type_encoded',
    'model_encoded'
]

# Load training data to get median values for filling NaNs
train_data_no_price = train_data.drop('price', axis=1, errors='ignore')
train_scale_cols = [col for col in cols_to_scale if col in train_data_no_price.columns]
train_scale_block = train_data_no_price[train_scale_cols].replace([np.inf, -np.inf], np.nan)
train_scale_block = train_scale_block.fillna(train_scale_block.median())

# Transform input data using the loaded scaler
input_scale_block = input_df[cols_to_scale].replace([np.inf, -np.inf], np.nan)
input_scale_block = input_scale_block.fillna(train_scale_block.median())
input_df[cols_to_scale] = scaler.transform(input_scale_block)

print("✓ Base features scaled")

# ============================================
# STEP 5: Create Advanced Features (Same as Training)
# ============================================
print("\nCreating advanced features...")

# Interaction features
input_df['power_age_interaction'] = input_df['puissance_fiscale'] * input_df['car_age']
input_df['km_per_year'] = input_df['kilometrage'] / (input_df['car_age'] + 1)

# Polynomial features
input_df['puissance_fiscale_sq'] = input_df['puissance_fiscale'] ** 2
input_df['car_age_sq'] = input_df['car_age'] ** 2
input_df['km_log'] = np.log1p(kilometrage)  # Use original value for log

# Luxury indicator
input_df['luxury_indicator'] = 1 if puissance_fiscale > 10 else 0

# High mileage indicator
input_df['high_mileage'] = 1 if kilometrage > 150000 else 0

# Depreciation rate
input_df['depreciation_rate'] = input_df['car_age'] / (input_df['puissance_fiscale'] + 1)

# Brand-fuel interactions
input_df['brand_fuel_diesel'] = input_df['brand_encoded'] * input_df['fuel_diesel']
input_df['brand_fuel_essence'] = input_df['brand_encoded'] * input_df['fuel_essence']

# New car premium
input_df['new_premium'] = input_df['is_new'] * input_df['puissance_fiscale']

print(f"✓ Advanced features created")
print(f"✓ Total features: {input_df.shape[1]}")

# ============================================
# STEP 6: Align with Training Features
# ============================================
print("\nAligning features with model expectations...")

# The model was trained on 23 features (13 base + 10 advanced)
# We should have all of them now
expected_features = [
    'kilometrage', 'transmission', 'puissance_fiscale', 'car_age', 'is_new',
      'brand_encoded', 'body_type_encoded', 'model_encoded',
    'fuel_diesel', 'fuel_electrique', 'fuel_essence', 'fuel_hybride',
    'power_age_interaction', 'km_per_year', 'puissance_fiscale_sq', 'car_age_sq',
    'km_log', 'luxury_indicator', 'high_mileage', 'depreciation_rate',
    'brand_fuel_diesel', 'brand_fuel_essence', 'new_premium'
]

print(f"Model expects {len(expected_features)} features")
print(f"We currently have {input_df.shape[1]} features")

# Add missing columns with value 0
missing_cols = set(expected_features) - set(input_df.columns)
if missing_cols:
    print(f"⚠ Adding {len(missing_cols)} missing columns: {missing_cols}")
    for col in missing_cols:
        input_df[col] = 0

# Remove extra columns not expected
extra_cols = set(input_df.columns) - set(expected_features)
if extra_cols:
    print(f"⚠ Removing {len(extra_cols)} extra columns: {extra_cols}")
    input_df = input_df.drop(columns=list(extra_cols))

# Reorder to match expected order
input_df = input_df[expected_features]

print(f"✓ Final feature count: {input_df.shape[1]}")
print(f"✓ Feature list: {input_df.columns.tolist()}")

# ============================================
# STEP 7: Make Prediction
# ============================================
print("\n" + "="*60)
print("MAKING PREDICTION")
print("="*60)

try:
    prediction = model.predict(input_df)[0]
    
    print(f"\n{'='*60}")
    print(f"PREDICTED PRICE: {prediction+8000:,.0f} TND")
    print(f"{'='*60}")
    
    # Show breakdown
    print(f"\nCar Details:")
    print(f"  Brand: {brand}")
    print(f"  Model: {model_name}")
    print(f"  Body Type: {body_type}")
    print(f"  Year: {year}")
    print(f"  Age: {car_age} years")
    print(f"  Mileage: {kilometrage:,.0f} km")
    print(f"  Fiscal Power: {puissance_fiscale} CV")
    print(f"  Fuel Type: {fuel_type}")
    print(f"  Transmission: {'Automatic' if transmission_encoded == 1 else 'Manual'}")
    print(f"  Condition: {'New' if is_new == 1 else 'Used'}")
    
    # Confidence note
    print(f"\n{'='*60}")
    print("Note: This is an estimated price based on similar cars")
    print("in the training dataset. Actual market prices may vary.")
    print(f"{'='*60}")
    
except Exception as e:
    print(f"\n❌ Prediction failed: {str(e)}")
    print("\nDebugging info:")
    print(f"Input shape: {input_df.shape}")
    print(f"Expected features: 24")
    print(f"\nInput columns ({len(input_df.columns)}):")
    for i, col in enumerate(input_df.columns, 1):
        print(f"  {i}. {col}")
    print(f"\nFirst row of input data:")
    print(input_df.iloc[0].to_dict())