import pandas as pd
import numpy as np
import pickle
import json
from feature_engineer import FeatureEngineer
from preprocessing import preprocess_data
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Step 1: Load raw data & feature engineer
print("Loading data...")
df = pd.read_csv("data.csv")
fe = FeatureEngineer(holidays_path='holidays.csv', inflation_path='inflation.csv')
df_fe = fe.fit_transform(df)

# Step 2: Preprocess (to get train/test split)
print("Preprocessing data...")
X_train, y_train, X_test, y_test = preprocess_data(df_fe)

# Step 3: Load selected features
with open("xgb_selected_features.json", "r") as f:
    xgb_features = json.load(f)
with open("lgb_selected_features.json", "r") as f:
    lgb_features = json.load(f)

# Step 4: Load trained models
with open("xgb_best_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)
with open("lgb_best_model.pkl", "rb") as f:
    lgb_model = pickle.load(f)

# Step 5: Prepare test data subsets for each model
X_test_xgb = X_test[xgb_features]
X_test_lgb = X_test[lgb_features]

# Step 6: Predict with each model
print("Predicting with XGBoost...")
xgb_preds = xgb_model.predict(X_test_xgb)
print("Predicting with LightGBM...")
lgb_preds = lgb_model.predict(X_test_lgb)

# Step 7: Blend predictions (you can tune weights)
xgb_weight = 0.6
lgb_weight = 0.4
blended_preds = xgb_weight * xgb_preds + lgb_weight * lgb_preds

# Step 8: Evaluate blended model
rmse = np.sqrt(mean_squared_error(y_test, blended_preds))
mae = mean_absolute_error(y_test, blended_preds)
r2 = r2_score(y_test, blended_preds)

print(f"\nBlended Model Evaluation Metrics:")
print(f"  MAE : {mae:,.2f}")
print(f"  RMSE: {rmse:,.2f}")
print(f"  R²  : {r2:.4f}")
