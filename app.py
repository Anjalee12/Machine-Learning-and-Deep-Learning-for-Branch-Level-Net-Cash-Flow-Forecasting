import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from feature_engineer import FeatureEngineer
from preprocessing import preprocess_data

# ---------------------------------------
# WeightedBlender Class (needed for pickle)
# ---------------------------------------
class WeightedBlender:
    def __init__(self, model1, model2, weight1, weight2, features1, features2):
        self.model1 = model1
        self.model2 = model2
        self.weight1 = weight1
        self.weight2 = weight2
        self.features1 = features1
        self.features2 = features2

    def predict(self, X):
        X1 = X[self.features1]
        X2 = X[self.features2]
        pred1 = self.model1.predict(X1)
        pred2 = self.model2.predict(X2)
        return self.weight1 * pred1 + self.weight2 * pred2

# ---------------------------------------
# Load Models, FeatureEngineer, and Mappings
# ---------------------------------------
@st.cache_resource
def load_models_and_data():
    with open("xgb_best_model.pkl", "rb") as f:
        xgb_model = pickle.load(f)
    with open("lgb_best_model.pkl", "rb") as f:
        lgb_model = pickle.load(f)
    with open("xgb_selected_features.json", "r") as f:
        xgb_features = json.load(f)
    with open("lgb_selected_features.json", "r") as f:
        lgb_features = json.load(f)
    with open("branch_map.json", "r", encoding="utf-8") as f:
        branch_map = json.load(f)
    # Load weighted blended model
    with open("weighted_blended_model.pkl", "rb") as f:
        blended_model = pickle.load(f)

    fe = FeatureEngineer(
        holidays_path='holidays.csv', 
        inflation_path='inflation.csv'
    )
    fe.fit(None)  # Load holidays and inflation

    return xgb_model, lgb_model, xgb_features, lgb_features, fe, branch_map, blended_model

# Load everything
xgb_model, lgb_model, xgb_features, lgb_features, fe, branch_map, blended_model = load_models_and_data()

# ---------------------------------------
# Streamlit UI
# ---------------------------------------
st.title("💰 Net Cash Flow Prediction App")

# Select transaction date
txn_date = st.date_input("Transaction Date")

# Select branch name from dropdown
branch_name = st.selectbox("Select Branch", options=list(branch_map.keys()))
branch_id = branch_map[branch_name]

# Prediction trigger
if st.button("Predict"):
    # Raw input dict
    input_dict = {
        'TXNDATE': pd.to_datetime(txn_date).strftime('%Y-%m-%d'),
        'BRANCHID': branch_id,
        'TOTALCREDITAMOUNT': np.nan,
        'TOTALDEBITAMOUNT': np.nan,
        'CUSTOMER': np.nan
    }

    input_df = pd.DataFrame([input_dict])

    # Feature engineering
    fe_df = fe.transform(input_df)

    # Preprocessing in prediction mode
    _, _, X_processed, _ = preprocess_data(
        fe_df,
        keep_features=list(set(xgb_features + lgb_features)),
        mode="predict"  # avoid dropping rows
    )

    # Make predictions using blended model
    pred = blended_model.predict(X_processed)[0]

    st.success(f"📊 Predicted Net Cash Flow: **{pred:,.2f}**")
