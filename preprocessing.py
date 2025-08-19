import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(
    df,
    split_date_str="2025-01-01",
    drop_multicollinear=True,
    keep_features=None,
    mode="train"  # "train" or "predict"
):
    df["TXNDATE"] = pd.to_datetime(df["TXNDATE"], format="%Y-%m-%d")
    df.sort_values(["BRANCHID", "TXNDATE"], inplace=True)

    df_encoded = df.copy()

    # ✅ Encode only if the column exists
    for col in ["BRANCH", "DISTRICT", "PROVINCE", "CODE"]:
        if col in df_encoded.columns:
            df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])

    rolling_lag_cols = [
        "Lag1_Credit", "Lag1_Debit", "Lag1_Customers", "Lag2_Credit", "Lag2_Debit", "Lag2_Customers",
        "Lag3_Credit", "Lag3_Debit", "Lag3_Customers", "Lag7_Credit", "Lag7_Debit", "Lag7_Customers",
        "Lag14_Credit", "Lag14_Debit", "Lag14_Customers",
        "Rolling7_DebitMean", "Rolling7_DebitStd", "Rolling7_CustomersMean", "Rolling7_CustomersStd",
        "Rolling7_CreditMean", "Rolling7_CreditStd",
        "Rolling3_CreditMean", "Rolling3_DebitMean", "Rolling3_CustomersMean", "Rolling3_CustomersStd",
        "Rolling3_CreditStd", "Rolling3_DebitStd",
        "Rolling30_CreditMean", "Rolling30_DebitMean", "Rolling30_CustomersMean", "Rolling30_CustomersStd",
        "Rolling30_CreditStd", "Rolling30_DebitStd",
        "Lag1_Credit_vs_Mean3", "Lag1_Credit_Ratio_Mean3", "Lag1_Credit_ZScore3",
        "Lag1_Credit_vs_Mean7", "Lag1_Credit_Ratio_Mean7", "Lag1_Credit_ZScore7",
        "Lag1_Debit_vs_Mean3", "Lag1_Debit_Ratio_Mean3", "Lag1_Debit_ZScore3",
        "Lag1_Debit_vs_Mean7", "Lag1_Debit_Ratio_Mean7", "Lag1_Debit_ZScore7",
        "Lag1_Customers_vs_Mean3", "Lag1_Customers_Ratio_Mean3", "Lag1_Customers_ZScore3",
        "Lag1_Customers_vs_Mean7", "Lag1_Customers_Ratio_Mean7", "Lag1_Customers_ZScore7",
        "Delta_Debit", "Delta_Credit", "Delta_Customers", "Debit_ZScore", "Credit_ZScore"
    ]

    # ✅ Only drop NA for training
    if mode == "train":
        df_cleaned = df_encoded.dropna(subset=rolling_lag_cols)
    else:
        df_cleaned = df_encoded.copy()

    # ✅ Train/test split (only relevant during training, harmless for single-row test)
    split_date = pd.to_datetime(split_date_str)
    train_df = df_cleaned[df_cleaned["TXNDATE"] < split_date].copy()
    test_df = df_cleaned[df_cleaned["TXNDATE"] >= split_date].copy()

    target_col = "NetCashFlow"
    drop_cols = ["TXNDATE", "BRANCHID", "NetCashFlow", "NetCashFlow_Per_Customer", "TOTALTXNAMOUNT"]
    feature_cols = [col for col in train_df.columns if col not in drop_cols]

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    # ✅ Optional multicollinearity dropper
    if drop_multicollinear:
        if keep_features is None:
            keep_features = []

        cor_matrix = X_train.corr().abs()
        high_corr_pairs = []
        for i in range(len(cor_matrix.columns)):
            for j in range(i):
                if cor_matrix.iloc[i, j] > 0.8:
                    f1 = cor_matrix.columns[i]
                    f2 = cor_matrix.columns[j]
                    high_corr_pairs.append((f1, f2, cor_matrix.iloc[i, j]))

        feature_target_corr = X_train.corrwith(y_train).abs()
        to_drop = []
        for f1, f2, _ in high_corr_pairs:
            if f1 in keep_features or f2 in keep_features:
                continue
            if f1 in to_drop or f2 in to_drop:
                continue
            if feature_target_corr[f1] < feature_target_corr[f2]:
                to_drop.append(f1)
            else:
                to_drop.append(f2)

        X_train = X_train.drop(columns=to_drop, errors='ignore')
        X_test = X_test.drop(columns=to_drop, errors='ignore')

    return X_train, y_train, X_test, y_test
