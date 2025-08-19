# feature_engineer.py

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, holidays_path='holidays.csv', inflation_path='inflation.csv'):
        self.holidays_path = holidays_path
        self.inflation_path = inflation_path

    def fit(self, X, y=None):
        # Load holidays and inflation data once
        self.holidays_df = pd.read_csv(self.holidays_path)
        self.holidays_df['DATE'] = pd.to_datetime(self.holidays_df['DATE'])
        self.holiday_dates = set(self.holidays_df['DATE'])

        self.inflation_df = pd.read_csv(self.inflation_path)
        self.inflation_df['Month'] = self.inflation_df['Month'].astype(int)
        self.inflation_df['Year'] = self.inflation_df['Year'].astype(int)
        return self

    def transform(self, X):
        df = X.copy()
        df['TXNDATE'] = pd.to_datetime(df['TXNDATE'], dayfirst=True)
        df = df.sort_values(by=['BRANCHID', 'TXNDATE']).reset_index(drop=True)

        # Date-based features
        df['DayOfWeek'] = df['TXNDATE'].dt.dayofweek
        df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
        df['Month'] = df['TXNDATE'].dt.month
        df['Year'] = df['TXNDATE'].dt.year
        df['IsHoliday'] = df['TXNDATE'].isin(self.holiday_dates).astype(int)
        df['IsNonWorkingDay'] = ((df['IsWeekend'] == 1) | (df['IsHoliday'] == 1)).astype(int)

        # Lag features
        for lag in [1, 2, 3, 7, 14]:
            df[f'Lag{lag}_Credit'] = df.groupby('BRANCHID')['TOTALCREDITAMOUNT'].shift(lag)
            df[f'Lag{lag}_Debit'] = df.groupby('BRANCHID')['TOTALDEBITAMOUNT'].shift(lag)
            df[f'Lag{lag}_Customers'] = df.groupby('BRANCHID')['CUSTOMER'].shift(lag)

        # Rolling features (mean and std) for windows 3, 7, 14, 30
        windows = [3, 7, 14, 30]
        for window in windows:
            df[f'Rolling{window}_CreditMean'] = df.groupby('BRANCHID')['TOTALCREDITAMOUNT'].transform(lambda x: x.shift(1).rolling(window).mean())
            df[f'Rolling{window}_CreditStd'] = df.groupby('BRANCHID')['TOTALCREDITAMOUNT'].transform(lambda x: x.shift(1).rolling(window).std())

            df[f'Rolling{window}_DebitMean'] = df.groupby('BRANCHID')['TOTALDEBITAMOUNT'].transform(lambda x: x.shift(1).rolling(window).mean())
            df[f'Rolling{window}_DebitStd'] = df.groupby('BRANCHID')['TOTALDEBITAMOUNT'].transform(lambda x: x.shift(1).rolling(window).std())

            df[f'Rolling{window}_CustomersMean'] = df.groupby('BRANCHID')['CUSTOMER'].transform(lambda x: x.shift(1).rolling(window).mean())
            df[f'Rolling{window}_CustomersStd'] = df.groupby('BRANCHID')['CUSTOMER'].transform(lambda x: x.shift(1).rolling(window).std())

        # Differences, ratios, z-scores for lag1 vs rolling means for windows 3 and 7
        target_cols = ['Credit', 'Debit', 'Customers']
        rolling_windows = [3, 7]

        for target in target_cols:
            lag_col = f'Lag1_{target}'
            for window in rolling_windows:
                mean_col = f'Rolling{window}_{target}Mean'
                std_col = f'Rolling{window}_{target}Std'

                df[f'{lag_col}_vs_Mean{window}'] = df[lag_col] - df[mean_col]
                df[f'{lag_col}_Ratio_Mean{window}'] = df[lag_col] / (df[mean_col] + 1e-6)
                df[f'{lag_col}_ZScore{window}'] = (df[lag_col] - df[mean_col]) / (df[std_col] + 1e-6)

        df = df.sort_values(['BRANCHID', 'TXNDATE'], ascending=[True, True])

        # Additional engineered features
        df['Debit_to_Credit_Ratio'] = df['TOTALDEBITAMOUNT'] / (df['TOTALCREDITAMOUNT'] + 1)
        df['CashOut_Per_Customer'] = df['TOTALDEBITAMOUNT'] / (df['CUSTOMER'] + 1)
        df['CashIn_Per_Customer'] = df['TOTALCREDITAMOUNT'] / (df['CUSTOMER'] + 1)
        df['NetCashFlow'] = df['TOTALCREDITAMOUNT'] - df['TOTALDEBITAMOUNT']
        df['NetCashFlow_Per_Customer'] = df['NetCashFlow'] / (df['CUSTOMER'] + 1)
        df['Delta_Debit'] = df.groupby('BRANCHID')['TOTALDEBITAMOUNT'].diff(1)
        df['Delta_Credit'] = df.groupby('BRANCHID')['TOTALCREDITAMOUNT'].diff(1)
        df['Delta_Customers'] = df.groupby('BRANCHID')['CUSTOMER'].diff(1)
        df['Debit_ZScore'] = (df['TOTALDEBITAMOUNT'] - df['Rolling7_DebitMean']) / (df['Rolling7_DebitStd'] + 1)
        df['Credit_ZScore'] = (df['TOTALCREDITAMOUNT'] - df['Rolling7_CreditMean']) / (df['Rolling7_CreditStd'] + 1)
        df['IsMonthStart'] = df['TXNDATE'].dt.is_month_start.astype(int)
        df['IsMonthEnd'] = df['TXNDATE'].dt.is_month_end.astype(int)
        df['DayOfMonth'] = df['TXNDATE'].dt.day
        df['DaysInMonth'] = df['TXNDATE'].dt.days_in_month
        df['IsFirst5Days'] = (df['DayOfMonth'] <= 5).astype(int)
        df['IsLast5Days'] = (df['DayOfMonth'] > (df['DaysInMonth'] - 5)).astype(int)

        # Merge inflation dataset
        df = df.merge(self.inflation_df, on=['Year', 'Month'], how='left')

        return df
