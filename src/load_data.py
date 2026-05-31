import os
import pandas as pd
import numpy as np


class DataProcessor:
    """
    Load a CSV and run preprocessing steps, returning the processed DataFrame.
    Steps:
      - read CSV
      - drop customerID if present
      - coerce TotalCharges to numeric
      - create tenure_group bins
      - compute avg_monthly_spend, contract_value, low_charge
      - apply log1p transform to selected numeric features
    No plotting performed.
    """
    def __init__(self, filepath=None):
        self.filepath = filepath or os.path.join('data', 'dataset.csv')
        self.df = None

    def load(self):
        """Load CSV into a DataFrame (lazy)."""
        if self.df is None:
            self.df = pd.read_csv(self.filepath)
        return self.df

    def convert_total_charges(self):
        """Convert TotalCharges column to numeric, coercing errors to NaN."""
        self.load()
        self.df['TotalCharges'] = pd.to_numeric(self.df['TotalCharges'], errors='coerce')
        return self.df

    def create_features(self):
        """Create tenure_group, avg_monthly_spend, contract_value, low_charge."""
        self.load()
        # ensure numeric conversions done
        if 'TotalCharges' not in self.df or not pd.api.types.is_numeric_dtype(self.df['TotalCharges']):
            self.convert_total_charges()

        # tenure_group bins
        self.df['tenure_group'] = pd.cut(
            self.df['tenure'],
            bins=[0, 6, 12, 24, 48, 72],
            labels=['<6m', '6-12m', '12-24m', '24-48m', '48m+'],
            include_lowest=True
        )

        # avoid division by zero: set avg_monthly_spend to NaN where tenure <= 0 or TotalCharges is NaN
        tenure_positive = self.df['tenure'] > 0
        self.df['avg_monthly_spend'] = np.where(
            tenure_positive,
            self.df['TotalCharges'] / self.df['tenure'],
            np.nan
        )

        # contract value and low charge
        self.df['contract_value'] = self.df['MonthlyCharges'] * self.df['tenure']
        # keep as integer 1/0
        self.df['low_charge'] = (self.df['MonthlyCharges'] < 30).astype(int)

        return self.df

    def log_transform(self, cols):
        """Apply natural log1p to specified columns (preserves NaN)."""
        self.load()
        for col in cols:
            if col in self.df.columns:
                self.df[col] = np.log1p(self.df[col])
        return self.df

    def preprocess(self):
        """Run the full preprocessing pipeline and return the processed DataFrame."""
        self.load()
        if 'customerID' in self.df.columns:
            self.df = self.df.drop(columns=['customerID'])
        self.convert_total_charges()
        self.create_features()
        self.log_transform(['TotalCharges', 'MonthlyCharges', 'avg_monthly_spend', 'contract_value'])
        return self.df

    def get_processed(self):
        """Alias to preprocess for clarity."""
        return self.preprocess()


if __name__ == "__main__":
    processor = DataProcessor()
    processed_df = processor.preprocess()
    print(processed_df.head())
