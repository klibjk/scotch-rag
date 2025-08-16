import pandas as pd
import pyarrow.parquet as pq

def profile_table(parquet_path):
    df = pd.read_parquet(parquet_path)
    profile = {}
    for col in df.columns:
        profile[col] = {
            'dtype': str(df[col].dtype),
            'nulls': int(df[col].isnull().sum()),
            'min': df[col].min() if pd.api.types.is_numeric_dtype(df[col]) else None,
            'max': df[col].max() if pd.api.types.is_numeric_dtype(df[col]) else None,
            'n_unique': df[col].nunique(),
            'natural_order': col == 'row_index_original',
        }
    return profile

def main():
    parquet_path = 'data/sample.parquet'
    profile = profile_table(parquet_path)
    for col, stats in profile.items():
        print(f"{col}: {stats}")

if __name__ == '__main__':
    main()
