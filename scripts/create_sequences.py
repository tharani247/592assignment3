import os
import numpy as np
import pandas as pd

# === Configuration ===
PROCESSED_FILE = 'data/processed_dataset.csv'
OUTPUT_DIR = 'data/sequences/'

LOOKBACK = 168
HORIZON = 24


def create_sequences(X_df, y_df):
    X, y = [], []
    for i in range(LOOKBACK, len(X_df) - HORIZON):
        X_seq = X_df.iloc[i - LOOKBACK:i].values
        y_seq = y_df.iloc[i:i + HORIZON].values
        X.append(X_seq)
        y.append(y_seq)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def main():
    print("Loading processed data...")
    df = pd.read_csv(PROCESSED_FILE)

    # Confirm target columns exist
    if 'Total Building (kW)' not in df.columns or 'PV (kW)' not in df.columns:
        raise ValueError("Missing required target columns in dataset.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Target columns
    target_total = ['Total Building (kW)']
    target_pv = ['PV (kW)']
    target_both = ['Total Building (kW)', 'PV (kW)']

    # Separate target columns first
    y_total_df = df[target_total]
    y_pv_df = df[target_pv]
    y_both_df = df[target_both]

    # Keep only numeric input features and remove targets
    X_df = df.select_dtypes(include=[np.number]).drop(columns=target_both)

    print("Creating sequences...")

    X, y_total = create_sequences(X_df, y_total_df)
    _, y_pv = create_sequences(X_df, y_pv_df)
    _, y_both = create_sequences(X_df, y_both_df)

    np.save(os.path.join(OUTPUT_DIR, 'X.npy'), X)
    np.save(os.path.join(OUTPUT_DIR, 'y_total.npy'), y_total)
    np.save(os.path.join(OUTPUT_DIR, 'y_pv.npy'), y_pv)
    np.save(os.path.join(OUTPUT_DIR, 'y_both.npy'), y_both)

    print(f"X shape: {X.shape}")
    print(f"y_total shape: {y_total.shape}")
    print(f"y_pv shape: {y_pv.shape}")
    print(f"y_both shape: {y_both.shape}")
    print(f"Sequences saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
