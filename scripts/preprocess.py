import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# === File paths ===
DATA_DIR = 'data'
MEASURED_FILE = os.path.join(DATA_DIR, 'rsfmeasureddata2011.csv')
WEATHER_FILE = os.path.join(DATA_DIR, 'rsfweatherdata2011.csv')
PROCESSED_FILE = os.path.join(DATA_DIR, 'processed_dataset.csv')
PLOT_FILE = 'outputs/plots/energy_trend.png'

def load_and_merge_data():
    print("Loading data...")

    energy = pd.read_csv(MEASURED_FILE)
    weather = pd.read_csv(WEATHER_FILE)

    print(f"Energy shape: {energy.shape}")
    print(f"Weather shape: {weather.shape}")

    # Standardize column names
    energy = energy.rename(columns={'Date and Time': 'DateTime'})
    weather = weather.rename(columns={'DATE AND TIME': 'DateTime'})

    # Parse datetime
    energy['DateTime'] = pd.to_datetime(energy['DateTime'], errors='coerce')
    weather['DateTime'] = pd.to_datetime(weather['DateTime'], errors='coerce')

    # Drop unneeded columns
    energy = energy.drop(columns=['Day of Week', 'Unnamed: 11'], errors='ignore')
    weather = weather.drop(columns=['Unnamed: 6'], errors='ignore')

    # Merge on datetime
    merged = pd.merge(energy, weather, on='DateTime', how='inner')
    print(f"Merged data shape: {merged.shape}")

    return merged


def clean_and_engineer(df):
    print("Cleaning and engineering features...")

    df = df.copy()

    # Drop original DateTime later after extracting features
    df['hour'] = df['DateTime'].dt.hour
    df['dayofweek'] = df['DateTime'].dt.dayofweek

    # Drop any rows with missing values BEFORE normalization
    df = df.dropna()

    # Drop datetime after extracting features
    df = df.drop(columns=['DateTime'])

    print(f"After cleaning: {df.shape}")
    return df


def normalize_data(df):
    print("Normalizing numeric features...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    print(f"After normalization: {df.shape}")
    return df


def plot_trend(df):
    os.makedirs(os.path.dirname(PLOT_FILE), exist_ok=True)
    plt.figure(figsize=(12, 4))
    if 'Total Building (kW)' in df.columns:
        plt.plot(df['Total Building (kW)'][:500], label='Total Building (kW)')
    elif 'Building Net (kW)' in df.columns:
        plt.plot(df['Building Net (kW)'][:500], label='Building Net (kW)')
    plt.title('Energy Trend (First 500 hrs)')
    plt.xlabel('Hour')
    plt.ylabel('kW')
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_FILE)
    plt.close()
    print(f"Saved plot to: {PLOT_FILE}")


def main():
    df = load_and_merge_data()
    df = clean_and_engineer(df)
    df = normalize_data(df)

    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(PROCESSED_FILE, index=False)
    print(f"Saved processed data to: {PROCESSED_FILE}")

    plot_trend(df)
    print("Preprocessing complete.")


if __name__ == '__main__':
    main()
