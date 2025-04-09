import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.seq2seq_gru import Seq2SeqGRU

X_FILE = 'data/sequences/X.npy'
Y_FILE = 'data/sequences/y_both.npy'
MODEL_PATH = 'outputs/models/gru_both.pt'
OUTPUT_DIR = 'outputs/plots/eval/'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_metrics(y_true, y_pred):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    cvrmse = rmse / np.mean(y_true)

    return rmse, mae, r2, cvrmse


def evaluate_model():
    print("Loading data...")
    X = np.load(X_FILE)
    y = np.load(Y_FILE)

    X = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    y = torch.tensor(y, dtype=torch.float32).to(DEVICE)

    input_dim = X.shape[2]
    output_dim = y.shape[2]

    print("Loading model...")
    model = Seq2SeqGRU(input_dim, hidden_dim=64, output_dim=output_dim)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    print("Running predictions...")
    with torch.no_grad():
        predictions = model(X)

    predictions = predictions.cpu().numpy()
    truths = y.cpu().numpy()

    print(f"Total predictions: {predictions.size}")
    print(f"Total ground truths: {truths.size}")
    print(f"NaNs in predictions: {np.isnan(predictions).sum()}")
    print(f"NaNs in ground truths: {np.isnan(truths).sum()}")

    # Check for NaNs before computing metrics
    mask = ~np.isnan(predictions).any(axis=(1, 2))
    predictions = predictions[mask]
    truths = truths[mask]

    if predictions.size == 0:
        raise ValueError("No valid predictions to evaluate.")

    print("\nCalculating metrics...")
    rmse, mae, r2, cvrmse = compute_metrics(truths, predictions)

    print("Evaluation Results:")
    print(f"RMSE:    {rmse:.4f}")
    print(f"MAE:     {mae:.4f}")
    print(f"R²:      {r2:.4f}")
    print(f"CVRMSE:  {cvrmse:.4f}")

    # Plotting
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    labels = ['Total Building (kW)', 'PV (kW)']

    for i in range(output_dim):
        plt.figure(figsize=(12, 4))
        plt.plot(truths[0, :, i], label="Ground Truth")
        plt.plot(predictions[0, :, i], label="Prediction")
        plt.title(f"Prediction vs Ground Truth - {labels[i]}")
        plt.xlabel("Hour")
        plt.ylabel("kW")
        plt.legend()
        filename = f"{OUTPUT_DIR}prediction_GRU_{labels[i].replace(' ', '_').replace('(', '').replace(')', '')}.png"
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print(f"Saved: {filename}")


if __name__ == "__main__":
    evaluate_model()
