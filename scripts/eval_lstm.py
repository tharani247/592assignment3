import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Add project root for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.seq2seq_lstm import Seq2SeqLSTM

# === Configuration ===
X_FILE = 'data/sequences/X.npy'
Y_FILE = 'data/sequences/y_both.npy'       # Change to y_total.npy or y_pv.npy if needed
MODEL_FILE = 'outputs/models/lstm_both.pt' # Match the model file
PLOT_DIR = 'outputs/plots/eval/'
SAMPLE_INDEX = 10                          # Index to visualize
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Filter out any NaNs
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]

    if len(y_true_clean) == 0:
        raise ValueError("No valid samples left after removing NaNs.")

    mse = mean_squared_error(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    r2 = r2_score(y_true_clean, y_pred_clean)
    cvrmse = rmse / np.mean(y_true_clean)

    return rmse, mae, r2, cvrmse


def evaluate_model():
    os.makedirs(PLOT_DIR, exist_ok=True)

    print("Loading data...")
    X = np.load(X_FILE)
    y = np.load(Y_FILE)

    X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    y_tensor = torch.tensor(y, dtype=torch.float32).to(DEVICE)

    input_dim = X.shape[2]
    output_dim = y.shape[2] if len(y.shape) == 3 else 1

    print("Loading model...")
    model = Seq2SeqLSTM(input_dim, hidden_dim=64, output_dim=output_dim).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_FILE))
    model.eval()

    print("Running predictions...")
    all_preds, all_truths = [], []

    with torch.no_grad():
        for i in range(len(X_tensor)):
            x_sample = X_tensor[i:i+1]
            y_true = y_tensor[i].cpu().numpy().reshape(-1)

            y_pred = model(x_sample, teacher_forcing_ratio=0.0)
            y_pred = y_pred.squeeze(0).cpu().numpy().reshape(-1)

            all_preds.extend(y_pred)
            all_truths.extend(y_true)

    # Debug: Print info about NaNs
    all_preds = np.array(all_preds)
    all_truths = np.array(all_truths)

    print(f"Total predictions: {len(all_preds)}")
    print(f"Total ground truths: {len(all_truths)}")
    print(f"NaNs in predictions: {np.isnan(all_preds).sum()}")
    print(f"NaNs in ground truths: {np.isnan(all_truths).sum()}")

    print("\nCalculating metrics...")
    rmse, mae, r2, cvrmse = compute_metrics(all_truths, all_preds)

    print("Evaluation Results:")
    print(f"RMSE:    {rmse:.4f}")
    print(f"MAE:     {mae:.4f}")
    print(f"R²:      {r2:.4f}")
    print(f"CVRMSE:  {cvrmse:.4f}")

    print("\nSaving prediction plots...")
    with torch.no_grad():
        pred_seq = model(X_tensor[SAMPLE_INDEX:SAMPLE_INDEX+1], teacher_forcing_ratio=0.0)

    pred_seq = pred_seq.squeeze(0).cpu().numpy()
    true_seq = y_tensor[SAMPLE_INDEX].cpu().numpy()

    for i in range(output_dim):
        pred = pred_seq[:, i] if output_dim > 1 else pred_seq
        true = true_seq[:, i] if output_dim > 1 else true_seq
        label = "Total Building (kW)" if (i == 0 or "total" in Y_FILE.lower()) else "PV (kW)"

        plt.figure(figsize=(8, 4))
        plt.plot(range(len(true)), true, label="Ground Truth", marker='o')
        plt.plot(range(len(pred)), pred, label="Prediction", marker='x')
        plt.title(f"Prediction vs Ground Truth – {label}")
        plt.xlabel("Hour")
        plt.ylabel("Scaled Value")
        plt.legend()
        plt.tight_layout()
        save_path = os.path.join(PLOT_DIR, f"prediction_{label.replace(' ', '_')}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved: {save_path}")


if __name__ == '__main__':
    evaluate_model()
