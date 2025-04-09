import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.seq2seq_lstm import Seq2SeqLSTM

# === Config ===
X_FILE = 'data/sequences/X.npy'
Y_FILE = 'data/sequences/y_both.npy'  # Change to y_total.npy or y_pv.npy
MODEL_SAVE_PATH = 'outputs/models/lstm_both.pt'
LOSS_PLOT_PATH = 'outputs/plots/loss_lstm_both.png'

LOOKBACK = 168
HORIZON = 24
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data():
    X = np.load(X_FILE)
    y = np.load(Y_FILE)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    input_dim = X.shape[2]
    output_dim = y.shape[2] if len(y.shape) > 2 else 1
    return loader, input_dim, output_dim


def train():
    os.makedirs('outputs/models', exist_ok=True)
    os.makedirs('outputs/plots', exist_ok=True)

    loader, input_dim, output_dim = load_data()

    model = Seq2SeqLSTM(input_dim, hidden_dim=64, output_dim=output_dim).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    losses = []

    print("Training started...\n")

    for epoch in range(EPOCHS):
        total_loss = 0
        model.train()

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()
            output = model(X_batch, y_batch, teacher_forcing_ratio=0.5)

            if torch.isnan(output).any():
                print("NaNs in output — skipping batch.")
                continue

            loss = criterion(output, y_batch)

            if torch.isnan(loss):
                print("NaN loss — skipping batch.")
                continue

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {avg_loss:.6f}")

    # === Post-training NaN Check ===
    model.eval()
    test_out = model(X_batch[:1].to(DEVICE))
    if torch.isnan(test_out).any():
        print("\nModel output contains NaNs — not saving model.")
    else:
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"\nModel saved to: {MODEL_SAVE_PATH}")

    # === Plot Loss ===
    plt.plot(losses, label="Train Loss")
    plt.title("LSTM Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(LOSS_PLOT_PATH)
    plt.close()
    print(f"Loss plot saved to: {LOSS_PLOT_PATH}")


if __name__ == "__main__":
    train()
