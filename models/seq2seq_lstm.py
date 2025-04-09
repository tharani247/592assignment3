import torch
import torch.nn as nn

class Seq2SeqLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.2):
        super(Seq2SeqLSTM, self).__init__()

        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.decoder = nn.LSTM(output_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, y_init=None, teacher_forcing_ratio=0.5):
        batch_size = x.size(0)
        horizon = 24
        output_dim = self.fc.out_features

        # Encode input sequence
        _, (hidden, cell) = self.encoder(x)

        # Initialize decoder input
        if y_init is None:
            decoder_input = torch.zeros(batch_size, 1, output_dim).to(x.device)
        else:
            decoder_input = y_init[:, 0:1, :]

        outputs = []

        for t in range(horizon):
            out, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            pred = self.fc(out)
            outputs.append(pred)

            if y_init is not None and torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = y_init[:, t:t+1, :]
            else:
                decoder_input = pred

        outputs = torch.cat(outputs, dim=1)  # shape: (batch, horizon, output_dim)
        return outputs
