import torch
import torch.nn as nn

class Seq2SeqGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        self.encoder = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.GRU(output_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, target=None, teacher_forcing_ratio=0.5):
        batch_size, _, _ = x.size()
        device = x.device

        # Encode input sequence
        _, hidden = self.encoder(x)

        # Initialize decoder input (start with zero)
        seq_len = target.size(1) if target is not None else 24
        output_dim = self.fc.out_features
        decoder_input = torch.zeros((batch_size, 1, output_dim), device=device)

        outputs = []

        for t in range(seq_len):
            out, hidden = self.decoder(decoder_input, hidden)
            pred = self.fc(out)
            outputs.append(pred)

            if target is not None and torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = target[:, t:t+1]
            else:
                decoder_input = pred

        return torch.cat(outputs, dim=1)
