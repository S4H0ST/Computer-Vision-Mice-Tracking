import torch
import torch.nn as nn


class RatActionRNN(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_classes=5, num_layers=2):
        """
        input_size: cx, cy, w, h, speed (5 datos)
        """
        super(RatActionRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (batch, sequence_length, features)
        # Inicializar estados ocultos
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))

        # Tomar solo el último paso de tiempo para la clasificación
        out = out[:, -1, :]
        out = self.fc(out)
        return out
