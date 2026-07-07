"""
Arquitetura híbrida CNN + BiLSTM para classificação de batimentos ECG.

A CNN extrai features morfológicas locais (formato da onda),
a BiLSTM modela a dependência temporal entre essas features.
"""

import torch
import torch.nn as nn


class CNNLSTMClassifier(nn.Module):
    def __init__(self, n_classes=5, lstm_hidden=64, lstm_layers=2, dropout=0.3):
        super().__init__()

        # Bloco convolucional: entrada (batch, 1, seq_len) -> extrai features locais
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        # BiLSTM sobre a sequência de features extraídas pela CNN
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden * 2, n_classes)  # *2 por ser bidirecional

    def forward(self, x):
        # x chega como (batch, seq_len, 1) -> Conv1d espera (batch, canais, seq_len)
        x = x.permute(0, 2, 1)          # (batch, 1, seq_len)
        x = self.conv(x)                # (batch, 128, seq_len_reduzido)
        x = x.permute(0, 2, 1)          # (batch, seq_len_reduzido, 128) -> formato da LSTM

        lstm_out, (h_n, _) = self.lstm(x)

        # concatena o último hidden state das duas direções
        forward_last = h_n[-2]
        backward_last = h_n[-1]
        combined = torch.cat([forward_last, backward_last], dim=1)

        out = self.dropout(combined)
        out = self.fc(out)

        return out