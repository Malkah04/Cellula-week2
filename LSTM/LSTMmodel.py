
import torch
import torch.nn as nn


class LSTMclassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embedding_dim, 
            hidden_size=hidden_dim, 
            num_layers=1, 
            batch_first=True, 
            bidirectional=True
            )

        self.fc = nn.Sequential(
            nn.Dropout(0.5), 
            nn.Linear(hidden_dim * 2, 32), 
            nn.Tanh(), 
            nn.Linear(32, num_classes)
            )

        self.embedding_dropout = nn.Dropout(0.5)

    # def forward(self, x):
    #     padding_mask = (x != 0)
    #     x = self.embedding(x)
    #     x = self.embedding_dropout(x)
    #     lstm_out, _ = self.lstm(x)

    #     mask = padding_mask.unsqueeze(-1)
    #     masked_lstm_out = lstm_out * mask
    #     summed = masked_lstm_out.sum(dim=1)
    #     token_count = mask.sum(dim=1).clamp(min=1)
    #     mean_pool = summed / token_count

    #     masked_for_max = lstm_out.masked_fill(~mask, -1e9)
    #     max_pool = masked_for_max.max(dim=1).values

    #     out = torch.cat([mean_pool, max_pool], dim=1)
    #     return self.fc(out)
 
    def forward(self, x):
        padding_mask = (x != 0)
        x = self.embedding(x)
        x = self.embedding_dropout(x)
        lstm_out, _ = self.lstm(x)

        mask = padding_mask.unsqueeze(-1)

        masked_for_max = lstm_out.masked_fill(~mask, -1e9)
        max_pool = masked_for_max.max(dim=1).values

        return self.fc(max_pool)

