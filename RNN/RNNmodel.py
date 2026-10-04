import torch
import torch.nn as nn


class RNNclassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        self.rnn = nn.RNN(input_size=embedding_dim, 
            hidden_size=hidden_dim, 
            num_layers=1, 
            batch_first=True, 
            nonlinearity="tanh"
            )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.LeakyReLU(), 
            nn.Dropout(0.5),
            nn.Linear(32, num_classes)
            )

    def forward(self, x):
        padding_mask = (x != 0).float()
        x = self.embedding(x)
        rnn_out, _ = self.rnn(x)
        mask = padding_mask.unsqueeze(-1)
        masked_rnn_out = rnn_out * mask
        summed = masked_rnn_out.sum(dim=1)
        token_count = mask.sum(dim=1).clamp(min=1)
        out = summed / token_count
        return self.fc(out)

