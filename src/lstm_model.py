import torch
import torch.nn as nn
import os
from tqdm import tqdm


class BiRNNClassifier(nn.Module):
    def __init__(self, vocab_size, hidden_dim=128, rnn_type="GRU", combine="concat"):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.combine = combine


        rnn_cls = {"RNN": nn.RNN, "GRU": nn.GRU, "LSTM": nn.LSTM}[rnn_type]
        self.rnn = rnn_cls(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)


        out_dim = hidden_dim * 2 if combine == "concat" else hidden_dim
        self.fc = nn.Linear(out_dim, vocab_size)


    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.rnn(emb)
        center = x.size(1) // 2
        hidden_forward = out[:, center, :out.size(2)//2]
        hidden_backward = out[:, center, out.size(2)//2:]
        hidden_agg = hidden_forward + hidden_backward if self.combine == "sum" else torch.cat([hidden_forward, hidden_backward], dim=1)
        linear_out = self.fc(hidden_agg)
        return linear_out
    


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    # Только тестирование модели, без train_loader
    vocab_size = 30522  # Примерный размер словаря BERT
    model = BiRNNClassifier(vocab_size)
    print(f"Тест модели: {sum(p.numel() for p in model.parameters()):,} параметров")