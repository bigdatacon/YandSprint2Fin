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
    from transformers import BertTokenizerFast
    import os
    
    # Определяем путь к токенизатору
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    tokenizer_path = os.path.join(project_root, "data", "processed", "tokenizer")
    
    # Загружаем токенизатор
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
    vocab_size = tokenizer.vocab_size
    
    print(f"Размер словаря: {vocab_size}")
    hidden_dim = 128

    rnn_types = ["RNN", "GRU", "LSTM"]
    combine_methods = ["sum", "concat"]


    # Сравнение
    print(f"{'RNN Type':<8} | {'Combine':<6} | {'Params':>10}")
    print("-" * 35)
    for rnn_type in rnn_types:
        for combine in combine_methods:
            model = BiRNNClassifier(vocab_size, hidden_dim, rnn_type, combine)
            param_count = count_parameters(model)
            print(f"{rnn_type:<8} | {combine:<6} | {param_count:>10,}") 