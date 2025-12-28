import torch
import torch.nn as nn
import os
from tqdm import tqdm


class BiRNNClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hidden_dim=128, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            emb_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, input_ids, hidden=None):

        emb = self.embedding(input_ids)
        out, hidden = self.lstm(emb, hidden)
        logits = self.fc(out)
        return logits, hidden
    
    def predict_next_token(self, input_ids, temperature=1.0, device="cpu"):
        self.eval()

        with torch.no_grad():
            if isinstance(input_ids, list):
                input_ids = torch.tensor([input_ids], device=device)
            elif input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0).to(device)
            else:
                input_ids = input_ids.to(device)

            logits, _ = self(input_ids)

            # берём логиты последнего токена
            logits = logits[:, -1, :] / temperature

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            return next_token.item()
    
    def generate_text(self, input_ids, max_new_tokens=50, temperature=1.0, device="cpu", tokenizer=None):
        self.eval()

        with torch.no_grad():
            if isinstance(input_ids, list):
                generated = input_ids.copy()
            else:
                generated = input_ids.tolist()

            for _ in range(max_new_tokens):
                next_token = self.predict_next_token(
                    generated,
                    temperature=temperature,
                    device=device
                )

                # стоп-токен
                if tokenizer and next_token == tokenizer.sep_token_id:
                    break

                generated.append(next_token)

                # ограничение контекста
                if len(generated) > 256:
                    generated = generated[-256:]

            return generated




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
            model = BiRNNClassifier(vocab_size=vocab_size, emb_dim=128,hidden_dim=128)
            param_count = count_parameters(model)
            print(f"{rnn_type:<8} | {combine:<6} | {param_count:>10,}") 