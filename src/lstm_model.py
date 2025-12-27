import torch
import torch.nn as nn
import os
from tqdm import tqdm


class BiRNNClassifier(nn.Module):
    def __init__(self, vocab_size, hidden_dim=128, rnn_type="LSTM", combine="concat"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.combine = combine

        rnn_cls = {"RNN": nn.RNN, "GRU": nn.GRU, "LSTM": nn.LSTM}[rnn_type]
        self.rnn = rnn_cls(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)

        out_dim = hidden_dim * 2 if combine == "concat" else hidden_dim
        self.fc = nn.Linear(out_dim, vocab_size)
        
    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.rnn(emb)
        
        # ИЗМЕНЕНИЕ: берем последние состояния, а не центральные
        # Для forward RNN: последний токен первой половины
        # Для backward RNN: первый токен второй половины
        seq_len = out.size(1)
        hidden_forward = out[:, -1, :self.hidden_dim]  # последний токен forward
        hidden_backward = out[:, 0, self.hidden_dim:]  # первый токен backward
        
        if self.combine == "sum":
            hidden_agg = hidden_forward + hidden_backward
        else:  # concat
            hidden_agg = torch.cat([hidden_forward, hidden_backward], dim=1)
        
        linear_out = self.fc(hidden_agg)
        return linear_out
    
    def predict_next_token(self, input_ids, temperature=1.0, device='cpu'):
        """Предсказание одного следующего токена"""
        self.eval()
        with torch.no_grad():
            # Преобразуем в тензор
            if isinstance(input_ids, list):
                input_tensor = torch.tensor([input_ids]).to(device)
            else:
                input_tensor = input_ids.unsqueeze(0).to(device) if input_ids.dim() == 1 else input_ids.to(device)
            
            # Получаем логиты
            logits = self.forward(input_tensor)
            
            # Применяем температуру
            logits = logits / temperature
            
            # Softmax для вероятностей
            probs = torch.softmax(logits, dim=-1)
            
            # Сэмплируем следующий токен
            next_token = torch.multinomial(probs, num_samples=1)
            
            return next_token.item()
    
    def generate_text(self, input_ids, max_length=50, temperature=1.0, device='cpu', tokenizer=None):
        """Генерация текста до конца фразы"""
        self.eval()
        generated = []
        
        with torch.no_grad():
            current_input = input_ids.copy() if isinstance(input_ids, list) else input_ids.tolist()
            
            for _ in range(max_length):
                # Предсказываем следующий токен
                next_token = self.predict_next_token(current_input, temperature, device)
                
                # Если [SEP] или длина слишком большая - останавливаемся
                if tokenizer and next_token == tokenizer.sep_token_id:
                    break
                    
                generated.append(next_token)
                current_input.append(next_token)
                
                # Ограничиваем длину контекста (опционально)
                if len(current_input) > 100:  # Ограничиваем историю
                    current_input = current_input[-100:]
        
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
            model = BiRNNClassifier(vocab_size, hidden_dim, rnn_type, combine)
            param_count = count_parameters(model)
            print(f"{rnn_type:<8} | {combine:<6} | {param_count:>10,}") 