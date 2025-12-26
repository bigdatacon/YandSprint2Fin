# train.py
import torch
import torch.nn as nn
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
from lstm_model import BiRNNClassifier  # импорт из нового файла
from next_token_dataset import NextTokenDataset, collate_fn

def main():
    # Настройки
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используемое устройство: {device}")
    
    # Пути
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, "data", "processed")
    tokenizer_path = os.path.join(data_dir, "tokenizer")
    
    # 1. Загружаем токенизатор
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
    vocab_size = tokenizer.vocab_size
    print(f"Токенизатор загружен. Размер словаря: {vocab_size}")
    
    # 2. Загружаем тексты
    def load_texts(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f]
    
    train_texts = load_texts(os.path.join(data_dir, "train.txt"))
    val_texts = load_texts(os.path.join(data_dir, "val.txt"))
    print(f"Загружено: train={len(train_texts)} текстов, val={len(val_texts)} текстов")
    
    # 3. Создаем датасеты и даталоадеры
    train_dataset = NextTokenDataset(train_texts, tokenizer)
    val_dataset = NextTokenDataset(val_texts, tokenizer)
    
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    print(f"Даталоадеры созданы: train={len(train_loader)} batches, val={len(val_loader)} batches")
    
    # 4. Создаем модель
    model = BiRNNClassifier(vocab_size, hidden_dim=128, rnn_type="LSTM", combine="concat")
    model = model.to(device)
    
    # 5. Оптимизатор и функция потерь
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.CrossEntropyLoss()
    
    # 6. Функция оценки
    def evaluate(model, loader):
        model.eval()
        correct, total = 0, 0
        sum_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)
                sum_loss += loss.item()
        return sum_loss / len(loader), correct / total
    
    # 7. Обучение
    n_epochs = 3
    
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.
        
        for x_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}"):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss, val_acc = evaluate(model, val_loader)
        
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f} | Val Accuracy: {val_acc:.2%}")
    
    # 8. Сохраняем модель
    os.makedirs("models", exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': vocab_size,
        'hidden_dim': 128,
        'rnn_type': 'LSTM',
        'combine': 'concat'
    }, "models/trained_model.pth")
    
    print("Модель сохранена в 'models/trained_model.pth'")

if __name__ == "__main__":
    main()