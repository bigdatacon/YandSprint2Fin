# train_with_rouge.py
import torch
import torch.nn as nn
import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
from lstm_model import BiRNNClassifier
from next_token_dataset import NextTokenDataset, collate_fn
from rouge import Rouge  # pip install rouge

def calculate_rouge(predictions, targets, tokenizer):
    """Вычисление метрики ROUGE"""
    rouge = Rouge()
    
    # Декодируем токены в текст
    pred_texts = [tokenizer.decode(pred, skip_special_tokens=True) for pred in predictions]
    target_texts = [tokenizer.decode(target, skip_special_tokens=True) for target in targets]
    
    scores = []
    for pred, target in zip(pred_texts, target_texts):
        if pred and target:  # Проверяем, что строки не пустые
            try:
                score = rouge.get_scores(pred, target, avg=True)
                scores.append(score)
            except:
                continue
    
    if scores:
        # Усредняем по всем примерам
        avg_scores = {
            'rouge-1': {'f': np.mean([s['rouge-1']['f'] for s in scores])},
            'rouge-2': {'f': np.mean([s['rouge-2']['f'] for s in scores])},
            'rouge-l': {'f': np.mean([s['rouge-l']['f'] for s in scores])}
        }
        return avg_scores
    return None

def evaluate_with_rouge(model, loader, tokenizer, device, generation_length=25):
    """Оценка модели с метрикой ROUGE"""
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for x_batch, y_batch in tqdm(loader, desc="ROUGE Evaluation"):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            # Для каждого примера в батче
            for i in range(x_batch.size(0)):
                # Берем 3/4 текста как контекст
                context_len = int(x_batch.size(1) * 0.75)
                context = x_batch[i, :context_len].tolist()
                
                # Генерируем продолжение (1/4 текста)
                generated = model.generate_text(
                    context,
                    max_length=generation_length,
                    temperature=0.8,
                    device=device,
                    tokenizer=tokenizer
                )
                
                # Целевая последовательность (оставшиеся 1/4)
                target = x_batch[i, context_len:].tolist()
                # Убираем паддинг
                target = [t for t in target if t != tokenizer.pad_token_id]
                
                all_predictions.append(generated)
                all_targets.append(target)
    
    # Вычисляем ROUGE
    rouge_scores = calculate_rouge(all_predictions, all_targets, tokenizer)
    return rouge_scores

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
    
    batch_size = 256  # Увеличиваем батч
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    print(f"Даталоадеры созданы: train={len(train_loader)} batches, val={len(val_loader)} batches")
    
    # 4. Создаем модель с большими размерами
    model = BiRNNClassifier(
        vocab_size, 
        hidden_dim=256,  # Увеличиваем скрытый слой
        rnn_type="LSTM", 
        combine="concat"
    )
    model = model.to(device)
    
    print(f"Параметров модели: {sum(p.numel() for p in model.parameters()):,}")
    
    # 5. Оптимизатор и функция потерь
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # 6. Функция оценки точности
    def evaluate_accuracy(model, loader):
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
    
    # 7. Обучение с метрикой ROUGE
    n_epochs = 10  # Увеличиваем эпохи
    
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.
        
        # Прогресс-бар
        train_iterator = tqdm(train_loader, desc=f'Epoch {epoch+1}/{n_epochs}')
        
        for x_batch, y_batch in train_iterator:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Клиппинг градиентов
            optimizer.step()
            
            train_loss += loss.item()
            train_iterator.set_postfix({'loss': f'{loss.item():.3f}'})
        
        train_loss /= len(train_loader)
        
        # Валидация точности
        val_loss, val_acc = evaluate_accuracy(model, val_loader)
        
        # Вычисляем ROUGE каждую 2-ю эпоху
        if (epoch + 1) % 2 == 0:
            print(f"Вычисление ROUGE метрики...")
            rouge_scores = evaluate_with_rouge(model, val_loader, tokenizer, device)
            
            if rouge_scores:
                print(f"Epoch {epoch+1} | Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.2%}")
                print(f"ROUGE-1 F1: {rouge_scores['rouge-1']['f']:.4f}")
                print(f"ROUGE-2 F1: {rouge_scores['rouge-2']['f']:.4f}")
                print(f"ROUGE-L F1: {rouge_scores['rouge-l']['f']:.4f}")
            else:
                print(f"Epoch {epoch+1} | Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.2%}")
                print("ROUGE: не удалось вычислить")
        else:
            print(f"Epoch {epoch+1} | Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.2%}")
        
        # Показываем примеры генерации каждую 3-ю эпоху
        if (epoch + 1) % 3 == 0:
            print("\nПримеры генерации:")
            test_prompts = [
                "I love this movie because",
                "The weather today is",
                "In the future, AI will",
            ]
            
            for prompt in test_prompts:
                input_ids = tokenizer.encode(prompt, add_special_tokens=False)
                generated = model.generate_text(
                    input_ids, 
                    max_length=20, 
                    temperature=0.8, 
                    device=device,
                    tokenizer=tokenizer
                )
                
                full_text = tokenizer.decode(input_ids + generated, skip_special_tokens=True)
                print(f"Промпт: '{prompt}'")
                print(f"Сгенерировано: '{full_text}'")
                print("-" * 50)
    
    # 8. Сохраняем модель
    os.makedirs("models", exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': vocab_size,
        'hidden_dim': 256,
        'rnn_type': 'LSTM',
        'combine': 'concat'
    }, "models/trained_model_with_rouge.pth")
    
    print("\nМодель сохранена в 'models/trained_model_with_rouge.pth'")

if __name__ == "__main__":
    main()