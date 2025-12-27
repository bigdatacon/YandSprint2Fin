# test_generation.py
import torch
import torch.nn as nn
import os
from transformers import BertTokenizerFast
from lstm_model import BiRNNClassifier

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Загружаем модель
    model_path = "models/trained_model_with_rouge.pth"
    checkpoint = torch.load(model_path, map_location=device)
    
    # Загружаем токенизатор
    tokenizer_path = "data/processed/tokenizer"
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
    
    # Создаем модель
    model = BiRNNClassifier(
        vocab_size=checkpoint['vocab_size'],
        hidden_dim=checkpoint['hidden_dim'],
        rnn_type=checkpoint['rnn_type'],
        combine=checkpoint['combine']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print("=" * 60)
    print("ТЕСТИРОВАНИЕ ПРЕДСКАЗАНИЯ СЛЕДУЮЩЕГО ТОКЕНА")
    print("=" * 60)
    
    # Тестовые примеры
    test_cases = [
        ("I love this movie", "because"),
        ("The weather today", "is"),
        ("In the future", "AI"),
        ("Hello world", "this"),
        ("How are you", "doing"),
    ]
    
    for i, (prompt, correct_next) in enumerate(test_cases):
        print(f"\nПример {i+1}:")
        print(f"  Промт: '{prompt}'")
        
        # Токенизируем промпт с [CLS]
        input_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt").to(device)
        
        # Предсказываем следующий токен
        with torch.no_grad():
            outputs = model(input_ids)
            probs = torch.softmax(outputs, dim=-1)
            predicted_id = torch.argmax(probs, dim=-1).item()
        
        predicted_token = tokenizer.decode([predicted_id], skip_special_tokens=True)
        
        print(f"  Предикт Y: '{predicted_token}'")
        print(f"  Правильный ответ: '{correct_next}'")
        
        # Проверяем совпадение
        if predicted_token.lower() == correct_next.lower():
            print(f"  ✓ СОВПАДЕНИЕ!")
        else:
            print(f"  ✗ НЕ СОВПАЛО")
    
    print("\n" + "=" * 60)
    print("ТЕСТИРОВАНИЕ НА РЕАЛЬНЫХ ДАННЫХ ИЗ ВАЛИДАЦИИ")
    print("=" * 60)
    
    # Загружаем валидационные тексты для тестирования
    def load_texts(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f]
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, "data", "processed")
    
    val_texts = load_texts(os.path.join(data_dir, "val.txt"))
    
    # Тестируем на первых 5 текстах
    for i, text in enumerate(val_texts[:5]):
        print(f"\nПример {i+1} из валидации:")
        
        # Берем первые 5 слов как промпт
        words = text.split()
        if len(words) > 5:
            prompt = ' '.join(words[:5])
            correct_next = words[5] if len(words) > 5 else "[КОНЕЦ]"
        else:
            prompt = text
            correct_next = "[КОНЕЦ]"
        
        print(f"  Промт: '{prompt}'")
        
        # Токенизируем
        input_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt").to(device)
        
        # Предсказываем
        with torch.no_grad():
            outputs = model(input_ids)
            probs = torch.softmax(outputs, dim=-1)
            predicted_id = torch.argmax(probs, dim=-1).item()
        
        predicted_token = tokenizer.decode([predicted_id], skip_special_tokens=True)
        
        print(f"  Предикт Y: '{predicted_token}'")
        print(f"  Правильный ответ: '{correct_next}'")

if __name__ == "__main__":
    main()