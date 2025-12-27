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

def test_text_generation():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Загружаем модель
    model_path = "models/trained_model_with_rouge.pth"
    checkpoint = torch.load(model_path, map_location=device)
    
    tokenizer = BertTokenizerFast.from_pretrained("data/processed/tokenizer")
    
    model = BiRNNClassifier(
        vocab_size=checkpoint['vocab_size'],
        hidden_dim=checkpoint['hidden_dim'],
        rnn_type=checkpoint['rnn_type'],
        combine=checkpoint['combine']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print("\n" + "=" * 60)
    print("ТЕСТИРОВАНИЕ ГЕНЕРАЦИИ ТЕКСТА (ДОПИСЫВАНИЕ)")
    print("=" * 60)
    
    test_prompts = [
        "I love this movie because",
        "The weather today is",
        "In the future, AI will",
        "I think that",
        "Yesterday I went to",
    ]
    
    for prompt in test_prompts:
        print(f"\nПромпт: '{prompt}'")
        
        # Генерируем продолжение
        input_ids = tokenizer.encode(prompt, add_special_tokens=False)
        generated = model.generate_text(
            input_ids, 
            max_length=50, 
            temperature=0.8, 
            device=device,
            tokenizer=tokenizer
        )
        
        # Декодируем результат
        full_text = tokenizer.decode(input_ids + generated, skip_special_tokens=True)
        
        print(f"Сгенерированный текст ({len(generated)} токенов):")
        print(f"  '{full_text}'")
        
        # Показываем, где закончился промпт
        prompt_text = tokenizer.decode(input_ids, skip_special_tokens=True)
        generated_text = tokenizer.decode(generated, skip_special_tokens=True)
        print(f"  Промпт: '{prompt_text}'")
        print(f"  Дописано: '{generated_text}'")
        print("-" * 50)
    
    # Тест с реальными текстами из валидации
    print("\n" + "=" * 60)
    print("ТЕСТ С РЕАЛЬНЫМИ ТЕКСТАМИ (3/4 текста -> предсказание 1/4)")
    print("=" * 60)
    
    def load_texts(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f]
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, "data", "processed")
    
    val_texts = load_texts(os.path.join(data_dir, "val.txt"))
    
    for i, text in enumerate(val_texts[:3]):
        print(f"\nПример {i+1}:")
        print(f"  Полный текст: '{text[:100]}...'")
        
        # Токенизируем весь текст
        all_tokens = tokenizer.encode(text, add_special_tokens=False)
        
        if len(all_tokens) > 10:
            # Берем 3/4 текста как промпт
            context_len = int(len(all_tokens) * 0.75)
            context = all_tokens[:context_len]
            target = all_tokens[context_len:]  # Оставшиеся 1/4 (правильный ответ)
            
            # Генерируем продолжение
            generated = model.generate_text(
                context, 
                max_length=len(target) + 10,  # Немного больше для надежности
                temperature=0.8, 
                device=device,
                tokenizer=tokenizer
            )
            
            # Декодируем
            context_text = tokenizer.decode(context, skip_special_tokens=True)
            generated_text = tokenizer.decode(generated, skip_special_tokens=True)
            target_text = tokenizer.decode(target, skip_special_tokens=True)
            
            print(f"  Промпт (3/4): '{context_text}...'")
            print(f"  Сгенерировано (1/4): '{generated_text}'")
            print(f"  Правильный ответ: '{target_text}'")
            
            # Простая оценка (количество совпавших слов)
            generated_words = set(generated_text.lower().split())
            target_words = set(target_text.lower().split())
            common_words = generated_words.intersection(target_words)
            
            if target_words:
                overlap = len(common_words) / len(target_words)
                print(f"  Пересечение слов: {overlap:.1%}")
        else:
            print("  Текст слишком короткий для теста")
        
        print("-" * 50)

if __name__ == "__main__":
    test_text_generation()