# импортируем библиотеки, которые пригодятся для задачи
import torch
import os
import torch.nn as nn
import pandas as pd
import re
import random
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast
from transformers import GPT2Tokenizer
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Пути к файлам
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_path = os.path.join(project_root, "data", "dataset_processed.txt")
random.seed(42)
torch.manual_seed(42)

# Загружаем данные
if data_path.endswith('.csv'):
    df = pd.read_csv(data_path)
    dataset = df['cleaned_text'].tolist()
else:
    with open(data_path, 'r', encoding='utf-8') as f:
        dataset = [line.strip() for line in f]

print(f"Загружено {len(dataset)} текстов")

# длины последовательностей в датасете
seq_len = 7

# удаляем слишком короткие тексты
cleaned_texts = [line for line in dataset if len(line.split()) >= seq_len]

# для упрощения используем только max_texts_count текстов
max_texts_count = 7000

# разбиение на тренировочную и валидационную выборки
val_size = 0.05

train_texts, val_texts = train_test_split(cleaned_texts[:max_texts_count], test_size=val_size, random_state=42)
print(f"Train texts: {len(train_texts)}, Val texts: {len(val_texts)}")

# класс датасета
class NextTokenDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []
        self.original_texts = []  # Сохраняем оригинальные тексты
        self.sample_to_text_idx = []  # Сохраняем индекс текста для каждого примера

        print("Токенизация текстов...")
        for text_idx, line in enumerate(tqdm(texts)):
            token_ids = tokenizer.encode(line, add_special_tokens=True, max_length=self.max_len, truncation=True)
            # Создаем пары для каждого токена в последовательности
            for i in range(1, len(token_ids) - 1):
                context = token_ids[1:i+1]
                target = token_ids[i+1]
                self.samples.append((context, target))
                self.original_texts.append(line)  # Сохраняем оригинальный текст
                self.sample_to_text_idx.append(text_idx)  # Сохраняем индекс текста
           
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y)
    
    def get_original_text(self, idx):
        """Возвращает оригинальный текст для примера с индексом idx"""
        return self.original_texts[idx]
    
    def get_text_index(self, idx):
        """Возвращает индекс текста в исходном списке"""
        return self.sample_to_text_idx[idx]

def collate_fn(batch):
    """
    Функция для объединения примеров в батч.
    """
    x_batch = [item[0] for item in batch]
    y_batch = [item[1] for item in batch]
    
    # Дополняем X до одинаковой длины
    padded_x = torch.nn.utils.rnn.pad_sequence(
        x_batch, 
        batch_first=True, 
        padding_value=0
    )
    
    # Y - это просто отдельные токены
    padded_y = torch.stack(y_batch)
    
    return padded_x, padded_y

# загружаем токенизатор
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# тренировочный и валидационный датасеты
train_dataset = NextTokenDataset(train_texts, tokenizer)
val_dataset = NextTokenDataset(val_texts, tokenizer)

print("DONE")
print(f"\nDataLoader'ы созданы:")
print(f"  Train: {len(train_dataset)} примеров")
print(f"  Val: {len(val_dataset)} примеров")
print(f"  Примерное число примеров на текст: {len(train_dataset) / len(train_texts):.1f}")

# даталоадеры
train_loader = DataLoader(
    train_dataset, 
    batch_size=64, 
    shuffle=True,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=64,
    collate_fn=collate_fn
)

# Демонстрация работы с выводом оригинального текста
print("\n" + "=" * 60)
print("ДЕМОНСТРАЦИЯ РАБОТЫ С ОРИГИНАЛЬНЫМИ ТЕКСТАМИ:")
print("=" * 60)

# Получаем первый батч
batch = next(iter(train_loader))
x_batch, y_batch = batch

print(f"\nРазмеры батча:")
print(f"  X (context): {x_batch.shape}")
print(f"  Y (target): {y_batch.shape}")

# Показываем несколько примеров с оригинальным текстом
print(f"\nПервые 5 примеров в батче:")
for example_idx in range(5):
    x = x_batch[example_idx]
    y = y_batch[example_idx]
    
    # Получаем индекс текста в датасете
    sample_idx_in_dataset = example_idx  # Это работает только для первого батча
    # В реальности нужно получить индекс из dataloader, но это сложно
    # Вместо этого покажем примеры из тестового датасета
    
    print(f"\nПример {example_idx}:")
    
    # Фильтруем паддинг для X
    x_filtered = x[x != tokenizer.pad_token_id]
    
    print(f"  X (контекст):")
    print(f"    Токены: {tokenizer.convert_ids_to_tokens(x_filtered)}")
    print(f"    Текст: {tokenizer.decode(x_filtered, skip_special_tokens=False)}")
    
    print(f"  Y (цель - следующий токен):")
    print(f"    Токен: {tokenizer.convert_ids_to_tokens([y.item()])[0]}")
    
    # Проверяем логику
    if len(x_filtered) > 0:
        last_x_token = tokenizer.convert_ids_to_tokens([x_filtered[-1].item()])[0]
        print(f"  Проверка: Последний токен X: '{last_x_token}' → Предсказанный Y: '{tokenizer.convert_ids_to_tokens([y.item()])[0]}'")

# Теперь покажем примеры из тестового текста с оригинальным текстом
print("\n" + "=" * 60)
print("ТЕСТОВАЯ ПРОВЕРКА ЛОГИКИ С ОРИГИНАЛЬНЫМ ТЕКСТОМ:")
print("=" * 60)

test_text = "Hello world this is a test"
print(f"\nОригинальный текст: '{test_text}'")

test_texts = [test_text]
test_dataset = NextTokenDataset(test_texts, tokenizer)

print(f"\nВсе примеры из этого текста ({len(test_dataset)} примеров):")

for i in range(len(test_dataset)):
    x, y = test_dataset[i]
    
    print(f"\nПример {i}:")
    print(f"  Оригинальный текст: '{test_dataset.get_original_text(i)}'")
    print(f"  X (контекст): {tokenizer.convert_ids_to_tokens(x)}")
    print(f"  Y (цель): {tokenizer.convert_ids_to_tokens([y.item()])[0]}")
    
    # Покажем, как это соответствует оригинальному тексту
    if len(x) > 0:
        context_text = tokenizer.decode(x, skip_special_tokens=True)
        print(f"  Контекст как текст: '{context_text}'")
        print(f"  Ожидаемое продолжение в оригинале: '{tokenizer.decode([y.item()], skip_special_tokens=True)}'")

# Правильная демонстрация генерации
print("\n" + "=" * 60)
print("ПРАВИЛЬНАЯ ДЕМОНСТРАЦИЯ ГЕНЕРАЦИИ:")
print("=" * 60)

def demonstrate_generation(prompt, tokenizer, max_steps=20):
    print(f"\nПромпт: '{prompt}'")
    
    # Токенизируем промпт
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    print(f"Токены с [CLS] и [SEP]: {tokenizer.convert_ids_to_tokens(input_ids)}")
    
    print("\nКак модель учится на этом тексте:")
    
    # Покажем все обучающие примеры из этого промпта
    for i in range(len(input_ids) - 1):
        context = input_ids[:i+1]
        target = input_ids[i+1]
        
        print(f"\n  Пример {i+1}:")
        print(f"    Контекст: {tokenizer.convert_ids_to_tokens(context)}")
        print(f"    Цель: {tokenizer.convert_ids_to_tokens([target])[0]}")
        
        if target == tokenizer.sep_token_id:
            print(f"    → Модель учится, что после '{tokenizer.decode(context[1:], skip_special_tokens=True)}' должен быть [SEP]")
    
    print("\nКак будет работать генерация в реальности:")
    print("(В реальности модель предсказывает следующий токен на основе текущего контекста)")
    
    # Симуляция работы обученной модели
    generated_ids = [tokenizer.cls_token_id]
    generated_tokens = [tokenizer.convert_ids_to_tokens([generated_ids[0]])[0]]
    
    # Декодируем промпт без [CLS] для начала
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    generated_ids.extend(prompt_ids)
    generated_tokens.extend(tokenizer.convert_ids_to_tokens(prompt_ids))
    
    print(f"\n  Начальное состояние:")
    print(f"    Токены: {generated_tokens}")
    print(f"    Текст: '{tokenizer.decode(generated_ids, skip_special_tokens=True)}'")
    
    # "Предсказываем" следующие токены (в реальности это делала бы модель)
    for step in range(max_steps):
        # В идеальной модели, если бы она идеально выучила текст:
        # После полного текста должен быть [SEP]
        if len(generated_ids) >= len(input_ids):
            next_token = tokenizer.sep_token_id
        else:
            next_token = input_ids[len(generated_ids)]
        
        generated_ids.append(next_token)
        generated_tokens.append(tokenizer.convert_ids_to_tokens([next_token])[0])
        
        print(f"\n  Шаг {step+1}:")
        print(f"    Предсказанный токен: {tokenizer.convert_ids_to_tokens([next_token])[0]}")
        print(f"    Текущий текст: '{tokenizer.decode(generated_ids, skip_special_tokens=True)}'")
        
        if next_token == tokenizer.sep_token_id:
            print(f"    → Модель предсказала [SEP], генерация завершена!")
            break

# Демонстрация
demonstrate_generation("Hello world this is a test", tokenizer)

print("\n" + "=" * 60)
print("ПОДГОТОВКА ЗАВЕРШЕНА УСПЕШНО!")
print("=" * 60)