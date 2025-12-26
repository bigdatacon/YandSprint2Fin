# импортируем библиотеки, которые пригодятся для задачи
import torch
import os
import torch.nn as nn

import re

import random

from datasets import load_dataset

from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset

from transformers import BertTokenizerFast
from transformers import GPT2Tokenizer  # Меняем на GPT-2

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
# seq_len = 7 => 3 токена до <MASK> + токен <MASK> + 3 токена после
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
class MaskedBertDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=512, ignore_first_token=True):
        # self.samples - список пар (x, y)
        # x - токенизированный текст с пропущенным токеном
        # y - пропущенный токен
        self.seq_len=seq_len
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.ignore_first_token = ignore_first_token
        self.samples = []

        print("Токенизация текстов...")
        for line in texts:
            token_ids = tokenizer.encode(line, add_special_tokens=True, max_length=self.max_len, truncation=True)
            for i in range(1, len(token_ids) - 1):
                context = token_ids[:i+1]  # все токены до текущей позиции
                target = token_ids[i+1]    # следующий токен
                
                # Добавляем в samples
                self.samples.append((context, target))
           
    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        x, y = self.samples[idx] # получите контекст и таргет для элемента с индексом idx
        return torch.tensor(x), torch.tensor(y)

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
    
    # Y - это просто отдельные токены, их не нужно паддить
    padded_y = torch.tensor(y_batch)
    
    return padded_x, padded_y

# загружаем токенизатор
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")


# тренировочный и валидационный датасеты
train_dataset = MaskedBertDataset(train_texts, tokenizer)
val_dataset = MaskedBertDataset(val_texts, tokenizer)


# даталоадеры
train_loader = DataLoader(
    train_dataset, 
    batch_size=64, 
    shuffle=True,
    collate_fn=collate_fn  # Добавляем collate_fn
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=64,
    collate_fn=collate_fn  # Добавляем collate_fn
)
print("DONE")
print(f"\nDataLoader'ы созданы:")
print(f"  Train: {len(train_dataset)} примеров")
print(f"  Val: {len(val_dataset)} примеров")



# Демонстрация работы
print("\n" + "=" * 60)
print("ДЕМОНСТРАЦИЯ РАБОТЫ:")
print("=" * 60)

# Получаем первый батч
batch = next(iter(train_loader))
x_batch, y_batch = batch  # Разделяем на x и y

print(f"\nРазмеры батча:")
print(f"  X (context): {x_batch.shape}")  # [batch_size, max_seq_len]
print(f"  Y (target): {y_batch.shape}")   # [batch_size] - один токен на пример

# Показываем несколько примеров
print(f"\nПервые 3 примера в батче:")
for example_idx in range(3):
    x = x_batch[example_idx]
    y = y_batch[example_idx]
    
    print(f"\nПример {example_idx}:")
    
    # Фильтруем паддинг для X
    x_filtered = x[x != tokenizer.pad_token_id] if tokenizer.pad_token_id is not None else x
    
    print(f"  X (контекст):")
    print(f"    Токены: {tokenizer.convert_ids_to_tokens(x_filtered)}")
    print(f"    Текст: {tokenizer.decode(x_filtered, skip_special_tokens=False)}")
    
    print(f"  Y (цель - следующий токен):")
    # y - это скаляр (один токен), оборачиваем его в список
    print(f"    Токен: {tokenizer.convert_ids_to_tokens([y.item()])[0]}")
    print(f"    ID: {y.item()}")
    
    # Проверяем логику: какой последний токен в X и что должен быть Y
    if len(x_filtered) > 0:
        last_x_token = tokenizer.convert_ids_to_tokens([x_filtered[-1].item()])[0]
        print(f"  Проверка: Последний токен X: '{last_x_token}' → Предсказанный Y: '{tokenizer.convert_ids_to_tokens([y.item()])[0]}'")

# Проверяем общую статистику
print(f"\nОбщая статистика:")
print(f"  Всего примеров в train: {len(train_dataset)}")
print(f"  Всего примеров в val: {len(val_dataset)}")
print(f"  Примерное число примеров на текст: {len(train_dataset) / len(train_texts):.1f}")

# Проверяем работу на тестовом примере
print("\n" + "=" * 60)
print("ТЕСТОВАЯ ПРОВЕРКА ЛОГИКИ:")
print("=" * 60)

test_text = "Hello world this is a test"
print(f"\nТестовый текст: '{test_text}'")

test_tokens = tokenizer.encode(test_text, add_special_tokens=True)
print(f"Токены с спецсимволами: {tokenizer.convert_ids_to_tokens(test_tokens)}")
print(f"ID токенов: {test_tokens}")

print(f"\nПримеры из датасета для этого текста:")
test_texts = [test_text]
test_dataset = MaskedBertDataset(test_texts, tokenizer)

for i in range(min(5, len(test_dataset))):
    x, y = test_dataset[i]
    
    print(f"\nПример {i}:")
    print(f"  X (контекст): {tokenizer.convert_ids_to_tokens(x)}")
    print(f"  Y (цель): {tokenizer.convert_ids_to_tokens([y.item()])[0]}")
    
    # Проверяем логику
    if len(x) > 0:
        print(f"  Последний токен X: {tokenizer.convert_ids_to_tokens([x[-1].item()])[0]}")
        print(f"  Предсказанный Y: {tokenizer.convert_ids_to_tokens([y.item()])[0]}")

# Демонстрация генерации текста
print("\n" + "=" * 60)
print("КАК БУДЕТ РАБОТАТЬ ГЕНЕРАЦИЯ:")
print("=" * 60)

# Симуляция работы модели
def simulate_generation(prompt, num_steps=5):
    print(f"\nПромпт: '{prompt}'")
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    
    print("Шаги генерации:")
    for step in range(num_steps):
        context = input_ids  # Текущий контекст
        # В реальности модель бы предсказала следующий токен
        # Здесь просто берем следующий токен из исходной последовательности для демонстрации
        if step < len(test_tokens) - 1:
            next_token = test_tokens[step + 1]
            input_ids.append(next_token)
            
            print(f"  Шаг {step+1}:")
            print(f"    Контекст: {tokenizer.convert_ids_to_tokens(context)}")
            print(f"    Предсказанный токен: {tokenizer.convert_ids_to_tokens([next_token])[0]}")
            print(f"    Новый текст: {tokenizer.decode(input_ids, skip_special_tokens=True)}")
        else:
            print("  Достигнут конец последовательности ([SEP])")
            break

simulate_generation("Hello world")

print("\n" + "=" * 60)
print("ПОДГОТОВКА ЗАВЕРШЕНА УСПЕШНО!")
print("=" * 60)