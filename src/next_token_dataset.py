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
            token_ids = tokenizer.encode(line, add_special_tokens=False, max_length=self.max_len, truncation=True)
            context = token_ids[1:-1]
            target = token_ids[2:]
            self.samples.append((context, target))
           
    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        x, y = self.samples[idx] # получите контекст и таргет для элемента с индексом idx
        return torch.tensor(x), torch.tensor(y)

def collate_fn(batch):
    """
    Функция для объединения примеров в батч.
    batch - список кортежей (x, y)
    """
    # Разделяем x и y
    x_batch = [item[0] for item in batch]
    y_batch = [item[1] for item in batch]
    
    # Дополняем последовательности до одинаковой длины
    padded_x = torch.nn.utils.rnn.pad_sequence(
        x_batch, 
        batch_first=True, 
        padding_value=0  # pad_token_id
    )
    
    padded_y = torch.nn.utils.rnn.pad_sequence(
        y_batch, 
        batch_first=True, 
        padding_value=0  # pad_token_id
    )
    
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
print(f"  X (context): {x_batch.shape}")
print(f"  Y (target): {y_batch.shape}")

# Показываем несколько примеров
print(f"\nПервые 3 примера в батче:")
for example_idx in range(3):
    x = x_batch[example_idx]
    y = y_batch[example_idx]
    
    print(f"\nПример {example_idx}:")
    
    # Фильтруем паддинг
    x_filtered = x[x != tokenizer.pad_token_id] if tokenizer.pad_token_id is not None else x
    y_filtered = y[y != tokenizer.pad_token_id] if tokenizer.pad_token_id is not None else y
    
    print(f"  X (контекст):")
    print(f"    Токены: {tokenizer.convert_ids_to_tokens(x_filtered)}")
    print(f"    Текст: {tokenizer.decode(x_filtered, skip_special_tokens=True)}")
    
    print(f"  Y (цель):")
    print(f"    Токены: {tokenizer.convert_ids_to_tokens(y_filtered)}")
    print(f"    Текст: {tokenizer.decode(y_filtered, skip_special_tokens=True)}")
    
    # Проверяем логику
    print(f"  Проверка: Длина X: {len(x_filtered)}, Длина Y: {len(y_filtered)}")

# Проверяем общую статистику
print(f"\nОбщая статистика:")
print(f"  Всего примеров в train: {len(train_dataset)}")
print(f"  Всего примеров в val: {len(val_dataset)}")

# Проверяем работу на тестовом примере
print("\n" + "=" * 60)
print("ТЕСТОВАЯ ПРОВЕРКА ЛОГИКИ:")
print("=" * 60)

test_text = "Hello world this is a test sentence for checking"
print(f"\nТестовый текст: '{test_text}'")

test_tokens = tokenizer.encode(test_text, add_special_tokens=False)
print(f"Токены: {tokenizer.convert_ids_to_tokens(test_tokens)}")
print(f"ID токенов: {test_tokens}")

print(f"\nПримеры из датасета для этого текста:")
test_texts = [test_text]
test_dataset = MaskedBertDataset(test_texts, tokenizer)

for i in range(min(3, len(test_dataset))):
    x, y = test_dataset[i]
    
    print(f"\nПример {i}:")
    print(f"  X: {tokenizer.convert_ids_to_tokens(x)}")
    print(f"  Y: {tokenizer.convert_ids_to_tokens(y)}")
    
    # Проверяем логику
    if len(x) > 0 and len(y) > 0:
        print(f"  Первый токен X: {tokenizer.convert_ids_to_tokens([x[0]])[0]}")
        print(f"  Первый токен Y: {tokenizer.convert_ids_to_tokens([y[0]])[0]}")
    
print("\n" + "=" * 60)
print("ПОДГОТОВКА ЗАВЕРШЕНА УСПЕШНО!")
print("=" * 60)