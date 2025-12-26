# импортируем библиотеки, которые пригодятся для задачи
import torch
import os
import torch.nn as nn
import pandas as pd  # Добавляем импорт pandas
import re
import random
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
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

print(f"После фильтрации осталось {len(cleaned_texts)} текстов")

# для упрощения используем только max_texts_count текстов
max_texts_count = 7000

# Разделяем данные на train/val/test
test_size = 0.1  # 10% на тест
val_size = 0.1   # 10% на валидацию от оставшихся после теста

print(f"\nРазделение данных на train/val/test...")

# Сначала разделим на train+val и test
train_val_texts, test_texts = train_test_split(
    cleaned_texts[:max_texts_count], 
    test_size=test_size, 
    random_state=42
)

# Затем train_val разделим на train и val
val_ratio = val_size / (1 - test_size)  # val_size от train_val_texts
train_texts, val_texts = train_test_split(
    train_val_texts, 
    test_size=val_ratio, 
    random_state=42
)

print(f"Размеры выборок:")
print(f"  Train: {len(train_texts)} текстов")
print(f"  Val: {len(val_texts)} текстов")  
print(f"  Test: {len(test_texts)} текстов")
print(f"  Всего: {len(train_texts) + len(val_texts) + len(test_texts)} текстов")

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

# тренировочный, валидационный и тестовый датасеты
print("\nСоздание тренировочного датасета...")
train_dataset = NextTokenDataset(train_texts, tokenizer)

print("\nСоздание валидационного датасета...")
val_dataset = NextTokenDataset(val_texts, tokenizer)

print("\nСоздание тестового датасета...")
test_dataset = NextTokenDataset(test_texts, tokenizer)

print("DONE")
print(f"\nDataLoader'ы созданы:")
print(f"  Train: {len(train_dataset)} примеров")
print(f"  Val: {len(val_dataset)} примеров")
print(f"  Test: {len(test_dataset)} примеров")
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

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    collate_fn=collate_fn
)

# Дополнительная функция для сохранения данных
def save_datasets(train_texts, val_texts, test_texts, tokenizer, save_dir="data/processed"):
    """Сохраняет разделенные данные и токенизатор"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Сохраняем тексты
    with open(os.path.join(save_dir, "train.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(train_texts))
    
    with open(os.path.join(save_dir, "val.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(val_texts))
        
    with open(os.path.join(save_dir, "test.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(test_texts))
    
    # Сохраняем токенизатор
    tokenizer.save_pretrained(os.path.join(save_dir, "tokenizer"))
    
    print(f"\nДанные сохранены в папку: {save_dir}")
    print(f"  train.txt: {len(train_texts)} текстов")
    print(f"  val.txt: {len(val_texts)} текстов")
    print(f"  test.txt: {len(test_texts)} текстов")

# Сохраняем данные для использования в других файлах
save_datasets(train_texts, val_texts, test_texts, tokenizer)

# Демонстрация работы
print("\n" + "=" * 60)
print("ДЕМОНСТРАЦИЯ РАБОТЫ:")
print("=" * 60)

# Получаем первый батч из тренировочного датасета
batch = next(iter(train_loader))
x_batch, y_batch = batch

print(f"\nРазмеры тренировочного батча:")
print(f"  X (context): {x_batch.shape}")
print(f"  Y (target): {y_batch.shape}")

# Проверяем работу на тестовом примере
print("\n" + "=" * 60)
print("ТЕСТОВАЯ ПРОВЕРКА ЛОГИКИ:")
print("=" * 60)

test_text = "Hello world this is a test"
print(f"\nТестовый текст: '{test_text}'")

test_tokens = tokenizer.encode(test_text, add_special_tokens=True)
print(f"Токены с спецсимволами: {tokenizer.convert_ids_to_tokens(test_tokens)}")

test_texts = [test_text]
test_small_dataset = NextTokenDataset(test_texts, tokenizer)

print(f"\nВсе примеры из этого текста ({len(test_small_dataset)} примеров):")

for i in range(min(7, len(test_small_dataset))):
    x, y = test_small_dataset[i]
    
    print(f"\nПример {i}:")
    print(f"  X (контекст): {tokenizer.convert_ids_to_tokens(x)}")
    print(f"  Y (цель): {tokenizer.convert_ids_to_tokens([y.item()])[0]}")
    
print("\n" + "=" * 60)
print("ПОДГОТОВКА ДАННЫХ ЗАВЕРШЕНА УСПЕШНО!")
print("=" * 60)