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
    def __init__(self, texts, tokenizer, max_len=512, ignore_first_token=True, seq_len=7):
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
            if len(token_ids) < self.seq_len:
                continue
            for i in range(1, len(token_ids) - 1):
                context = token_ids[i:-1]
                if len(context) < self.seq_len:
                    continue
                target = token_ids[i+1:]
                self.samples.append((context, target))
           
    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        x, y = x, y = self.samples[idx] # получите контекст и таргет для элемента с индексом idx
        return torch.tensor(x), torch.tensor(y)


# загружаем токенизатор
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")


# тренировочный и валидационный датасеты
train_dataset = MaskedBertDataset(train_texts, tokenizer, seq_len=seq_len)
val_dataset = MaskedBertDataset(val_texts, tokenizer, seq_len=seq_len)


# даталоадеры
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
print("DONE")