"""
next_token_dataset.py
Dataset для задачи автодополнения с использованием GPT-2 токенизатора
"""
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from transformers import GPT2Tokenizer  # Меняем на GPT-2
import os
from tqdm import tqdm

class NextTokenDataset(Dataset):
    """
    Dataset для задачи предсказания следующего токена.
    
    GPT-2 не использует [CLS], только <|endoftext|> для обозначения конца
    """
    def __init__(self, texts, tokenizer, max_len=100):
        """
        Args:
            texts: список текстов
            tokenizer: GPT-2 токенизатор
            max_len: максимальная длина последовательности
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # Предварительно токенизируем все тексты
        print("Токенизация текстов...")
        self.encodings = []
        for text in tqdm(texts):
            # GPT-2 добавляет только <|endoftext|> в конце
            tokens = self.tokenizer.encode(
                text,
                max_length=self.max_len,
                truncation=True,
                return_tensors='pt'
            )
            self.encodings.append(tokens[0])
    
    def __len__(self):
        return len(self.encodings)
    
    def __getitem__(self, idx):
        # Получаем токенизированную последовательность
        token_ids = self.encodings[idx]
        
        # Для next token prediction:
        # X: все токены, кроме последнего
        # Y: все токены, кроме первого
        
        # GPT-2 подход: учимся предсказывать следующий токен
        x_ids = token_ids[:-1]  # все кроме последнего
        y_ids = token_ids[1:]   # все кроме первого (смещение на 1)
        
        # Маска внимания
        attention_mask = (x_ids != self.tokenizer.pad_token_id).long()
        
        return {
            'input_ids': x_ids,
            'attention_mask': attention_mask,
            'labels': y_ids
        }

def collate_fn(batch):
    """
    Функция для объединения примеров в батч
    """
    # Собираем все элементы батча
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    # Дополняем последовательности до одинаковой длины
    padded_input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, 
        batch_first=True, 
        padding_value=0  # pad_token_id обычно 0 у BERT
    )
    
    padded_attention_masks = torch.nn.utils.rnn.pad_sequence(
        attention_masks, 
        batch_first=True, 
        padding_value=0
    )
    
    padded_labels = torch.nn.utils.rnn.pad_sequence(
        labels, 
        batch_first=True, 
        padding_value=-100  # -100 игнорируется при вычислении потерь
    )
    
    return {
        'input_ids': padded_input_ids,
        'attention_mask': padded_attention_masks,
        'labels': padded_labels
    }

def prepare_datasets(data_path, train_size=0.8, val_size=0.1, test_size=0.1, max_len=50):
    """
    Подготавливает данные с GPT-2 токенизатором
    """
    # Загружаем данные
    texts = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            text = line.strip()
            if text:
                texts.append(text)
    
    print(f"Загружено {len(texts)} текстов")
    
    # Инициализируем GPT-2 токенизатор
    print("Загрузка GPT-2 токенизатора...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Добавляем pad token (у GPT-2 его нет по умолчанию)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    # Разделяем данные
    train_val_texts, test_texts = train_test_split(texts, test_size=test_size, random_state=42)
    val_ratio = val_size / (train_size + val_size)
    train_texts, val_texts = train_test_split(train_val_texts, test_size=val_ratio, random_state=42)
    
    print(f"Размеры выборок: Train={len(train_texts)}, Val={len(val_texts)}, Test={len(test_texts)}")
    
    # Создаем Dataset'ы
    train_dataset = NextTokenDataset(train_texts, tokenizer, max_len)
    val_dataset = NextTokenDataset(val_texts, tokenizer, max_len)
    test_dataset = NextTokenDataset(test_texts, tokenizer, max_len)
    
    return train_dataset, val_dataset, test_dataset, tokenizer

# ... остальной код остается без изменений ...
def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=32):
    """
    Создает DataLoader'ы для train/val/test
    """
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader, test_loader

# Пример использования
if __name__ == "__main__":
    import os
    
    # Пути к файлам
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, "data", "dataset_processed.txt")
    
    # Проверяем существование файла
    if not os.path.exists(data_path):
        print(f"Ошибка: файл {data_path} не найден!")
        print("Сначала выполните очистку данных: python data_utils.py")
        exit(1)
    
    print("=" * 60)
    print("ПОДГОТОВКА ДАННЫХ ДЛЯ ЗАДАЧИ АВТОДОПОЛНЕНИЯ")
    print("=" * 60)
    
    # Подготавливаем данные
    train_dataset, val_dataset, test_dataset, tokenizer = prepare_datasets(
        data_path=data_path,
        train_size=0.8,
        val_size=0.1,
        test_size=0.1,
        max_len=50  # Максимальная длина последовательности
    )
    
    # Создаем DataLoader'ы
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset, batch_size=32
    )
    
    print(f"\nDataLoader'ы созданы:")
    print(f"  Train: {len(train_loader.dataset)} примеров")
    print(f"  Val: {len(val_loader.dataset)} примеров")
    print(f"  Test: {len(test_loader.dataset)} примеров")
    
    # Демонстрация работы
    print("\n" + "=" * 60)
    print("ДЕМОНСТРАЦИЯ РАБОТЫ:")
    print("=" * 60)
    
    # Получаем первый батч
    batch = next(iter(train_loader))
    
    print(f"\nРазмеры батча:")
    print(f"  input_ids: {batch['input_ids'].shape}")
    print(f"  attention_mask: {batch['attention_mask'].shape}")
    print(f"  labels: {batch['labels'].shape}")
    
    # Показываем пример
    print(f"\nПример 0 в батче:")
    input_ids = batch['input_ids'][0]
    labels = batch['labels'][0]
    
    print(f"\nВходная последовательность (X):")
    print(f"  Токены: {tokenizer.convert_ids_to_tokens(input_ids)}")
    print(f"  Текст: {tokenizer.decode(input_ids, skip_special_tokens=True)}")
    
    print(f"\nЦелевая последовательность (Y - смещенная на 1 токен):")
    # Фильтруем -100 (игнорируемые токены)
    label_tokens = [tokenizer.convert_ids_to_tokens([label_id])[0] 
                   for label_id in labels if label_id != -100]
    print(f"  Токены: {label_tokens}")
    
    # Проверяем смещение
    print(f"\nПроверка смещения (первые 5 токенов):")
    for i in range(min(5, len(input_ids))):
        input_token = tokenizer.convert_ids_to_tokens([input_ids[i]])[0] if input_ids[i] != 0 else "[PAD]"
        label_token = tokenizer.convert_ids_to_tokens([labels[i]])[0] if labels[i] != -100 else "[IGNORE]"
        print(f"  X[{i}] = {input_token} → Y[{i}] = {label_token}")
    
    print("\n" + "=" * 60)
    print("ПОДГОТОВКА ЗАВЕРШЕНА УСПЕШНО!")
    print("=" * 60)