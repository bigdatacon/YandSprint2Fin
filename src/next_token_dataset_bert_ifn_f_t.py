"""
next_token_dataset.py
Dataset для задачи автодополнения текста с использованием BERT токенизатора
"""
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from transformers import BertTokenizerFast
from tqdm import tqdm

class NextTokenDataset(Dataset):
    """
    Dataset для задачи предсказания следующего токена.
    
    Для последовательности токенов [CLS] w1 w2 w3 [SEP]:
    - X: [CLS] w1 w2 w3
    - Y: w1 w2 w3 [SEP]  (если ignore_first_token=False)
    - Y: -100 w2 w3 [SEP] (если ignore_first_token=True)
    
    Альтернативный подход (проще):
    - X: w1 w2 w3 (без [CLS])
    - Y: w2 w3 [SEP] (начинаем предсказания со второго токена)
    """
    def __init__(self, texts, tokenizer, max_len=50, ignore_first_token=True):
        """
        Args:
            texts: список текстов
            tokenizer: BERT токенизатор
            max_len: максимальная длина последовательности (включая спец. токены)
            ignore_first_token: если True, начинаем предсказания со второго токена после [CLS]
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.ignore_first_token = ignore_first_token
        
        # Предварительно токенизируем все тексты
        print("Токенизация текстов...")
        self.encodings = []
        for text in tqdm(texts):
            # Токенизируем текст
            tokens = self.tokenizer.encode(
                text,
                add_special_tokens=True,  # Добавляем [CLS] и [SEP]
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
        
        # Упрощенная логика: если игнорируем первый токен после [CLS],
        # просто начинаем на 1 токен позже
        if self.ignore_first_token and len(token_ids) > 2:
            # Проверяем, что первый токен действительно [CLS]
            if token_ids[0] == self.tokenizer.cls_token_id:
                # X начинаем с w1 (второй токен), Y начинаем с w2 (третий токен)
                # и обрезаем последний токен для X
                x_ids = token_ids[1:-1]  # w1 w2 w3
                y_ids = token_ids[2:]    # w2 w3 [SEP]
            else:
                # Если нет [CLS], используем стандартную логику
                x_ids = token_ids[:-1]   # все кроме последнего
                y_ids = token_ids[1:]    # все кроме первого
        else:
            # Стандартная логика: сдвиг на 1 токен
            x_ids = token_ids[:-1]  # [CLS] w1 w2 w3
            y_ids = token_ids[1:]   # w1 w2 w3 [SEP]
        
        # Создаем маску внимания (1 для реальных токенов, 0 для паддинга)
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

def prepare_datasets(data_path, train_size=0.8, val_size=0.1, test_size=0.1, max_len=50, ignore_first_token=True):
    """
    Подготавливает данные и создает Dataset'ы
    """
    # Загружаем данные
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
        texts = df['cleaned_text'].tolist()
    else:
        with open(data_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f]
    
    print(f"Загружено {len(texts)} текстов")
    
    # Инициализируем BERT токенизатор
    print("Загрузка BERT токенизатора...")
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    
    # Разделяем данные на train/val/test
    print("Разделение данных...")
    
    # Сначала разделим на train+val и test
    train_val_texts, test_texts = train_test_split(
        texts, 
        test_size=test_size, 
        random_state=42
    )
    
    # Затем train_val разделим на train и val
    val_ratio = val_size / (train_size + val_size)
    train_texts, val_texts = train_test_split(
        train_val_texts, 
        test_size=val_ratio, 
        random_state=42
    )
    
    print(f"Размеры выборок:")
    print(f"  Train: {len(train_texts)}")
    print(f"  Val: {len(val_texts)}")
    print(f"  Test: {len(test_texts)}")
    
    # Создаем Dataset'ы
    print("Создание Dataset'ов...")
    train_dataset = NextTokenDataset(train_texts, tokenizer, max_len, ignore_first_token)
    val_dataset = NextTokenDataset(val_texts, tokenizer, max_len, ignore_first_token)
    test_dataset = NextTokenDataset(test_texts, tokenizer, max_len, ignore_first_token)
    
    return train_dataset, val_dataset, test_dataset, tokenizer

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
        max_len=50,  # Максимальная длина последовательности
        ignore_first_token=True  # Игнорировать первый токен после [CLS]
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
    print("ДЕМОНСТРАЦИЯ РАБОТЫ (ignore_first_token=True):")
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
    
    print(f"\nЦелевая последовательность (Y):")
    label_tokens = []
    for label_id in labels:
        if label_id == -100:
            label_tokens.append("[PAD]")
        else:
            label_tokens.append(tokenizer.convert_ids_to_tokens([label_id])[0])
    
    print(f"  Токены: {label_tokens}")
    print(f"  Текст: {tokenizer.decode(labels[labels != -100], skip_special_tokens=True)}")
    
    # Проверяем смещение
    print(f"\nСравнение последовательностей:")
    print(f"  Исходный токенизированный текст: {tokenizer.convert_ids_to_tokens(train_dataset.encodings[0])}")
    
    print("\n" + "=" * 60)
    print("ПОДГОТОВКА ЗАВЕРШЕНА УСПЕШНО!")
    print("=" * 60)
    
    # Дополнительная проверка
    print("\n" + "=" * 60)
    print("ПРОВЕРКА ЛОГИКИ:")
    print("=" * 60)
    
    # Тест с игнорированием первого токена
    print("\n1. ignore_first_token=True:")
    test_texts = ["Hello world this is test"]
    test_dataset1 = NextTokenDataset(test_texts, tokenizer, max_len=10, ignore_first_token=True)
    item1 = test_dataset1[0]
    
    print(f"  Исходные токены: {tokenizer.convert_ids_to_tokens(test_dataset1.encodings[0])}")
    print(f"  X (input_ids): {tokenizer.convert_ids_to_tokens(item1['input_ids'])}")
    print(f"  Y (labels): {tokenizer.convert_ids_to_tokens(item1['labels'][item1['labels'] != -100])}")
    print(f"  Первый токен X: {tokenizer.convert_ids_to_tokens([item1['input_ids'][0]])[0]}")
    print(f"  Первый токен Y: {tokenizer.convert_ids_to_tokens([item1['labels'][0]])[0] if item1['labels'][0] != -100 else '[PAD]'}")
    
    # Тест без игнорирования первого токена
    print("\n2. ignore_first_token=False:")
    test_dataset2 = NextTokenDataset(test_texts, tokenizer, max_len=10, ignore_first_token=False)
    item2 = test_dataset2[0]
    
    print(f"  Исходные токены: {tokenizer.convert_ids_to_tokens(test_dataset2.encodings[0])}")
    print(f"  X (input_ids): {tokenizer.convert_ids_to_tokens(item2['input_ids'])}")
    print(f"  Y (labels): {tokenizer.convert_ids_to_tokens(item2['labels'][item2['labels'] != -100])}")
    print(f"  Первый токен X: {tokenizer.convert_ids_to_tokens([item2['input_ids'][0]])[0]}")
    print(f"  Первый токен Y: {tokenizer.convert_ids_to_tokens([item2['labels'][0]])[0] if item2['labels'][0] != -100 else '[PAD]'}")