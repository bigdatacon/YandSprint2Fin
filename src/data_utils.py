import re
import pandas as pd
from tqdm import tqdm
import os

def clean_tweet(text):
    """
    Очистка и нормализация текста твита
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Привести к нижнему регистру
    text = text.lower()
    
    # 2. Удалить ссылки
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # 3. Удалить упоминания пользователей
    text = re.sub(r'@\w+', '', text)
    
    # 4. Удалить специальные символы и цифры (опционально)
    # Оставляем только буквы и пробелы
    text = re.sub(r'[^a-z\s]', ' ', text)

    # 4.5 Удалить хэштеги и символ # (важное исправление!)
    text = re.sub(r'#', '', text)
    
    # 5. Удалить эмодзи и специальные символы Unicode
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    
    # 6. Заменить множественные пробелы одним
    text = re.sub(r'\s+', ' ', text)
    
    # 7. Удалить пробелы в начале и конце
    text = text.strip()
    
    return text

def process_tweets_file(input_file, output_file):
    """
    Основная функция обработки файла с твитами
    """
    print(f"Чтение файла {input_file}...")
    
    # Чтение файла
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
        tweets = f.readlines()
    
    print(f"Загружено {len(tweets)} твитов")
    
    processed_texts = []
    
    # Обработка каждого твита с прогресс-баром
    for tweet in tqdm(tweets, desc="Обработка твитов"):
        cleaned_text = clean_tweet(tweet)
        
        # Пропускаем пустые строки после очистки
        if cleaned_text and len(cleaned_text) > 3:  # Минимальная длина
            processed_texts.append(cleaned_text)
    
    # Сохранение в CSV - просто текст, по одному тексту на строку
    with open(output_file, 'w', encoding='utf-8') as f:
        for text in processed_texts:
            f.write(text + '\n')
    
    # Вывод статистики
    print(f"\nСтатистика обработки:")
    print(f"Обработано твитов: {len(processed_texts)}")
    print(f"Сохранено в файл: {output_file}")
    
    if processed_texts:
        # Примеры очищенных твитов
        print("\nПримеры очищенных твитов:")
        for i in range(min(3, len(processed_texts))):
            print(f"{i+1}. {processed_texts[i][:100]}...")
    
    return processed_texts

# Основная часть скрипта
if __name__ == "__main__":
    # Получаем путь к директории, где находится текущий скрипт
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Поднимаемся на уровень выше (из src в корень проекта)
    project_root = os.path.dirname(script_dir)
    
    # Формируем пути
    input_file = os.path.join(project_root, "data", "tweets.txt")
    output_file = os.path.join(project_root, "data", "dataset_processed.txt")  # Меняем на .txt
    
    print(f"Путь к исходному файлу: {input_file}")
    print(f"Путь к выходному файлу: {output_file}")
    
    # Запуск обработки
    try:
        processed_texts = process_tweets_file(input_file, output_file)
        print("Обработка завершена успешно!")
        
    except FileNotFoundError:
        print(f"Ошибка: Файл {input_file} не найден.")
        print("Пожалуйста, убедитесь, что файл находится в той же директории.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")