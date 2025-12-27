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

def process_tweets_file(input_file, output_file, max_tweets=100000, random_sample=False):
    """
    Основная функция обработки файла с твитами с ограничением
    
    Args:
        input_file: входной файл с твитами
        output_file: выходной файл
        max_tweets: максимальное количество твитов для обработки
        random_sample: если True - случайная выборка, если False - первые N
    """
    print(f"Чтение файла {input_file}...")
    
    if random_sample:
        # Читаем все строки для случайной выборки
        with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
            all_tweets = f.readlines()
        
        print(f"Всего твитов в файле: {len(all_tweets)}")
        
        if len(all_tweets) > max_tweets:
            tweets = random.sample(all_tweets, max_tweets)
            print(f"Взята случайная выборка из {max_tweets} твитов")
        else:
            tweets = all_tweets
            print(f"Используются все {len(tweets)} твитов")
    else:
        # Читаем только первые N строк
        tweets = []
        with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f):
                if i >= max_tweets:
                    break
                tweets.append(line)
        
        print(f"Загружено {len(tweets)} твитов (первые {max_tweets})")
    
    processed_texts = []
    
    # Обработка каждого твита с прогресс-баром
    for tweet in tqdm(tweets, desc="Обработка твитов"):
        cleaned_text = clean_tweet(tweet)
        
        if cleaned_text and len(cleaned_text) > 3:
            processed_texts.append(cleaned_text)
    
    # Сохранение
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for text in processed_texts:
            f.write(text + '\n')
    
    print(f"\nСтатистика:")
    print(f"Обработано твитов: {len(processed_texts)}")
    print(f"Сохранено в: {output_file}")
    
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