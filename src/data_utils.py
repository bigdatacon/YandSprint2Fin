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
    
    # 5. Удалить эмодзи и специальные символы Unicode
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    
    # 6. Заменить множественные пробелы одним
    text = re.sub(r'\s+', ' ', text)
    
    # 7. Удалить пробелы в начале и конце
    text = text.strip()
    
    return text

def tokenize_text(text):
    """
    Простая токенизация по пробелам
    """
    return text.split()

def process_tweets_file(input_file, output_file):
    """
    Основная функция обработки файла с твитами
    """
    print(f"Чтение файла {input_file}...")
    
    # Чтение файла
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
        tweets = f.readlines()
    
    print(f"Загружено {len(tweets)} твитов")
    
    processed_data = []
    
    # Обработка каждого твита с прогресс-баром
    for tweet in tqdm(tweets, desc="Обработка твитов"):
        cleaned_text = clean_tweet(tweet)
        
        # Пропускаем пустые строки после очистки
        if cleaned_text and len(cleaned_text) > 3:  # Минимальная длина
            tokens = tokenize_text(cleaned_text)
            
            # Сохраняем как очищенный текст и как список токенов
            processed_data.append({
                'original': tweet.strip()[:100],  # Первые 100 символов оригинала
                'cleaned_text': cleaned_text,
                'tokens': tokens,
                'token_count': len(tokens)
            })
    
    # Создание DataFrame
    df = pd.DataFrame(processed_data)
    
    # Сохранение в CSV
    df.to_csv(output_file, index=False, encoding='utf-8')
    
    # Вывод статистики
    print(f"\nСтатистика обработки:")
    print(f"Обработано твитов: {len(processed_data)}")
    print(f"Сохранено в файл: {output_file}")
    
    if not df.empty:
        avg_tokens = df['token_count'].mean()
        print(f"Среднее количество токенов на твит: {avg_tokens:.2f}")
        
        # Примеры очищенных твитов
        print("\nПримеры очищенных твитов:")
        for i, row in df.head(3).iterrows():
            print(f"{i+1}. Оригинал: {row['original']}...")
            print(f"   Очищенный: {row['cleaned_text']}")
            print(f"   Токены: {row['tokens']}")
            print()
    
    return df

# Основная часть скрипта
if __name__ == "__main__":
    # Файлы
    # input_file = "tweets.txt"
    # output_file = "dataset_processed.csv"
        # Получаем путь к директории, где находится текущий скрипт
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Поднимаемся на уровень выше (из src в корень проекта)
    project_root = os.path.dirname(script_dir)
    
    # Формируем пути
    input_file = os.path.join(project_root, "data", "tweets.txt")
    output_file = os.path.join(project_root, "data", "dataset_processed.csv")
    
    print(f"Путь к исходному файлу: {input_file}")
    print(f"Путь к выходному файлу: {output_file}")
    
    # Запуск обработки
    try:
        df_processed = process_tweets_file(input_file, output_file)
        print("Обработка завершена успешно!")
        
    except FileNotFoundError:
        print(f"Ошибка: Файл {input_file} не найден.")
        print("Пожалуйста, убедитесь, что файл находится в той же директории.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")