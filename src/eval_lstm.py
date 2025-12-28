import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
from lstm_model import BiRNNClassifier
import os
from tqdm import tqdm
import evaluate
import numpy as np

# Загружаем метрику ROUGE
rouge_metric = evaluate.load("rouge")


def generate_and_evaluate(model, tokenizer, text, device):
    """
    Генерация и вычисление ROUGE для одного текста:
    - один токен
    - 1/4 текста
    """
    tokens = tokenizer.encode(text, add_special_tokens=True)
    if len(tokens) < 2:
        return None

    # 1️⃣ Генерация одного токена
    context_one_token = tokens[:-1]
    target_one_token = [tokens[-1]]
    generated_one = model.generate_text(
        context_one_token,
        max_new_tokens=1,
        temperature=0.8,
        device=device,
        tokenizer=tokenizer
    )

    # 2️⃣ Генерация 1/4 текста
    context_len = int(len(tokens) * 0.75)
    context_quarter = tokens[:context_len]
    target_quarter = tokens[context_len:]
    generated_quarter = model.generate_text(
        context_quarter,
        max_new_tokens=len(target_quarter),
        temperature=0.8,
        device=device,
        tokenizer=tokenizer
    )

    # Декодируем для печати
    context_text = tokenizer.decode(context_quarter, skip_special_tokens=True)
    gen_one_text = tokenizer.decode(generated_one, skip_special_tokens=True)
    gen_quarter_text = tokenizer.decode(generated_quarter, skip_special_tokens=True)
    target_one_text = tokenizer.decode(target_one_token, skip_special_tokens=True)
    target_quarter_text = tokenizer.decode(target_quarter, skip_special_tokens=True)
    original_text = tokenizer.decode(tokens, skip_special_tokens=True)

    # ROUGE
    rouge_one = rouge_metric.compute(predictions=[gen_one_text], references=[target_one_text])
    rouge_quarter = rouge_metric.compute(predictions=[gen_quarter_text], references=[target_quarter_text])

    return {
        "original": original_text,
        "context": context_text,
        "generated_one": gen_one_text,
        "generated_quarter": gen_quarter_text,
        "target_one": target_one_text,
        "target_quarter": target_quarter_text,
        "rouge_one": rouge_one,
        "rouge_quarter": rouge_quarter
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Путь к модели и токенизатору
    model_path = "models/birnn_lstm_lm.pth"
    checkpoint = torch.load(model_path, map_location=device)
    tokenizer_path = "data/processed/tokenizer"

    # Загружаем токенизатор
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)

    # Загружаем модель
    checkpoint = torch.load(model_path, map_location=device)
    model = BiRNNClassifier(
        vocab_size=checkpoint['vocab_size'],
        hidden_dim=checkpoint['hidden_dim']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Загружаем тестовые тексты
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "processed")
    test_file = os.path.join(data_dir, "test.txt")
    with open(test_file, "r", encoding="utf-8") as f:
        test_texts = [line.strip() for line in f]

    # 1️⃣ Вывод первых 10 примеров
    print("\n=== ПЕРВЫЕ 10 ТЕСТОВЫХ ПРИМЕРОВ ===")
    for i, text in enumerate(test_texts[:10]):
        res = generate_and_evaluate(model, tokenizer, text, device)
        if res is None:
            continue

        print(f"\nПример {i+1}:")
        print(f"Оригинальный текст: {res['original']}")
        print(f"Промт (3/4 текста): {res['context']}")
        print(f"Сгенерировано один токен: {res['generated_one']} (правильный: {res['target_one']})")
        print(f"Сгенерировано 1/4 текста: {res['generated_quarter']} (правильное продолжение: {res['target_quarter']})")
        print("-" * 80)

    # 2️⃣ Вычисление среднего ROUGE для следующих 100 примеров
    print("\n=== СРЕДНИЙ ROUGE ПО 100 ТЕСТАМ ===")
    rouge_one_scores = []
    rouge_quarter_scores = []

    for text in tqdm(test_texts[10:110], desc="Calculating average ROUGE"):
        res = generate_and_evaluate(model, tokenizer, text, device)
        if res is None:
            continue
        # Усредняем только F1 score
        rouge_one_scores.append(res['rouge_one']['rouge1'])
        rouge_quarter_scores.append(res['rouge_quarter']['rouge1'])

    avg_rouge_one = np.mean(rouge_one_scores) if rouge_one_scores else 0.0
    avg_rouge_quarter = np.mean(rouge_quarter_scores) if rouge_quarter_scores else 0.0

    print(f"Средний ROUGE-1 для одного токена: {avg_rouge_one:.4f}")
    print(f"Средний ROUGE-1 для 1/4 текста: {avg_rouge_quarter:.4f}")


if __name__ == "__main__":
    main()
