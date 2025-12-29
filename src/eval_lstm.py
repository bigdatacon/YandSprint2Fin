import torch
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
    Генерация и оценка текста:
    1️⃣ Один токен
    2️⃣ 1/4 текста
    """
    tokens = tokenizer.encode(text, add_special_tokens=True)
    if len(tokens) < 2:
        return None

    # Контекст для генерации: 3/4 текста
    context_len = int(len(tokens) * 0.75)
    context_one_token = tokens[:context_len]
    target_quarter = tokens[context_len:]

    # 1️⃣ Определяем target_one как первый «реальный» токен из оставшейся четверти
    target_one_token = None
    for tok in target_quarter:
        if tok not in {tokenizer.sep_token_id, tokenizer.cls_token_id, tokenizer.pad_token_id, tokenizer.cls_token_id}:
            target_one_token = [tok]
            break
    if target_one_token is None:
        target_one_token = []

    # Генерация одного токена
    generated_one = model.generate_text(
        context_one_token,
        max_new_tokens=1,
        temperature=1.0,
        device=device,
        tokenizer=tokenizer
    )
    generated_one_token = generated_one[-1:]  # последний токен

    # 2️⃣ Генерация 1/4 текста, начиная с сгенерированного токена
    # context_quarter = context_one_token + generated_one_token
    remaining_len = len(target_quarter)  
    generated_quarter = model.generate_text(
        context_one_token,
        max_new_tokens=remaining_len,
        temperature=1,
        device=device,
        tokenizer=tokenizer
    )
    # Берём только сгенерированное продолжение четверти текста
    generated_quarter_only = generated_one_token + generated_quarter[len(context_one_token):]

    # Декодирование
    original_text = tokenizer.decode(tokens, skip_special_tokens=True)
    context_text = tokenizer.decode(context_one_token, skip_special_tokens=True)
    gen_one_text = tokenizer.decode(generated_one_token, skip_special_tokens=True)
    gen_quarter_text = tokenizer.decode(generated_quarter_only, skip_special_tokens=True)
    target_one_text = tokenizer.decode(target_one_token, skip_special_tokens=True)
    target_quarter_text = tokenizer.decode(target_quarter, skip_special_tokens=True)

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



def evaluate_texts(model, tokenizer, texts, device, print_examples=True):
    rouge_one_scores = []
    rouge_quarter_scores = []

    # Печать первых 10 примеров
    if print_examples:
        print("\n=== ПЕРВЫЕ 10 ТЕСТОВЫХ ПРИМЕРОВ ===")
        for i, text in enumerate(texts[20:30]):
            res = generate_and_evaluate(model, tokenizer, text, device)
            if res is None:
                continue

            print(f"\nПример {i+1}:")
            print(f"Оригинальный текст: {res['original']}")
            print(f"Промт (3/4 текста): {res['context']}")
            print(f"Сгенерировано один токен: {res['generated_one']} (правильный: {res['target_one']})")
            print(f"Сгенерировано 1/4 текста: {res['generated_quarter']} (правильное продолжение: {res['target_quarter']})")
            print(f"ROUGE-1 один токен: {res['rouge_one']['rouge1']:.4f}")
            print(f"ROUGE-1 1/4 текста: {res['rouge_quarter']['rouge1']:.4f}")
            print("-" * 80)

    # Усреднение ROUGE по всем текстам
    for text in tqdm(texts[:210], desc="Calculating average ROUGE"):
        res = generate_and_evaluate(model, tokenizer, text, device)
        if res is None:
            continue
        rouge_one_scores.append(res['rouge_one']['rouge1'])
        rouge_quarter_scores.append(res['rouge_quarter']['rouge1'])

    avg_rouge_one = np.mean(rouge_one_scores) if rouge_one_scores else 0.0
    avg_rouge_quarter = np.mean(rouge_quarter_scores) if rouge_quarter_scores else 0.0

    print("\n=== СРЕДНИЙ ROUGE ===")
    print(f"Средний ROUGE-1 для одного токена: {avg_rouge_one:.4f}")
    print(f"Средний ROUGE-1 для 1/4 текста: {avg_rouge_quarter:.4f}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Путь к модели и токенизатору
    model_path = "models/birnn_lstm_lm.pth"
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
        test_texts = [line.strip() for line in f if line.strip()]

    # Вызываем функцию оценки
    evaluate_texts(model, tokenizer, test_texts, device, print_examples=True)


if __name__ == "__main__":
    main()
