import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import evaluate
import numpy as np
from tqdm import tqdm

# --- Функция для загрузки тестовых текстов ---
def load_texts(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

# --- Пути к данным ---
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "processed")
test_file = os.path.join(data_dir, "test.txt")

# Загружаем тестовые тексты
test_texts = load_texts(test_file)[:100]  # берём первые 100

# --- Загрузка модели и токенизатора ---
model_name = "GPT2" # на модели distilgpt2 rouge около 2% то есть меньше чем на LSTM поэтому переделал на GPT2
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# --- Pipeline для генерации ---
generator = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    device=-1,  # CPU; для GPU ставьте 0
)

# --- Метрика ROUGE ---
rouge = evaluate.load("rouge")

# --- Функция для генерации и оценки ---
def generate_and_evaluate_transformer(text):
    words = text.split()
    if len(words) < 5 or len(words) > 100:
        return None

    # Делим на 3/4 и 1/4
    split_idx = int(len(words) * 0.75)
    prompt_text = ' '.join(words[:split_idx])
    target_text = ' '.join(words[split_idx:])

    # Генерация продолжения
    out = generator(
        prompt_text,
        max_new_tokens=len(words) - split_idx,
        num_return_sequences=1,
        do_sample=False,
        top_p=0.95,
        temperature=0.8
    )

    generated_full = out[0]["generated_text"]
    # Берём только сгенерированное продолжение
    if generated_full.startswith(prompt_text):
        generated_part = generated_full[len(prompt_text):].strip()
    else:
        generated_part = generated_full

    # ROUGE
    rouge_scores = rouge.compute(predictions=[generated_part], references=[target_text])

    return {
        "original": text,
        "prompt": prompt_text,
        "generated_quarter": generated_part[:100],
        "target_quarter": target_text,
        "rouge1": rouge_scores["rouge1"],
        "rouge2": rouge_scores["rouge2"],
        "rougeL": rouge_scores["rougeL"]
    }

# --- 1️⃣ Вывод первых 10 примеров ---
print("\n=== ПЕРВЫЕ 10 ПРИМЕРОВ ===")
for i, text in tqdm(enumerate(test_texts[:10])):
    res = generate_and_evaluate_transformer(text)
    if res is None:
        continue

    print(f"\nПример {i+1}:")
    print(f"Оригинальный текст: {res['original']}")
    print(f"Промт (3/4 текста): {res['prompt']}")
    print(f"Сгенерировано 1/4 текста: {res['generated_quarter']}")
    print(f"Правильное продолжение: {res['target_quarter']}")
    print(f"ROUGE-1: {res['rouge1']:.4f}")
    print(f"ROUGE-2: {res['rouge2']:.4f}")
    print(f"ROUGE-L: {res['rougeL']:.4f}")
    print("-" * 80)

# --- 2️⃣ Средний ROUGE по 100 тестовым текстам ---
rouge1_scores = []
rouge2_scores = []
rougeL_scores = []

for text in tqdm(test_texts[:40], desc="Вычисление среднего ROUGE"):
    res = generate_and_evaluate_transformer(text)
    if res is None:
        continue
    rouge1_scores.append(res['rouge1'])
    rouge2_scores.append(res['rouge2'])
    rougeL_scores.append(res['rougeL'])

avg_rouge1 = np.mean(rouge1_scores) if rouge1_scores else 0.0
avg_rouge2 = np.mean(rouge2_scores) if rouge2_scores else 0.0
avg_rougeL = np.mean(rougeL_scores) if rougeL_scores else 0.0

print("\n=== СРЕДНИЙ ROUGE ПО 100 ТЕСТАМ ===")
print(f"Средний ROUGE-1: {avg_rouge1:.4f}")
print(f"Средний ROUGE-2: {avg_rouge2:.4f}")
print(f"Средний ROUGE-L: {avg_rougeL:.4f}")
