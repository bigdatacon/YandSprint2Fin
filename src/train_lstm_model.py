# train.py
import torch
import torch.nn as nn
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast

from lstm_model import BiRNNClassifier
from next_token_dataset import NextTokenDataset, collate_fn


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for x, y in tqdm(loader, desc="Training"):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits, _ = model(x)
        # logits: (B, T, V)
        # y:      (B, T)

        loss = criterion(
            logits.view(-1, logits.size(-1)),
            y.view(-1)
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    for x, y in tqdm(loader, desc="Evaluating"):
        x = x.to(device)
        y = y.to(device)

        logits, _ = model(x)

        loss = criterion(
            logits.view(-1, logits.size(-1)),
            y.view(-1)
        )

        total_loss += loss.item()

    return total_loss / len(loader)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # пути
    data_dir = "data/processed"
    tokenizer = BertTokenizerFast.from_pretrained(os.path.join(data_dir, "tokenizer"))
    vocab_size = tokenizer.vocab_size

    # данные
    def load_texts(path):
        with open(path, encoding="utf-8") as f:
            return [l.strip() for l in f if l.strip()]

    train_texts = load_texts(os.path.join(data_dir, "train.txt"))
    val_texts   = load_texts(os.path.join(data_dir, "val.txt"))

    train_ds = NextTokenDataset(train_texts, tokenizer)
    val_ds   = NextTokenDataset(val_texts, tokenizer)

    train_loader = DataLoader(
        train_ds,
        batch_size=64,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=64,
        shuffle=False,
        collate_fn=collate_fn
    )

    # модель
    model = BiRNNClassifier(
        vocab_size=vocab_size,
        emb_dim=128,
        hidden_dim=128
    ).to(device)

    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    criterion = nn.CrossEntropyLoss(
        ignore_index=tokenizer.pad_token_id
    )

    n_epochs = 3

    for epoch in range(n_epochs):
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss = eval_epoch(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch {epoch+1}/{n_epochs} | "
            f"Train loss: {train_loss:.4f} | "
            f"Val loss: {val_loss:.4f}"
        )

    # сохранение
    os.makedirs("models", exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab_size": vocab_size,
            "emb_dim": 128,
            "hidden_dim": 128,
        },
        "models/birnn_lstm_lm.pth"
    )

    print("Mодель сохранена : models/birnn_lstm_lm.pth")


if __name__ == "__main__":
    main()
