# LoRA with GPT-2, rank = 4, COLA dataset, Key-Value-Query-Projection layers
# GPT-2-base: 79.70  Accuracy (LR: 1e-3, BS: 256, Epochs: 15)
# GPT-2-medium: 83.68  Accuracy (LR: 8e-4, BS: 128, Epochs: 50)

import torch
from transformers import AutoTokenizer

# from GPTmodel import GPT
from LoRAmodel import LoRAGPT
from utils import *


def main(args: ArgStorage) -> None:
    device = get_device(args.gpu_id)
    
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    train_loader = get_data_loader(
        'data/in_domain_train.tsv', args.batch_size, tokenizer)
    val_loader = get_data_loader(
        'data/in_domain_dev.tsv', args.batch_size, tokenizer, shuffle=False)

    # model = GPT.from_pretrained('gpt2')
    model = LoRAGPT()
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.NLLLoss()

    hist = 0.0
    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0
        correct = 0
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device),\
                attention_mask.to(device), labels.to(device)
            logits = model(input_ids, attention_mask)
            log_probs = torch.log_softmax(logits, dim=-1)
            loss = criterion(log_probs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            correct += (logits.argmax(1) == labels).sum().item()
        train_loss /= len(train_loader)
        train_acc = correct / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                input_ids, attention_mask, labels = batch
                input_ids, attention_mask, labels = input_ids.to(device),\
                    attention_mask.to(device), labels.to(device)
                logits = model(input_ids, attention_mask)
                log_probs = torch.log_softmax(logits, dim=-1)
                loss = criterion(log_probs, labels)
                val_loss += loss.item()
                correct += (logits.argmax(1) == labels).sum().item()
        val_loss /= len(val_loader)
        val_acc = correct / len(val_loader.dataset)

        if val_acc > hist:
            hist = val_acc
            model.save_trainable_params('model.pth')

        print(
            f'Epoch {epoch+1}/{args.epochs} | '
            f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | '
            f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}'
        )


if __name__ == '__main__':   
    torch.manual_seed(2024)
    torch.cuda.manual_seed_all(2024)
    torch.backends.cudnn.deterministic = True

    args = ArgStorage(
        gpu_id=0,
        lr=8e-4,
        batch_size=128,
        epochs=50,
    )

    main(args)
