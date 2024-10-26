import torch
from transformers import AutoTokenizer

from LoRAmodel import LoRAGPT
from utils import *


def main(args: ArgStorage) -> None:
    device = get_device(args.gpu_id)
    
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    val_loader = get_data_loader(
        'data/in_domain_dev.tsv', args.batch_size, tokenizer, shuffle=False)

    model = LoRAGPT()
    model.to(device)

    model.load_trainable_params('model.pth')
    
    criterion = torch.nn.NLLLoss()

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
    print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}')


if __name__ == '__main__':   
    torch.manual_seed(2024)
    torch.cuda.manual_seed_all(2024)
    torch.backends.cudnn.deterministic = True

    args = ArgStorage(
        gpu_id=0,
        batch_size=128,
    )

    main(args)
