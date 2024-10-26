import torch
from transformers import AutoTokenizer

from LoRAGPT import LoRAGPT


tokenizer = AutoTokenizer.from_pretrained('gpt2')

model = LoRAGPT.from_pretrained('gpt2')

sentence = "What is the capital of France?"

input_ids = tokenizer.encode(sentence, return_tensors='pt')
model.eval()
output = model.generate(input_ids, 100)
print()
print(tokenizer.decode(output[0]))
