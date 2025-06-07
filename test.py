import clip
import torch

model, _ = clip.load("ViT-B/16")
word = "dog"
tokenized_word = clip.tokenize([word, "I have a dogs"])
print(torch.argmax(tokenized_word, dim=-1))