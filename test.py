import clip
import torch

model, _ = clip.load("RN50")
tokenized_word = clip.tokenize(["I have a dog"]).to("cuda")
print(tokenized_word.shape)
print(model.token_embedding(tokenized_word).shape)
print(model.encode_text(tokenized_word).shape)