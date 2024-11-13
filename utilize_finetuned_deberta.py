import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("ft_deberta_similarity/checkpoint-65")
model = AutoModelForSequenceClassification.from_pretrained("ft_deberta_similarity/checkpoint-65", num_labels=1)
print(model)

inputs = tokenizer("Hello, my dog is cute", "My dog cute", return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits
    print(logits)
