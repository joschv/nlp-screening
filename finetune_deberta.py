import evaluate
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, DataCollatorWithPadding, \
    Trainer

model_checkpoint = "microsoft/deberta-v3-xsmall"

# load the pretrained model and tokenizer
# model is smaller than 300MB but compared to some other options i tried its rather large
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=1)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

finetuning_datasets = load_dataset("glue", "stsb")
# labels: similarity score 0-5

def tokenize_fn(r):
    return tokenizer(r["sentence1"], r["sentence2"], truncation=True)

tokenized_datasets = finetuning_datasets.map(tokenize_fn, batched=True, batch_size=8)
data_collator = DataCollatorWithPadding(tokenizer)

training_args = TrainingArguments(
    "ft_deberta_similarity",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy="epoch"
)


metric = evaluate.load("accuracy")

def compute_metrics(eval_predictions):
    logits, labels = eval_predictions
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# to reduce computation time and cost, only a subset of the dataset is used (increase for better results)
TRAIN_SIZE = 200
VALIDATION_SIZE = 100

if TRAIN_SIZE is None:
    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]
else:
    train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(TRAIN_SIZE))
    eval_dataset = tokenized_datasets["validation"].shuffle(seed=42).select(range(TRAIN_SIZE))

trainer = Trainer(
    model,
    training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    processing_class=tokenizer,
    # compute_metrics=compute_metrics
)

trainer.train()
