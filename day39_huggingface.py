# DAY 39: Hugging Face Deep Dive 🤗
# Fine-tuning + Datasets + Model Hub

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
from transformers import DataCollatorWithPadding
from datasets import Dataset
from datasets import DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("Hugging Face Deep Dive! 🤗")
print(f"PyTorch: {torch.__version__}")

device = "cuda" if torch.cuda.is_available() \
         else "cpu"
print(f"Device: {device}")

# ===== PART 1: DATASET =====
print("\n" + "="*55)
print("PART 1: HUGGING FACE DATASETS")
print("="*55)

positive_reviews = [
    "This movie was absolutely brilliant",
    "Amazing film loved every minute",
    "Outstanding performance by all actors",
    "Masterpiece of modern cinema",
    "Fantastic story and great direction",
    "Loved the beautiful cinematography",
    "Perfect film highly recommended",
    "Incredible acting emotional story",
    "Stunning visuals amazing narrative",
    "Best movie of the year easily",
    "Wonderful heartwarming experience",
    "Brilliant screenplay great acting",
    "Exceptional film must watch",
    "Superb direction and story",
    "Outstanding movie loved it",
    "Kunal loved this amazing film",
    "Sonu and friends enjoyed it",
    "Best Bollywood film ever made",
    "Durgesh recommended brilliant movie",
    "Om watched twice loved it",
    "Great cricket film loved it",
    "Amazing sports movie brilliant",
    "Fantastic thriller enjoyed completely",
    "Beautiful love story wonderful",
    "Action packed exciting brilliant",
]

negative_reviews = [
    "Terrible movie waste of time",
    "Awful film horrible acting",
    "Worst movie ever made boring",
    "Disappointing film poor direction",
    "Hated every minute terrible",
    "Complete disaster avoid watching",
    "Poor story bad acting boring",
    "Waste of money horrible experience",
    "Horrible film never watch again",
    "Dreadful performance poor movie",
    "Pathetic story bad direction",
    "Disgusting waste of time",
    "Boring fell asleep watching",
    "Terrible acting ruined story",
    "Worst film decade avoid",
    "Glou hated this boring film",
    "Vedu walked out terrible",
    "Tantan found it very boring",
    "Sam said worst movie ever",
    "Abhinav wasted money on this",
    "Terrible cricket film boring",
    "Awful sports movie disappointing",
    "Horrible thriller very boring",
    "Bad love story disappointing",
    "Boring action nothing happens",
]

texts = positive_reviews + negative_reviews
labels = ([1] * len(positive_reviews) +
          [0] * len(negative_reviews))

df = pd.DataFrame({
    "text": texts,
    "label": labels
})

print(f"Total reviews: {len(df)}")
print(f"Positive: {sum(labels)}")
print(f"Negative: {len(labels)-sum(labels)}")
print(df.head())

# ===== PART 2: TOKENIZER =====
print("\n" + "="*55)
print("PART 2: TOKENIZER DEEP DIVE")
print("="*55)

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(
    model_name)

print(f"\nModel: {model_name}")
print(f"Vocab size: {tokenizer.vocab_size}")
print(f"Max length: {tokenizer.model_max_length}")

sample_texts = [
    "Kunal loves cricket",
    "This movie is amazing",
    "Terrible waste of time"
]

print("\nTokenization examples:")
for text in sample_texts:
    encoded = tokenizer(
        text,
        padding="max_length",
        max_length=20,
        truncation=True,
        return_tensors="pt"
    )
    tokens = tokenizer.convert_ids_to_tokens(
        encoded["input_ids"][0])
    print(f"\nText: {text}")
    print(f"Tokens: {tokens}")
    print(f"Input IDs: "
          f"{encoded['input_ids'][0].tolist()}")
    print(f"Attention mask: "
          f"{encoded['attention_mask'][0].tolist()}")

print("\nAttention mask:")
print("1 = real token (pay attention!)")
print("0 = padding token (ignore!)")

# ===== PART 3: PREPARE DATASET =====
print("\n" + "="*55)
print("PART 3: PREPARE DATASET")
print("="*55)

train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42,
    stratify=df["label"]
)
train_df, val_df = train_test_split(
    train_df, test_size=0.2,
    random_state=42,
    stratify=train_df["label"]
)

print(f"Train: {len(train_df)}")
print(f"Val: {len(val_df)}")
print(f"Test: {len(test_df)}")

train_dataset = Dataset.from_pandas(
    train_df.reset_index(drop=True))
val_dataset = Dataset.from_pandas(
    val_df.reset_index(drop=True))
test_dataset = Dataset.from_pandas(
    test_df.reset_index(drop=True))

dataset_dict = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset,
    "test": test_dataset
})

print(f"\nDataset structure:")
print(dataset_dict)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        max_length=64,
        truncation=True
    )

print("\nTokenizing dataset...")
tokenized_datasets = dataset_dict.map(
    tokenize_function, batched=True)
print("Tokenization complete! ✅")

# ===== PART 4: FINE-TUNE =====
print("\n" + "="*55)
print("PART 4: FINE-TUNE DISTILBERT")
print("="*55)

print("\nLoading DistilBERT...")
model = AutoModelForSequenceClassification\
    .from_pretrained(
        model_name, num_labels=2)

print(f"Model loaded! ✅")
print(f"Parameters: "
      f"{model.num_parameters():,}")

# Fixed TrainingArguments
training_args = TrainingArguments(
    output_dir="./movie-sentiment-model",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=10,
    weight_decay=0.01,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none"
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("\nStarting fine-tuning...")
trainer.train()
print("\nFine-tuning complete! ✅")

# ===== PART 5: EVALUATE =====
print("\n" + "="*55)
print("PART 5: EVALUATE")
print("="*55)

results = trainer.evaluate(
    tokenized_datasets["test"])
print(f"\nTest Results:")
for key, value in results.items():
    print(f"  {key}: {value:.4f}")

# ===== PART 6: PREDICTIONS =====
print("\n" + "="*55)
print("PART 6: PREDICTIONS")
print("="*55)

fine_tuned_pipeline = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=0 if device=="cuda" else -1
)

test_reviews = [
    "This movie was absolutely brilliant!",
    "Terrible film waste of money",
    "Kunal loved this amazing cricket movie",
    "Boring and completely disappointing",
    "Outstanding performance recommend everyone",
    "Worst movie ever made avoid watching",
    "Beautiful heartwarming wonderful story",
    "Horrible acting ruined good story",
]

print(f"\n{'Review':42} {'Label':12} {'Score':8}")
print("-" * 65)
for review in test_reviews:
    result = fine_tuned_pipeline(review)[0]
    label = "POSITIVE" \
            if result["label"]=="LABEL_1" \
            else "NEGATIVE"
    emoji = "😊" if label=="POSITIVE" else "😞"
    print(f"{review[:40]:42} "
          f"{label:12} "
          f"{result['score']:.4f} {emoji}")

# ===== PART 7: COMPARISON =====
print("\n" + "="*55)
print("PART 7: MODEL COMPARISON")
print("="*55)

pretrained = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

comparison_reviews = [
    "Brilliant movie loved it",
    "Terrible waste of time",
    "Kunal enjoyed the film",
    "Boring disappointing film",
    "Amazing performance",
]

print(f"\n{'Review':35} {'Pre-trained':12} "
      f"{'Fine-tuned':12}")
print("-" * 62)
for review in comparison_reviews:
    pt_result = pretrained(review)[0]
    pt_label = "POS" \
               if pt_result["label"]=="POSITIVE" \
               else "NEG"
    ft_result = fine_tuned_pipeline(review)[0]
    ft_label = "POS" \
               if ft_result["label"]=="LABEL_1" \
               else "NEG"
    match = "✅" if pt_label==ft_label else "❌"
    print(f"{review[:33]:35} "
          f"{pt_label:12} "
          f"{ft_label:12} {match}")

# ===== PART 8: SAVE =====
print("\n" + "="*55)
print("PART 8: SAVE MODEL")
print("="*55)

save_path = "./my-movie-sentiment-model"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"Model saved to: {save_path}")

loaded_model = AutoModelForSequenceClassification\
    .from_pretrained(save_path)
loaded_tokenizer = AutoTokenizer\
    .from_pretrained(save_path)
print("Model loaded back! ✅")

loaded_pipeline = pipeline(
    "text-classification",
    model=loaded_model,
    tokenizer=loaded_tokenizer
)

test_text = "This is an amazing movie!"
result = loaded_pipeline(test_text)[0]
print(f"\nTest: {test_text}")
print(f"Result: {result}")

# ===== PART 9: VISUALIZATION =====
train_history = trainer.state.log_history

train_loss = []
eval_acc = []
epochs = []

for log in train_history:
    if "loss" in log:
        train_loss.append(log["loss"])
    if "eval_accuracy" in log:
        eval_acc.append(log["eval_accuracy"])
        epochs.append(log["epoch"])

fig, axes = plt.subplots(1, 2,
                          figsize=(12, 4))

if train_loss:
    axes[0].plot(train_loss,
                 color="blue",
                 label="Training Loss")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Steps")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

if eval_acc:
    axes[1].plot(epochs, eval_acc,
                 "go-", label="Val Accuracy")
    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1.1)
    for ep, acc in zip(epochs, eval_acc):
        axes[1].text(ep, acc+0.02,
                     f"{acc:.3f}",
                     ha="center")
    axes[1].legend()

plt.suptitle("Fine-Tuning Training History",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.show()

# ===== PART 10: ECOSYSTEM =====
print("\n" + "="*55)
print("PART 10: HF ECOSYSTEM")
print("="*55)
print("""
🤗 Hugging Face Ecosystem:

1. transformers → Pre-trained models
2. datasets     → 50,000+ datasets!
3. Model Hub    → 500,000+ models!
4. tokenizers   → Fast tokenization
5. evaluate     → Standard metrics
6. Spaces       → Free deployment!

Popular models:
→ bert-base-uncased (understanding)
→ gpt2 (generation)
→ distilbert (fast BERT)
→ roberta (improved BERT)
→ t5 (text-to-text)
→ whisper (speech recognition)
→ stable-diffusion (images)
""")

# ===== SUMMARY =====
print("\n" + "="*55)
print("DAY 39 SUMMARY")
print("="*55)
print("✅ Hugging Face ecosystem")
print("✅ Custom dataset creation")
print("✅ Tokenizer deep dive")
print("✅ Attention masks")
print("✅ Dataset preparation")
print("✅ DistilBERT fine-tuning")
print("✅ Model evaluation")
print("✅ Predictions")
print("✅ Pre-trained vs fine-tuned")
print("✅ Model saving and loading")
print("✅ Training visualization")
print(f"\n🤗 Hugging Face — COMPLETE!")
print(f"🚀 Tomorrow: LLM APIs + Claude API!")