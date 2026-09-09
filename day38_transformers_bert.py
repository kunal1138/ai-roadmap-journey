# DAY 38: Transformers & BERT 🤖
# The architecture that changed AI!

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModel
from transformers import BertTokenizer
from transformers import BertModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from matplotlib.patches import Patch

print("Transformers & BERT Day! 🤖")
print(f"PyTorch version: {torch.__version__}")

# ===== PART 1: SENTIMENT ANALYSIS =====
print("\n" + "="*55)
print("PART 1: HUGGING FACE PIPELINES")
print("="*55)

print("\nLoading pre-trained models...")

sentiment = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

reviews = [
    "This movie was absolutely amazing!",
    "Terrible film waste of time and money",
    "Kunal loves learning machine learning!",
    "The worst experience I ever had",
    "Outstanding performance highly recommended",
    "Boring and completely disappointing",
]

print(f"\n{'Review':45} {'Label':10} {'Score':8}")
print("-" * 65)
for review in reviews:
    result = sentiment(review)[0]
    emoji = "😊" if result["label"]=="POSITIVE" \
            else "😞"
    print(f"{review[:43]:45} "
          f"{result['label']:10} "
          f"{result['score']:.4f} {emoji}")

# ===== PART 2: ZERO-SHOT =====
print("\n" + "="*55)
print("PART 2: ZERO-SHOT CLASSIFICATION")
print("="*55)

print("\nClassifying without training! 🤯")

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

texts = [
    "Kunal is learning Python and machine learning",
    "India won the cricket world cup match",
    "The biryani and butter chicken were delicious",
    "TensorFlow and PyTorch are deep learning tools",
]
labels = ["technology","sports","food","education"]

print(f"\n{'Text':45} {'Top Label':12} {'Score':8}")
print("-" * 67)
for text in texts:
    result = classifier(text,
                        candidate_labels=labels)
    top_label = result["labels"][0]
    top_score = result["scores"][0]
    print(f"{text[:43]:45} "
          f"{top_label:12} "
          f"{top_score:.4f}")

# ===== PART 3: TEXT GENERATION =====
print("\n" + "="*55)
print("PART 3: TEXT GENERATION (GPT-2)")
print("="*55)

generator = pipeline(
    "text-generation",
    model="gpt2"
)

prompts = [
    "Machine learning is",
    "Kunal is learning Python because",
    "The future of AI will",
]

print("\nGPT-2 text generation:")
for prompt in prompts:
    result = generator(
        prompt,
        max_new_tokens=30,
        num_return_sequences=1,
        pad_token_id=50256
    )
    generated = result[0]["generated_text"]
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {generated}")

# ===== PART 4: BERT TOKENIZER =====
print("\n" + "="*55)
print("PART 4: BERT TOKENIZER")
print("="*55)

tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased")

texts_to_tokenize = [
    "Kunal is learning NLP",
    "Cricket is popular in India",
    "Machine learning is amazing"
]

print(f"\n{'Text':35} {'Tokens'}")
print("-" * 70)
for text in texts_to_tokenize:
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.encode(text)
    print(f"\nText: {text}")
    print(f"Tokens: {tokens}")
    print(f"Token IDs: {token_ids}")

print("\nSpecial BERT tokens:")
print("[CLS] = Classification token (start)")
print("[SEP] = Separator token (end)")
print("[MASK] = Masked word (for training)")
print("[PAD] = Padding token (for batching)")

# ===== PART 5: BERT EMBEDDINGS =====
print("\n" + "="*55)
print("PART 5: BERT EMBEDDINGS")
print("="*55)

print("\nLoading BERT model...")
bert_model = BertModel.from_pretrained(
    "bert-base-uncased")
bert_model.eval()

def get_bert_embedding(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=True
    )
    with torch.no_grad():
        outputs = bert_model(**inputs)
    cls_embedding = outputs.last_hidden_state[
        :, 0, :].numpy()
    return cls_embedding[0]

sentences = [
    "Cricket is exciting sport",
    "Cricket match was amazing",
    "Python programming language",
    "Machine learning with Python",
    "Nagpur oranges are delicious",
    "Indian food is very tasty",
    "Kunal loves learning everyday",
    "Deep learning neural networks",
]

print(f"\nGetting BERT embeddings...")
embeddings = []
for sent in sentences:
    emb = get_bert_embedding(sent)
    embeddings.append(emb)
    print(f"✅ {sent[:40]}")

embeddings = np.array(embeddings)
print(f"\nEmbedding shape: {embeddings.shape}")
print(f"(sentences × BERT dimensions)")

# ===== PART 6: SEMANTIC SIMILARITY =====
print("\n" + "="*55)
print("PART 6: SEMANTIC SIMILARITY")
print("="*55)

sim_matrix = cosine_similarity(embeddings)

print(f"\n{'Sentence 1':30} "
      f"{'Sentence 2':30} {'Sim':6}")
print("-" * 70)
for i in range(len(sentences)):
    for j in range(i+1, len(sentences)):
        sim = sim_matrix[i][j]
        if sim > 0.85:
            print(
                f"{sentences[i][:28]:30}"
                f"{sentences[j][:28]:30}"
                f"{sim:.4f} 🔥 similar!")

# ===== PART 7: BERT SENTIMENT =====
print("\n" + "="*55)
print("PART 7: SENTIMENT WITH BERT")
print("="*55)

reviews_data = [
    ("Amazing movie loved every minute", 1),
    ("Terrible waste of time and money", 0),
    ("Outstanding performance brilliant", 1),
    ("Worst movie ever made boring", 0),
    ("Fantastic film highly recommend", 1),
    ("Awful horrible acting poor story", 0),
    ("Masterpiece of cinema remarkable", 1),
    ("Disappointing film poor direction", 0),
    ("Brilliant story great acting", 1),
    ("Horrible film never watch again", 0),
    ("Loved it incredible experience", 1),
    ("Disgusting waste avoid watching", 0),
    ("Beautiful heartwarming wonderful", 1),
    ("Boring fell asleep watching it", 0),
    ("Stunning visuals amazing story", 1),
    ("Pathetic terrible dreadful film", 0),
    ("Perfect film must watch everyone", 1),
    ("Complete disaster awful movie", 0),
    ("Kunal loved this film amazing", 1),
    ("Sonu hated this boring movie", 0),
]

print(f"\nGetting BERT embeddings for reviews...")
X_reviews = []
y_reviews = []

for text, label in reviews_data:
    emb = get_bert_embedding(text)
    X_reviews.append(emb)
    y_reviews.append(label)

X_reviews = np.array(X_reviews)
y_reviews = np.array(y_reviews)

X_train, X_test, y_train, y_test = \
    train_test_split(
        X_reviews, y_reviews,
        test_size=0.2, random_state=42
    )

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
bert_acc = accuracy_score(y_test, y_pred)

print(f"\nBERT + LR Accuracy: "
      f"{bert_acc*100:.2f}%")
print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=["Negative","Positive"]))

# ===== PART 8: COMPARISON =====
print("\n" + "="*55)
print("PART 8: BERT vs TRADITIONAL NLP")
print("="*55)

print(f"\n{'Method':25} {'Accuracy':10}")
print("-" * 37)
print(f"{'TF-IDF + NB (Day 36)':25} {'87.50%':10}")
print(f"{'Word2Vec (Day 37)':25} {'Limited':10}")
print(f"{'BERT + LR (Today)':25} "
      f"{bert_acc*100:.2f}%")

print("""
Why BERT wins:
→ Pre-trained on billions of words!
→ Understands context deeply!
→ Bidirectional reading!
→ Transfer learning power!
""")

# ===== PART 9: VISUALIZATION =====
pca = PCA(n_components=2)
emb_2d = pca.fit_transform(embeddings)

fig, axes = plt.subplots(1, 2,
                          figsize=(14, 6))

colors = ["blue","blue",
          "orange","orange",
          "green","green",
          "red","red"]
categories = ["Cricket","Cricket",
              "Tech","Tech",
              "Food","Food",
              "People","People"]

for i, (sent, color, cat) in enumerate(
    zip(sentences, colors, categories)):
    axes[0].scatter(
        emb_2d[i,0], emb_2d[i,1],
        c=color, s=200, alpha=0.7)
    axes[0].annotate(
        sent[:15]+"...",
        (emb_2d[i,0], emb_2d[i,1]),
        fontsize=8)

axes[0].set_title("BERT Embeddings PCA 2D")
axes[0].set_xlabel("PCA 1")
axes[0].set_ylabel("PCA 2")
legend = [
    Patch(color="blue", label="Cricket"),
    Patch(color="orange", label="Tech"),
    Patch(color="green", label="Food"),
    Patch(color="red", label="People"),
]
axes[0].legend(handles=legend)

im = axes[1].imshow(
    sim_matrix, cmap="RdYlGn",
    vmin=0, vmax=1)
plt.colorbar(im, ax=axes[1])
axes[1].set_xticks(range(len(sentences)))
axes[1].set_yticks(range(len(sentences)))
short = [s[:10]+"..." for s in sentences]
axes[1].set_xticklabels(
    short, rotation=90, fontsize=7)
axes[1].set_yticklabels(short, fontsize=7)
axes[1].set_title("BERT Similarity Heatmap")

for i in range(len(sentences)):
    for j in range(len(sentences)):
        axes[1].text(
            j, i,
            f"{sim_matrix[i,j]:.2f}",
            ha="center", va="center",
            fontsize=6)

plt.suptitle("BERT Embeddings Visualization",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()

# ===== PART 10: FILL MASK =====
print("\n" + "="*55)
print("PART 10: BERT FILL MASK")
print("="*55)

fill_mask = pipeline(
    "fill-mask",
    model="bert-base-uncased"
)

masked_sentences = [
    "Kunal is learning [MASK] learning",
    "Cricket is the most popular [MASK] in India",
    "[MASK] is the best programming language",
    "Nagpur is famous for its [MASK]",
]

print("\nBERT Fill-Mask predictions:")
for sent in masked_sentences:
    print(f"\nMasked: {sent}")
    results = fill_mask(sent)
    print("Predictions:")
    for r in results[:3]:
        print(f"  '{r['token_str']}'"
              f" → {r['score']:.4f}")

# ===== SUMMARY =====
print("\n" + "="*55)
print("DAY 38 SUMMARY")
print("="*55)
print("✅ Transformer architecture")
print("✅ Attention mechanism")
print("✅ BERT pre-trained model")
print("✅ Hugging Face pipelines")
print("✅ Sentiment analysis pipeline")
print("✅ Zero-shot classification")
print("✅ GPT-2 text generation")
print("✅ BERT tokenization")
print("✅ BERT embeddings (768 dim)")
print("✅ Semantic similarity")
print(f"✅ BERT accuracy: "
      f"{bert_acc*100:.2f}%")
print("✅ Fill-mask predictions")
print("\n🤖 Transformers & BERT — COMPLETE!")
print("🚀 Tomorrow: Hugging Face Deep Dive!")