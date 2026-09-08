# DAY 37: Word Embeddings & Word2Vec 🔤
# How computers understand word meaning!

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import gensim
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("punkt_tab")

print("Word Embeddings Day! 🔤")
print(f"Gensim version: {gensim.__version__}")

# ===== WHY EMBEDDINGS =====
print("\n" + "="*50)
print("PART 1: WHY WORD EMBEDDINGS?")
print("="*50)
print("""
Problem with BoW/TF-IDF:
→ "Good" and "Great" = completely different!
→ No semantic meaning!

Word Embedding solution:
→ Similar words = similar vectors!
→ Captures relationships!

Famous equations:
King - Man + Woman = Queen 🤯
Paris - France + Italy = Rome 🤯
""")

# ===== TRAINING DATA =====
print("\n" + "="*50)
print("PART 2: TRAINING DATA")
print("="*50)

sentences_raw = [
    # Cricket
    "Kunal loves cricket and plays every weekend",
    "Cricket is the most popular sport in India",
    "Virat Kohli is the best cricket player",
    "Rohit Sharma hits amazing sixes in cricket",
    "The cricket match was exciting and thrilling",
    "India won the cricket world cup finally",
    "MS Dhoni is a legendary cricket captain",
    "Cricket fans celebrated the victory loudly",
    "IPL cricket tournament is very popular",
    "Nagpur has excellent cricket stadium ground",
    # Technology
    "Python is the best programming language",
    "Machine learning uses Python for data science",
    "Deep learning neural networks are powerful",
    "Artificial intelligence is changing the world",
    "TensorFlow and PyTorch are deep learning tools",
    "NLP helps computers understand human language",
    "Word embeddings capture semantic word meaning",
    "BERT and GPT are transformer based models",
    "Kunal is learning machine learning every day",
    "AI and ML are future of technology industry",
    # Food
    "Nagpur oranges are famous all over India",
    "Poha is popular breakfast dish in Nagpur",
    "Butter chicken is delicious Indian food",
    "Biryani rice dish is loved by everyone",
    "Vada pav is famous Mumbai street food",
    "Dosa idli are popular South Indian foods",
    "Samosa is crispy fried Indian snack food",
    "Chai tea is favourite drink in India",
    "Mango is the king of all fruits India",
    "Indian food is spicy and very delicious",
    # Friends
    "Kunal and Sonu study together every day",
    "Durgesh is good at programming and coding",
    "Om plays cricket with Kunal on weekends",
    "Abhinav loves football and outdoor sports",
    "Glou and Vedu are best friends forever",
    "Tantan helps friends with difficult problems",
    "Sam is the fastest runner in the group",
    "Friends support each other during hard times",
    "Studying together makes learning more fun",
    "Good friends make life better and happier",
    # Education
    "Students study hard to get good marks",
    "College education opens many career doors",
    "Learning programming leads to good jobs",
    "Machine learning engineer earns high salary",
    "Consistent practice makes perfect skills",
    "Reading books improves knowledge and wisdom",
    "Online courses help learn new technologies",
    "GitHub portfolio impresses job recruiters",
    "Certifications validate technical skills",
    "Hard work and dedication leads to success",
]

stop_words = set(stopwords.words("english"))

def preprocess(sentence):
    tokens = word_tokenize(sentence.lower())
    tokens = [w for w in tokens
              if w.isalpha()
              and w not in stop_words
              and len(w) > 2]
    return tokens

tokenized = [preprocess(s)
             for s in sentences_raw]

print(f"Training sentences: {len(tokenized)}")
print(f"\nSample tokenized:")
for i in range(3):
    print(f"  {i+1}. {tokenized[i]}")

# ===== TRAIN WORD2VEC =====
print("\n" + "="*50)
print("PART 3: TRAIN WORD2VEC MODEL")
print("="*50)

model_w2v = Word2Vec(
    sentences=tokenized,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4,
    epochs=100,
    sg=1
)

print(f"Vocabulary size: {len(model_w2v.wv)}")
print(f"Vector size: {model_w2v.vector_size}")
vocab_sample = list(
    model_w2v.wv.key_to_index.keys())[:20]
print(f"\nSample vocabulary: {vocab_sample}")

# ===== WORD VECTORS =====
print("\n" + "="*50)
print("PART 4: WORD VECTORS")
print("="*50)

word = "cricket"
if word in model_w2v.wv:
    vector = model_w2v.wv[word]
    print(f"Vector for '{word}':")
    print(f"Shape: {vector.shape}")
    print(f"First 10 values: "
          f"{vector[:10].round(4)}")

# ===== SIMILAR WORDS =====
print("\n" + "="*50)
print("PART 5: SIMILAR WORDS")
print("="*50)

test_words = ["cricket","python",
              "kunal","learning",
              "food","friends"]

for word in test_words:
    if word in model_w2v.wv:
        similar = model_w2v.wv.most_similar(
                    word, topn=5)
        print(f"\n'{word}' most similar:")
        for sim_word, score in similar:
            bar = "█" * int(score * 20)
            print(f"  {sim_word:15}: "
                  f"{bar} {score:.4f}")

# ===== WORD RELATIONSHIPS =====
print("\n" + "="*50)
print("PART 6: WORD RELATIONSHIPS")
print("="*50)

analogies = [
    ("cricket","sport","python",
     "programming language?"),
    ("india","cricket","nagpur",
     "oranges?"),
]

print("Word analogies (A - B + C = ?):")
for word_a, word_b, word_c, expected in analogies:
    try:
        if all(w in model_w2v.wv
               for w in [word_a,word_b,word_c]):
            result = model_w2v.wv.most_similar(
                positive=[word_a, word_c],
                negative=[word_b],
                topn=3
            )
            print(f"\n{word_a} - {word_b}"
                  f" + {word_c} = ?")
            print(f"Expected: {expected}")
            print(f"Got: {[r[0] for r in result]}")
    except:
        print(f"  Skipping...")

# ===== COSINE SIMILARITY =====
print("\n" + "="*50)
print("PART 7: COSINE SIMILARITY")
print("="*50)

word_pairs = [
    ("cricket","sport"),
    ("cricket","python"),
    ("kunal","sonu"),
    ("kunal","cricket"),
    ("learning","studying"),
    ("food","mango"),
    ("python","tensorflow"),
    ("india","nagpur"),
]

print(f"\n{'Word 1':12} {'Word 2':12} "
      f"{'Similarity':10} {'Relationship'}")
print("-" * 55)

for w1, w2 in word_pairs:
    if w1 in model_w2v.wv and \
       w2 in model_w2v.wv:
        sim = model_w2v.wv.similarity(w1, w2)
        if sim > 0.7:
            rel = "Very similar! 🔥"
        elif sim > 0.4:
            rel = "Somewhat similar 😊"
        elif sim > 0.1:
            rel = "Slightly related"
        else:
            rel = "Different 😐"
        print(f"{w1:12} {w2:12} "
              f"{sim:10.4f} {rel}")

# ===== VISUALIZATION =====
print("\n" + "="*50)
print("PART 8: VISUALIZATION")
print("="*50)

words_to_plot = [
    "cricket","sport","match","india",
    "python","learning","machine","deep",
    "food","mango","biryani","nagpur",
    "kunal","sonu","friends","study",
]
words_to_plot = [w for w in words_to_plot
                 if w in model_w2v.wv]

vectors = np.array(
    [model_w2v.wv[w] for w in words_to_plot])

pca = PCA(n_components=2)
vectors_2d = pca.fit_transform(vectors)

fig, axes = plt.subplots(1, 2,
                          figsize=(16, 6))

axes[0].scatter(vectors_2d[:,0],
                vectors_2d[:,1],
                c="steelblue", s=100)
for i, word in enumerate(words_to_plot):
    axes[0].annotate(
        word,
        xy=(vectors_2d[i,0],
            vectors_2d[i,1]),
        fontsize=12,
        ha="center", va="bottom")
axes[0].set_title(
    "Word Embeddings - PCA 2D")
axes[0].set_xlabel("PCA Component 1")
axes[0].set_ylabel("PCA Component 2")
axes[0].grid(True, alpha=0.3)

sim_words = ["cricket","python",
             "kunal","food",
             "learning","india"]
sim_words = [w for w in sim_words
             if w in model_w2v.wv]

sim_matrix = np.zeros(
    (len(sim_words), len(sim_words)))
for i, w1 in enumerate(sim_words):
    for j, w2 in enumerate(sim_words):
        sim_matrix[i,j] = \
            model_w2v.wv.similarity(w1, w2)

im = axes[1].imshow(
    sim_matrix, cmap="YlOrRd",
    vmin=0, vmax=1)
axes[1].set_xticks(range(len(sim_words)))
axes[1].set_yticks(range(len(sim_words)))
axes[1].set_xticklabels(
    sim_words, rotation=45)
axes[1].set_yticklabels(sim_words)
plt.colorbar(im, ax=axes[1])
axes[1].set_title(
    "Word Similarity Heatmap")

for i in range(len(sim_words)):
    for j in range(len(sim_words)):
        axes[1].text(
            j, i,
            f"{sim_matrix[i,j]:.2f}",
            ha="center", va="center",
            fontsize=8)

plt.tight_layout()
plt.show()

# ===== SENTENCE EMBEDDINGS =====
print("\n" + "="*50)
print("PART 9: SENTENCE EMBEDDINGS")
print("="*50)

def sentence_embedding(sentence, model):
    tokens = preprocess(sentence)
    vectors = [model.wv[w]
               for w in tokens
               if w in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    return np.zeros(model.vector_size)

test_sentences = [
    "Kunal loves cricket",
    "Cricket match was exciting",
    "Python machine learning",
    "Deep learning neural networks",
    "Nagpur oranges are delicious",
    "Indian food is amazing",
]

embeddings = [sentence_embedding(s, model_w2v)
              for s in test_sentences]
embeddings = np.array(embeddings)
sim_matrix = cosine_similarity(embeddings)

print(f"\n{'Sentence 1':30} "
      f"{'Sentence 2':30} {'Sim':6}")
print("-" * 70)
for i in range(len(test_sentences)):
    for j in range(i+1,
                   len(test_sentences)):
        sim = sim_matrix[i][j]
        if sim > 0.5:
            print(
                f"{test_sentences[i][:28]:30}"
                f"{test_sentences[j][:28]:30}"
                f"{sim:.4f} ← similar!")

# ===== SAVE AND LOAD =====
print("\n" + "="*50)
print("PART 10: SAVE AND LOAD")
print("="*50)

model_w2v.save("word2vec_model.bin")
print("Model saved!")

loaded_model = Word2Vec.load(
    "word2vec_model.bin")
print("Model loaded!")

test_word = "kunal"
if test_word in loaded_model.wv:
    similar = loaded_model.wv.most_similar(
        test_word, topn=3)
    print(f"\n'{test_word}' similar words:")
    for w, s in similar:
        print(f"  {w}: {s:.4f}")

# ===== SUMMARY =====
print("\n" + "="*50)
print("DAY 37 SUMMARY")
print("="*50)
print(f"✅ Word Embeddings concept")
print(f"✅ Word2Vec Skip-gram model")
print(f"✅ Vocabulary: "
      f"{len(model_w2v.wv)} words")
print(f"✅ Vector size: "
      f"{model_w2v.vector_size} dimensions")
print(f"✅ Similar words finding")
print(f"✅ Word analogies")
print(f"✅ Cosine similarity")
print(f"✅ PCA visualization")
print(f"✅ Sentence embeddings")
print(f"✅ Save and load model")
print(f"\n🔤 Word2Vec — COMPLETE!")
print(f"🚀 Tomorrow: Transformers & BERT!")