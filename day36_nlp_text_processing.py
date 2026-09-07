# DAY 36: NLP - Natural Language Processing
# Text Processing & Sentiment Analysis 📝

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import string
from collections import Counter

import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Download NLTK data
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt_tab")
nltk.download("averaged_perceptron_tagger")

print("NLP Libraries loaded! 📝")

# ===== PART 1: TEXT BASICS =====
print("\n" + "="*50)
print("PART 1: TEXT BASICS")
print("="*50)

text = """
Natural Language Processing is amazing!
Machine Learning helps computers understand text.
Deep Learning models like BERT and GPT are powerful.
Python is the best language for AI and NLP.
Kunal is learning NLP on his mobile phone!
"""

print("Original text:")
print(text)

sentences = sent_tokenize(text.strip())
print(f"\nSentences found: {len(sentences)}")
for i, sent in enumerate(sentences):
    print(f"  {i+1}. {sent}")

words = word_tokenize(text.lower())
print(f"\nTotal words: {len(words)}")
print(f"First 15 words: {words[:15]}")

# ===== PART 2: PREPROCESSING =====
print("\n" + "="*50)
print("PART 2: TEXT PREPROCESSING")
print("="*50)

def preprocess_text(text):
    text = text.lower()
    text = text.translate(
        str.maketrans("", "",
        string.punctuation))
    text = re.sub(r"\d+", "", text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    tokens = [w for w in tokens
              if w not in stop_words]
    tokens = [w for w in tokens if len(w) > 2]
    return tokens

sample = "The Quick Brown Fox Jumps Over The Lazy Dog!!!"
processed = preprocess_text(sample)
print(f"\nOriginal: {sample}")
print(f"Processed: {processed}")

print("\nProcessed sentences:")
for sent in sentences:
    processed = preprocess_text(sent)
    print(f"  Original: {sent[:40]}...")
    print(f"  Processed: {processed}\n")

# ===== PART 3: STEMMING vs LEMMATIZATION =====
print("\n" + "="*50)
print("PART 3: STEMMING vs LEMMATIZATION")
print("="*50)

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

words_to_compare = [
    "running","runs","ran",
    "studying","studies","studied",
    "playing","plays","played",
    "better","good","best",
    "coding","codes","coded"
]

print(f"{'Word':15} {'Stemmed':15} {'Lemmatized':15}")
print("-" * 45)
for word in words_to_compare:
    stemmed = stemmer.stem(word)
    lemmatized = lemmatizer.lemmatize(word, pos="v")
    print(f"{word:15} {stemmed:15} {lemmatized:15}")

# ===== PART 4: WORD FREQUENCY =====
print("\n" + "="*50)
print("PART 4: WORD FREQUENCY ANALYSIS")
print("="*50)

reviews = [
    "This movie was absolutely amazing and fantastic",
    "Great film with wonderful performances",
    "Terrible movie waste of time and money",
    "Awful film horrible acting and bad story",
    "The best movie I have ever seen in my life",
    "Outstanding performance by all actors",
    "Worst movie ever made completely boring",
    "Loved every minute of this incredible film",
    "Disappointing film with poor direction",
    "Masterpiece of cinema truly remarkable"
]

all_text = " ".join(reviews).lower()
all_words = word_tokenize(all_text)
stop_words = set(stopwords.words("english"))
filtered = [w for w in all_words
            if w not in stop_words
            and w.isalpha()
            and len(w) > 2]

freq = Counter(filtered)
print("\nTop 15 most frequent words:")
for word, count in freq.most_common(15):
    bar = "█" * count
    print(f"  {word:15}: {bar} ({count})")

# ===== PART 5: BAG OF WORDS =====
print("\n" + "="*50)
print("PART 5: BAG OF WORDS")
print("="*50)

sample_docs = [
    "I love cricket and football",
    "Cricket is my favourite sport",
    "Football match was exciting",
    "I hate boring games"
]

cv = CountVectorizer()
bow_matrix = cv.fit_transform(sample_docs)

print(f"Vocabulary: {cv.vocabulary_}")
print(f"\nMatrix shape: {bow_matrix.shape}")
df_bow = pd.DataFrame(
    bow_matrix.toarray(),
    columns=cv.get_feature_names_out()
)
print(df_bow)

# ===== PART 6: TF-IDF =====
print("\n" + "="*50)
print("PART 6: TF-IDF")
print("="*50)

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(sample_docs)

df_tfidf = pd.DataFrame(
    tfidf_matrix.toarray().round(3),
    columns=tfidf.get_feature_names_out()
)
print("TF-IDF Matrix:")
print(df_tfidf)
print("\nBoW  → just counts words")
print("TF-IDF → weights by importance!")

# ===== PART 7: SENTIMENT ANALYSIS =====
print("\n" + "="*50)
print("PART 7: SENTIMENT ANALYSIS")
print("="*50)

reviews_data = [
    ("This movie was absolutely amazing", "positive"),
    ("Great film wonderful performances", "positive"),
    ("The best movie I have ever seen", "positive"),
    ("Outstanding performance by actors", "positive"),
    ("Loved every minute incredible film", "positive"),
    ("Masterpiece of cinema remarkable", "positive"),
    ("Brilliant story and great acting", "positive"),
    ("Fantastic movie highly recommended", "positive"),
    ("Excellent film must watch everyone", "positive"),
    ("Superb direction and great story", "positive"),
    ("Amazing cinematography loved it", "positive"),
    ("Perfect film beautiful experience", "positive"),
    ("Wonderful movie heartwarming story", "positive"),
    ("Incredible acting emotional film", "positive"),
    ("Stunning visuals amazing story", "positive"),
    ("Kunal loved this cricket movie!", "positive"),
    ("Sonu and friends enjoyed film", "positive"),
    ("Best Bollywood film ever made", "positive"),
    ("Durgesh recommended this great movie", "positive"),
    ("Om watched twice amazing film", "positive"),
    ("Terrible movie waste of time money", "negative"),
    ("Awful film horrible acting bad", "negative"),
    ("Worst movie ever made boring", "negative"),
    ("Disappointing film poor direction", "negative"),
    ("Hated every minute terrible film", "negative"),
    ("Complete disaster awful movie", "negative"),
    ("Poor story bad acting boring", "negative"),
    ("Waste of money terrible experience", "negative"),
    ("Horrible film never watch again", "negative"),
    ("Dreadful movie poor performance", "negative"),
    ("Pathetic story bad direction", "negative"),
    ("Disgusting film waste of time", "negative"),
    ("Boring movie fell asleep watching", "negative"),
    ("Terrible acting ruined good story", "negative"),
    ("Worst film decade avoid watching", "negative"),
    ("Glou said terrible boring movie", "negative"),
    ("Vedu hated this awful film", "negative"),
    ("Tantan found movie very boring", "negative"),
    ("Worst Bollywood film ever made", "negative"),
    ("Sam walked out terrible movie", "negative"),
]

df_reviews = pd.DataFrame(
    reviews_data,
    columns=["review","sentiment"])

print(f"Total reviews: {len(df_reviews)}")
print(df_reviews["sentiment"].value_counts())

X = df_reviews["review"]
y = df_reviews["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

cv_vec = CountVectorizer(
    stop_words="english",
    ngram_range=(1,2))
X_train_cv = cv_vec.fit_transform(X_train)
X_test_cv = cv_vec.transform(X_test)

tfidf_vec = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1,2))
X_train_tfidf = tfidf_vec.fit_transform(X_train)
X_test_tfidf = tfidf_vec.transform(X_test)

print("\n===== MODEL COMPARISON =====")
print(f"{'Model':25} {'BoW':10} {'TF-IDF':10}")
print("-" * 47)

models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(
                            max_iter=1000)
}

best_acc = 0
best_model = None
best_vec = None

for name, model in models.items():
    model.fit(X_train_cv, y_train)
    bow_acc = accuracy_score(
        y_test, model.predict(X_test_cv))
    model.fit(X_train_tfidf, y_train)
    tfidf_acc = accuracy_score(
        y_test, model.predict(X_test_tfidf))

    if tfidf_acc > best_acc:
        best_acc = tfidf_acc
        best_model = model
        best_vec = tfidf_vec

    print(f"{name:25} {bow_acc*100:8.2f}%  "
          f"{tfidf_acc*100:8.2f}%")

print(f"\n🏆 Best Accuracy: {best_acc*100:.2f}%")

print("\nClassification Report:")
y_pred = best_model.predict(
    best_vec.transform(X_test))
print(classification_report(y_test, y_pred))

# ===== PART 8: PREDICTIONS =====
print("\n" + "="*50)
print("PART 8: PREDICT NEW REVIEWS")
print("="*50)

new_reviews = [
    "This movie was absolutely brilliant!",
    "Terrible film waste of money",
    "Kunal loved this cricket movie!",
    "Boring and disappointing experience",
    "Outstanding performance highly recommend",
    "Worst movie I have ever seen"
]

predictions = best_model.predict(
    best_vec.transform(new_reviews))

print(f"\n{'Review':40} {'Sentiment':10}")
print("-" * 52)
for review, pred in zip(new_reviews, predictions):
    emoji = "😊" if pred=="positive" else "😞"
    print(f"{review[:40]:40} {pred} {emoji}")

# ===== PART 9: VISUALIZATION =====
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

top_words = freq.most_common(10)
words_list = [w[0] for w in top_words]
counts = [w[1] for w in top_words]
axes[0].barh(words_list, counts,
             color="steelblue")
axes[0].set_title("Top 10 Words")
axes[0].set_xlabel("Frequency")

sent_counts = df_reviews[
    "sentiment"].value_counts()
axes[1].pie(sent_counts.values,
            labels=sent_counts.index,
            autopct="%1.1f%%",
            colors=["green","red"])
axes[1].set_title("Sentiment Distribution")

model_names = ["NB BoW","NB TF-IDF",
               "LR BoW","LR TF-IDF"]
accs_list = []
for name, model in models.items():
    model.fit(X_train_cv, y_train)
    accs_list.append(accuracy_score(
        y_test,model.predict(X_test_cv))*100)
    model.fit(X_train_tfidf, y_train)
    accs_list.append(accuracy_score(
        y_test,model.predict(X_test_tfidf))*100)

axes[2].bar(model_names, accs_list,
            color=["blue","navy",
                   "orange","darkorange"])
axes[2].set_title("Model Comparison")
axes[2].set_ylabel("Accuracy %")
axes[2].set_ylim(0, 110)
for i, acc in enumerate(accs_list):
    axes[2].text(i, acc+1, f"{acc:.0f}%",
                 ha="center")
axes[2].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.show()

# ===== SUMMARY =====
print("\n" + "="*50)
print("DAY 36 SUMMARY")
print("="*50)
print(f"✅ Text preprocessing pipeline")
print(f"✅ Tokenization (word + sentence)")
print(f"✅ Stop words removal")
print(f"✅ Stemming vs Lemmatization")
print(f"✅ Word frequency analysis")
print(f"✅ Bag of Words (CountVectorizer)")
print(f"✅ TF-IDF Vectorizer")
print(f"✅ Sentiment Analysis")
print(f"✅ Best accuracy: {best_acc*100:.2f}%")
print(f"\n📝 NLP Day 1 — COMPLETE!")
print(f"🚀 Tomorrow: Word Embeddings & Word2Vec!")