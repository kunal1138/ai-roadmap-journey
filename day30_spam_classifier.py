# DAY 30: Spam Classifier
# NLP - Natural Language Processing 📧

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Dataset
messages = [
    # SPAM messages
    ("Win FREE iPhone now! Click here!", "spam"),
    ("You won ₹10,00,000! Claim now!", "spam"),
    ("FREE money transfer! Act now!", "spam"),
    ("Congratulations! You won lottery!", "spam"),
    ("Click here to claim your prize!", "spam"),
    ("Buy cheap medicines online now!", "spam"),
    ("Make money fast! Work from home!", "spam"),
    ("FREE offer expires today! Hurry!", "spam"),
    ("You are selected for cash prize!", "spam"),
    ("Urgent: Your account needs update!", "spam"),
    ("Win big prizes click this link!", "spam"),
    ("Get rich quick scheme available!", "spam"),
    ("Free vacation package won by you!", "spam"),
    ("Investment opportunity 500% returns!", "spam"),
    ("Claim your free gift card today!", "spam"),
    ("Hot singles in your area now!", "spam"),
    ("Lose weight fast with this pill!", "spam"),
    ("Your loan approved call now!", "spam"),
    ("Free credit score check now!", "spam"),
    ("Exclusive deal only for you today!", "spam"),

    # HAM messages
    ("Hey Kunal, are you coming today?", "ham"),
    ("Meeting at 3pm please confirm", "ham"),
    ("Can you share the notes please?", "ham"),
    ("Happy birthday bro have great day", "ham"),
    ("Let us study together tonight", "ham"),
    ("Assignment submission is tomorrow", "ham"),
    ("Mom said dinner is ready come home", "ham"),
    ("Match tonight India vs Australia", "ham"),
    ("Did you finish the project work?", "ham"),
    ("Library books due date is Friday", "ham"),
    ("Professor cancelled class today", "ham"),
    ("Sonu called you back please call", "ham"),
    ("Results declared check your marks", "ham"),
    ("New movie releasing this Friday!", "ham"),
    ("Can we meet at college canteen?", "ham"),
    ("Please bring my book tomorrow", "ham"),
    ("Power cut in our area from 2pm", "ham"),
    ("College fest registration open now", "ham"),
    ("Glou said practice is at 5pm", "ham"),
    ("Your package has been delivered", "ham"),
    ("Study group meeting room 204", "ham"),
    ("Cricket practice cancelled today", "ham"),
    ("Durgesh birthday party on Sunday", "ham"),
    ("Internet is slow call provider", "ham"),
    ("Vedu asking about tomorrow plan", "ham"),
]

df = pd.DataFrame(messages, columns=["message","label"])

print("===== SPAM DATASET =====")
print(df.head(10))
print(f"\nTotal messages: {len(df)}")
print(f"Spam: {len(df[df['label']=='spam'])}")
print(f"Ham:  {len(df[df['label']=='ham'])}")

# Text Analysis
print("\n===== TEXT ANALYSIS =====")
df["word_count"] = df["message"].apply(
                    lambda x: len(x.split()))
df["char_count"] = df["message"].apply(len)

print("\nAverage word count:")
print(df.groupby("label")["word_count"].mean().round(2))
print("\nAverage char count:")
print(df.groupby("label")["char_count"].mean().round(2))

# Preprocessing
print("\n===== PREPROCESSING =====")
df["label_num"] = df["label"].map(
                  {"ham": 0, "spam": 1})

X = df["message"]
y = df["label_num"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training messages: {len(X_train)}")
print(f"Testing messages: {len(X_test)}")

# Vectorization
print("\n===== VECTORIZATION =====")
cv = CountVectorizer()
X_train_cv = cv.fit_transform(X_train)
X_test_cv = cv.transform(X_test)

print(f"Vocabulary size: {len(cv.vocabulary_)}")
print(f"Feature matrix shape: {X_train_cv.shape}")

tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Model Comparison
print("\n===== MODEL COMPARISON =====")
print(f"{'Model':25} {'CV Acc':10} {'TFIDF Acc':10}")
print("-" * 47)

models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression()
}

for name, model in models.items():
    model.fit(X_train_cv, y_train)
    cv_acc = accuracy_score(y_test,
              model.predict(X_test_cv))

    model.fit(X_train_tfidf, y_train)
    tfidf_acc = accuracy_score(y_test,
                model.predict(X_test_tfidf))

    print(f"{name:25} {cv_acc*100:8.2f}%  "
          f"{tfidf_acc*100:8.2f}%")

# Best Model
print("\n===== BEST MODEL DETAILS =====")
best_model = MultinomialNB()
best_model.fit(X_train_tfidf, y_train)
y_pred = best_model.predict(X_test_tfidf)

print(f"Accuracy: {accuracy_score(y_test,y_pred)*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=["Ham","Spam"]))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print("Predicted →  Ham  Spam")
print(f"Actual Ham   {cm[0]}")
print(f"Actual Spam  {cm[1]}")

# Predict New Messages
print("\n===== PREDICT NEW MESSAGES =====")
test_messages = [
    "Win free iPhone click now!",
    "Hey Kunal what time is class?",
    "Congratulations you won lottery!",
    "Mom dinner is ready come home",
    "FREE money transfer act now!",
    "Cricket match tonight at 7pm",
    "Claim your prize immediately!",
    "Sonu birthday party on Sunday"
]

predictions = best_model.predict(
    tfidf.transform(test_messages))

print(f"{'Message':35} {'Prediction':10}")
print("-" * 47)
for msg, pred in zip(test_messages, predictions):
    label = "🚫 SPAM" if pred==1 else "✅ HAM"
    print(f"{msg[:35]:35} {label}")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(12,4))

labels = ["Ham", "Spam"]
counts = [len(df[df["label"]=="ham"]),
          len(df[df["label"]=="spam"])]
axes[0].bar(labels, counts,
            color=["green","red"])
axes[0].set_title("Ham vs Spam Count")
axes[0].set_ylabel("Count")

df.boxplot(column="word_count",
           by="label", ax=axes[1])
axes[1].set_title("Word Count by Label")
axes[1].set_xlabel("Label")
axes[1].set_ylabel("Word Count")

plt.tight_layout()
plt.show()

# Summary
print("\n===== PROJECT SUMMARY =====")
print(f"Total messages: {len(df)}")
print(f"Spam messages: {len(df[df['label']=='spam'])}")
print(f"Ham messages: {len(df[df['label']=='ham'])}")
print(f"Vocabulary size: {len(cv.vocabulary_)}")
print(f"Best Model: Naive Bayes + TF-IDF")
print(f"Accuracy: {accuracy_score(y_test,y_pred)*100:.2f}%")