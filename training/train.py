from pathlib import Path
import re

import joblib
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "SMSSpamCollection"
MODEL_DIR = BASE_DIR / "app" / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print(f"Loading data from {DATA_PATH}")
messages = pd.read_csv(DATA_PATH, sep="\t", names=["label", "message"])

nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))  # built once, not on every loop iteration
ps = PorterStemmer()

corpus = []
for i in range(len(messages)):
    review = re.sub("[^a-zA-Z]", " ", messages["message"][i])
    review = review.lower().split()
    review = [ps.stem(word) for word in review if word not in stop_words]
    corpus.append(" ".join(review))

y = (messages["label"] == "spam").astype(int).values

X_train, X_test, y_train, y_test = train_test_split(
    corpus, y, test_size=0.20, random_state=42
)

cv = CountVectorizer(max_features=2500, ngram_range=(1, 2))
X_train_vec = cv.fit_transform(X_train).toarray()
X_test_vec = cv.transform(X_test).toarray()

spam_detect_model = MultinomialNB().fit(X_train_vec, y_train)

y_pred = spam_detect_model.predict(X_test_vec)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))

joblib.dump(spam_detect_model, MODEL_DIR / "model.joblib")
joblib.dump(cv, MODEL_DIR / "vectorizer.joblib")
print(f"Saved model.joblib and vectorizer.joblib to {MODEL_DIR}")