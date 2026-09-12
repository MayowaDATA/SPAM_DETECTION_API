from pathlib import Path

import joblib
from fastapi import FastAPI

from app.schemas import Message, Prediction

MODEL_DIR = Path(__file__).resolve().parent / "model"
MODEL_PATH = MODEL_DIR / "model.joblib"
VECTORIZER_PATH = MODEL_DIR / "vectorizer.joblib"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

app = FastAPI(
    title="Spam Detection API",
    description="Classifies a message as Spam or Ham using a trained Naive Bayes model.",
)


@app.get("/")
def health_check():
    """Quick sanity check that the server is up, before testing /predict."""
    return {"status": "ok", "message": "Spam Detection API is running"}


@app.post("/predict", response_model=Prediction)
def predict(payload: Message):
    """Classify a single message as spam or ham."""
    vectorized = vectorizer.transform([payload.message]).toarray()
    result = model.predict(vectorized)[0]
    label = "spam" if result == 1 else "ham"
    return Prediction(message=payload.message, prediction=label)
