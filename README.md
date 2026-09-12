# SMS Spam Detection API

An end-to-end NLP and containerized microservice built to classify incoming SMS notifications as legitimate (`ham`) or `spam` in real time.
<img width="2560" height="1440" alt="Spam_Detection" src="https://github.com/user-attachments/assets/5d1eefe2-08a3-4408-9b47-a4c0e132486a" />

---

## 📌 Project Overview & Scenario

At **Zenith Reach** (a Lagos-based platform helping small businesses across Nigeria send automated SMS updates such as order confirmations, delivery tracking, and appointment reminders), spam messages began penetrating outbound pipelines. 

This project delivers an end-to-end Machine Learning solution:
1. **Model Engineering:** Preprocessing raw text, vectorizing messages, and training a high-performing classification model.
2. **API Architecture:** Designing a RESTful API with strict input validation, single-load model initialization, and standard HTTP response codes.
3. **Containerization:** Packaging the service inside Docker with an enterprise-ready server for zero-drift deployment.

---

## 🗂️ Project Structure

```text
SPAM_DETECTION_API/
├── app/
│   ├── __init__.py
│   ├── main.py              # API application routes and inference logic
│   ├── schemas.py           # Pydantic request/response data contracts
│   └── model/
│       ├── model.joblib      # Trained Multinomial Naive Bayes model
│       └── vectorizer.joblib # Fitted CountVectorizer
├── data/
│   └── SMSSpamCollection    # Raw SMS labeled dataset
├── EDA/
│   └── 9b. Spam Ham Classification Project Using BOW And TFIDF And ML.ipynb
├── training/
│   └── train.py             # Script for data cleaning, training, and artifact export
├── .dockerignore
├── .gitignore
├── Dockerfile
├── README.md
└── requirements.txt
```

---

## ⚙️ Phase 1: Preprocessing & Model Training

### 1. Data Cleaning & Inspection
- Handled missing records and eliminated duplicate messages to prevent biased feature frequencies.
- Evaluated class distribution (~87% ham vs. ~13% spam) to ensure proper evaluation metrics were selected instead of relying strictly on accuracy.

### 2. Text Preprocessing Pipeline
- **Regex Cleaning:** Stripped punctuation, numbers, and special characters, keeping only alphabetic characters.
- **Lowercasing:** Standardized token representations across all entries.
- **Stopwords Removal:** Filtered out common English stopwords using NLTK to retain high-information tokens.
- **Stemming:** Applied Porter Stemmer to reduce terms to their linguistic roots (e.g., `running`, `runs` → `run`).

### 3. Feature Extraction (Vectorization)
- Used **Bag of Words (`CountVectorizer`)** with:
  - `max_features=2500` (focusing on the most relevant tokens)
  - `ngram_range=(1, 2)` (capturing both single keywords and two-word combinations such as *free entry*, *claim call*, *urgent contact*)
- Compared against TF-IDF during exploratory data analysis.

### 4. Model Training & Evaluation
Trained a **Multinomial Naive Bayes (`MultinomialNB`)** classifier on an 80/20 train/test split:

```text
              precision    recall  f1-score   support

         ham       0.99      0.99      0.99       966
        spam       0.95      0.93      0.94       149

    accuracy                           0.98      1115
   macro avg       0.97      0.96      0.96      1115
weighted avg       0.98      0.98      0.98      1115
```

- **Accuracy:** `98.39%`
- **Key Metric Highlight:** Spam precision achieved `95%` and recall achieved `93%`, preventing legitimate business messages from being mistakenly blocked while filtering harmful spam.

Both `model.joblib` and `vectorizer.joblib` are serialized to `app/model/` to guarantee zero feature skew during inference.

---

## 🚀 Phase 2: API Endpoints & Request/Response Contracts

The API loads the trained vectorizer and model **once during startup** rather than on each request to maximize throughput.

### 1. Health Check
- **Endpoint:** `GET /`
- **Response:** `200 OK`
```json
{
  "status": "ok",
  "message": "Spam Detection API is running"
}
```

### 2. Spam Prediction
- **Endpoint:** `POST /predict`
- **Request Body (`application/json`):**
```json
{
  "message": "Congratulations! You have won $1,000"
}
```
- **Response Body (`200 OK`):**
```json
{
  "message": "Congratulations! You have won $1,000",
  "prediction": "spam"
}
```

### 3. Validation & Error Handling
- Automatic validation via **Pydantic schemas** (`schemas.py`) rejects non-string or malformed requests with `422 Unprocessable Entity` (or `400 Bad Request` in manual validation scenarios).
- Interactive Swagger UI documentation is available out of the box at `/docs`.

---

## 💻 Local Setup & Execution

### Prerequisites
- Python 3.10+
- Virtual Environment tool (`venv`)

### 1. Clone and Setup Environment
```bash
# Clone repository
git clone <your-repo-link>
cd SPAM_DETECTION_API

# Create and activate virtual environment
python -m venv venv

# Windows (Command Prompt / PowerShell):
venv\Scripts\activate

# macOS / Linux:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the Model
```bash
python training/train.py
```

### 4. Run the API Locally
```bash
uvicorn app.main:app --reload
```
The server will start at `http://127.0.0.1:8000`. Access Swagger documentation at `http://127.0.0.1:8000/docs`.

---

## 🐳 Phase 3: Docker Containerization

### 1. Build the Docker Image
```bash
docker build -t sms-spam-api .
```

### 2. Run the Container
```bash
docker run -d -p 8000:8000 --name spam-api-container sms-spam-api
```

### 3. Test Container Prediction
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"message": "Urgent! Claim your prize now"}'
```

---

## 📝 Key Takeaways & Architecture Notes
- **FastAPI vs. Flask:** FastAPI offers automatic request validation via Pydantic and self-documenting OpenAPI interfaces (`/docs`), whereas standard Flask requires manual JSON payload inspection and manual status code error management.
- **Production Server:** Using Uvicorn/Gunicorn as an ASGI/WSGI production server instead of built-in development servers provides worker management and connection stability under concurrent traffic.
🧑‍💻 Author & Contact
Author: Mayowa Adeboye

Email: [adeboyemayowa86@gmail.com](mailto:adeboyemayowa86@gmail.com)

LinkedIn: [mayowaadeboye](https://linkedin.com/in/mayowaadeboye)

GitHub: [MayowaDATA](https://github.com/MayowaDATA)

