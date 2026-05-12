# Loan Risk Service

Loan Risk Service is an end-to-end machine learning project for loan default prediction. The repository includes:

- a LightGBM training pipeline
- a FastAPI inference service
- a Gradio frontend for manual testing
- CSV-based request logging and post-review labeling
- optional GCP sync for request logs
- a RAG document query module using sentence-transformers and FAISS
- an Ollama `gemma4:26b` integration for grounded document answers
- a rule-based Agent Router that chooses RAG, prediction, or a combined flow

The current codebase is designed for local development first, with Docker support for running the API and frontend together. Ollama is expected to run locally on the host machine in the first version because `gemma4:26b` is large.

## What the Project Does

The system accepts applicant and loan information, applies the same feature engineering used during training, and returns a default-risk probability.

Based on the predicted probability, the UI and API use the following decision bands:

- `>= 0.50`: high risk
- `>= 0.20 and < 0.50`: manual review / medium risk
- `< 0.20`: low risk

Every prediction request is logged to `data/loan_requests_full.csv` with a generated `case_id`, so the case can later be labeled through the API.

The RAG module supports questions about loan conditions, application flow, review rules, supplementary documents, and manual review policy. The Agent Router is rule-based: it does not act as an autonomous agent, and it does not guess missing prediction fields.

## Repository Layout

```text
loan-risk-service/
|-- app/
|   |-- main.py                        # FastAPI app and API endpoints
|   |-- agent/
|   |   |-- agent_router.py            # rule-based route selection
|   |   |-- ollama_client.py           # Ollama /api/chat client
|   |   `-- prediction_tool.py         # reuses current /predict schema and logic
|   |-- rag/
|   |   |-- document_loader.py         # markdown/txt/pdf loader
|   |   |-- text_splitter.py           # markdown-aware chunking
|   |   |-- index_builder.py           # sentence-transformers + FAISS index build
|   |   |-- retriever.py               # top-k cosine retrieval
|   |   `-- rag_service.py             # retrieval + grounded LLM answer
|   `-- services/                      # logging, GCP upload, notifications
|-- data/
|   |-- documents/
|   |   |-- loan_policy.md
|   |   |-- review_rules.md
|   |   `-- application_flow.md
|   |-- loan_train_36000.csv
|   |-- loan_test_9000.csv
|   `-- loan_requests_full.csv
|-- models/
|   |-- lgbm_best_model.zip
|   |-- lgbm_best_model_features.json
|   `-- rag_index/
|       |-- faiss.index
|       |-- chunks.json
|       `-- index_metadata.json
|-- frontend/
|   |-- gradio_app.py
|   |-- api_client.py
|   `-- static/styles.css
|-- pipeline/
|-- docker-compose.yml
|-- Dockerfile.api
|-- Dockerfile.frontend
|-- .env.example
|-- requirements.txt
`-- README.md
```

## Main Features

### 1. Model training

`main_train.py` runs the training pipeline:

1. load `data/loan_train_36000.csv`
2. drop missing rows
3. apply feature engineering
4. split train and validation sets
5. train a LightGBM binary classifier
6. print evaluation metrics
7. save the trained model and feature list

Training-related files:

- `pipeline/train.py`
- `pipeline/evaluate.py`
- `pipeline/feature_engineering.py`
- `pipeline/save.py`

### 2. Shared feature engineering

The inference path reuses `pipeline/feature_engineering.py`, so training and prediction stay aligned.

Current feature engineering includes:

- categorical mapping for `person_gender`
- categorical mapping for `person_home_ownership`
- categorical mapping for `loan_intent`
- derived feature `log_income = log1p(person_income)`
- derived feature `interest_pressure = loan_int_rate * loan_percent_income`

### 3. FastAPI service

Current endpoints:

- `GET /`
  Returns API health information, model status, fallback alert status, and GCP sync runtime status.
- `POST /predict`
  Accepts applicant data, runs prediction, logs the case, and returns probability, risk level, and `case_id`.
- `POST /label`
  Writes a manual `loan_status` back to the CSV log for an existing `case_id`.
- `POST /admin/gcp/upload`
  Manually triggers upload of pending log rows to GCP destinations when enabled.
- `POST /rag/query`
  Runs document retrieval and asks Ollama to answer only from retrieved document chunks.
- `POST /agent/ask`
  Uses a rule-based router to choose RAG, prediction, or prediction followed by RAG.

Example `POST /predict` response:

```json
{
  "status": "success",
  "probability": 0.23,
  "risk_level": "medium",
  "decision": "manual review",
  "case_id": "CASE_20260421_12345",
  "model_status": "loaded"
}
```

### 4. RAG document query

The RAG module reads documents from `data/documents/`, splits them into chunks, embeds each chunk with sentence-transformers, and stores a FAISS index under `models/rag_index/`.

The first version uses:

```env
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Retrieval uses cosine similarity with:

```text
embedding normalize -> FAISS IndexFlatIP -> top-k retrieval
```

`chunks.json` stores metadata with each chunk:

```json
{
  "chunk_id": "loan_policy_001",
  "source_file": "loan_policy.md",
  "section_title": "貸款資格條件",
  "text": "申請人需年滿 20 歲，具備穩定收入來源..."
}
```

`index_metadata.json` records the embedding model, similarity method, chunk size, overlap, and chunk count.

### 5. Ollama grounded answer generation

The RAG service calls Ollama through `/api/chat` with:

```env
OLLAMA_MODEL=gemma4:26b
```

The prompt instructs the model to:

- answer only from retrieved document chunks
- say `目前文件中沒有明確說明` when the documents do not contain enough information
- avoid inventing loan policies, rates, review rules, or流程
- include source file and chunk information
- answer in Traditional Chinese

If `gemma4:26b` is not available, the API returns a clear error message such as:

```json
{
  "error": "Ollama model gemma4:26b is not available. Please run: ollama pull gemma4:26b"
}
```

### 6. Rule-based Agent Router

The Agent Router supports these routes:

- `rag_only`: document question without enough prediction fields
- `predict_only`: prediction question with all fields required by the current `/predict` schema
- `predict_then_rag`: prediction plus rule explanation, with all prediction fields available
- `need_more_info`: prediction intent exists but required fields are missing

The router reuses the existing `/predict` Pydantic schema and prediction function. It does not create a separate prediction schema and does not guess missing applicant data.

Required prediction fields are the existing API fields:

```text
person_age, person_gender, person_education, person_income, person_emp_exp,
person_home_ownership, loan_amnt, loan_intent, loan_int_rate,
loan_percent_income, cb_person_cred_hist_length, credit_score
```

## Local Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Start Ollama locally:

```bash
ollama pull gemma4:26b
ollama run gemma4:26b
```

Build the RAG index:

```bash
python -m app.rag.index_builder
```

Start the API:

```bash
python run_api.py
```

Start the frontend in another terminal:

```bash
python gradio_app.py
```

Default local URLs:

- API docs: `http://localhost:8000/docs`
- frontend: `http://localhost:7860`
- Ollama: `http://localhost:11434`

## Docker Compose

```bash
docker compose up --build
```

This starts:

- `loan-api` on port `8000`
- `loan-frontend` on port `7860`

Ollama is not bundled into Docker in this version. If the API runs directly on the host, use:

```env
OLLAMA_BASE_URL=http://localhost:11434
```

If the API runs in Docker and Ollama runs on the Windows host, use:

```env
OLLAMA_BASE_URL=http://host.docker.internal:11434
```

The included `docker-compose.yml` defaults the API container to `http://host.docker.internal:11434`.

## Environment Variables

See `.env.example`:

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma4:26b
OLLAMA_TIMEOUT=120
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Other existing variables include model paths, log paths, GCP upload settings, frontend `BASE_URL`, and fallback notification settings.

## API Examples

### POST /rag/query

Request:

```json
{
  "question": "中風險案件要怎麼處理？",
  "top_k": 3
}
```

Response:

```json
{
  "answer": "根據審核規則，中風險案件需要進入人工覆核...",
  "sources": [
    {
      "source_file": "review_rules.md",
      "section_title": "中風險案件處理",
      "chunk_id": "review_rules_002",
      "text": "中風險案件需由審核人員進一步確認收入證明..."
    }
  ]
}
```

### POST /agent/ask

Request:

```json
{
  "question": "申請人年齡 35 歲，男性，學士，收入 50000，就業年資 6，租賃，貸款金額 20000，個人周轉，利率 12%，負債比 40%，信用歷史長度 5，信用分數 680，風險高嗎？如果是中風險要看哪些資料？",
  "top_k": 3
}
```

Response:

```json
{
  "route": "predict_then_rag",
  "prediction": {
    "probability": 0.42,
    "risk_level": "medium",
    "case_id": "CASE_20260512_12345",
    "model_status": "loaded"
  },
  "answer": "模型預測違約風險機率為 42.00%，風險分級為 medium...",
  "sources": [
    {
      "source_file": "review_rules.md",
      "section_title": "中風險案件處理"
    }
  ]
}
```

## Test Cases

### 1. Pure document query

Input:

```text
中風險案件要怎麼處理？
```

Expected route:

```text
rag_only
```

### 2. Pure risk prediction

Input:

```text
申請人年齡 35 歲，男性，學士，收入 50000，就業年資 6，租賃，貸款金額 20000，個人周轉，利率 12%，負債比 40%，信用歷史長度 5，信用分數 680，風險高嗎？
```

Expected route:

```text
predict_only
```

If the question only provides partial fields such as income, loan amount, credit score, and interest rate, the expected route is `need_more_info`.

### 3. Prediction plus document explanation

Input:

```text
申請人年齡 35 歲，男性，學士，收入 50000，就業年資 6，租賃，貸款金額 20000，個人周轉，利率 12%，負債比 40%，信用歷史長度 5，信用分數 680，這個申請人風險高嗎？如果是中風險，人工審核要看哪些資料？
```

Expected route:

```text
predict_then_rag
```

## Portfolio Description

Recommended wording:

```text
延伸開發基於 RAG 的貸款文件查詢模組，使用 sentence-transformers 將貸款條件、申請流程與審核規則文件轉為向量，並以 FAISS 建立本地化語意檢索流程，支援規則問答與來源引用。
```

```text
整合 Ollama gemma4:26b 與 rule-based Agent Router，根據使用者問題類型自動選擇 RAG 文件檢索或 FastAPI 風險預測端點，支援貸款規則查詢、預測結果摘要與混合式問答情境。
```

Avoid overstating the project as an autonomous AI agent, complete model explainability system, or automated loan approval system.

## Development Notes

- The saved feature list in `models/lgbm_best_model_features.json` must match the trained model.
- `loan_requests_full.csv` is part of runtime behavior, not just sample data.
- Build the RAG index again after editing files in `data/documents/`.
- GCP upload is optional and disabled by default unless the related environment variables are enabled.

## Suggested Next Improvements

- move request logging from CSV to a database
- add automated tests for training, inference, RAG, and API endpoints
- add authentication or admin protection for cloud upload operations
- add a UI button to rebuild the RAG index in development environments

## Author

Henry Wu
