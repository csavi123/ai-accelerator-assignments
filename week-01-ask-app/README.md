# Ask App

A tiny app that lets you ask Claude a question.

- **Backend:** FastAPI (`backend.py`) with three endpoints:
  - `POST /ask` with `{"question": "..."}` streams back Claude's answer as plain text (at least 1 word)
  - `POST /summarise` with `{"text": "..."}` returns `{"summary": "...", "key_points": [...]}` (the text must be at least 5 words, otherwise it returns a 400)
  - `POST /analyze-sentiment` with `{"text": "..."}` returns `{"sentiment": "positive|negative|neutral|mixed", "confidence": 0.0-1.0, "explanation": "..."}` (at least 1 word)
- **Frontend:** Streamlit (`frontend.py`) with three tabs: **Ask**, **Summarise** and **Sentiment**, one per endpoint

## Setup

```bash
cd ask-app
python -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then open .env and paste your Anthropic API key
```

## Run

You need **two terminals** (activate the venv in both).

**Terminal 1: start the backend**

```bash
uvicorn backend:app --reload
```

The API runs at http://localhost:8000. You can try it in your browser at http://localhost:8000/docs.

**Terminal 2: start the frontend**

```bash
streamlit run frontend.py
```

Streamlit opens a page in your browser. Type a question and click **Ask**.

## Error handling

- **Timeout:** each call to Claude gives up after 60 seconds. The frontend waits up to 5 minutes for the backend.
- **Retries:** the Anthropic SDK retries timeouts, rate limits, server errors and network problems up to 3 times, waiting longer each time.
- **No 500 errors:** failures are logged (with the full error) in the backend terminal, and the user gets a friendly message instead:

| Problem | Status | Example message |
|---|---|---|
| Claude took too long | 504 | "Claude took too long to respond..." |
| Too many requests | 429 | "Too many requests right now..." |
| Can't reach Claude | 503 | "Could not connect to Claude..." |
| Not enough words | 400 | "Please enter at least 1 word (you entered 0)." |
| Input Claude can't process | 400 | "Claude couldn't process that input..." |
| Claude refused / no usable answer | 502 | "Claude couldn't produce a result..." |
| Anything else (e.g. bad API key) | 503 | "Something went wrong on our side..." |

`/ask` returns a 400 if the question is empty. Once it starts streaming it returns 200, and if something fails after that, the answer ends with a ⚠️ message instead.
