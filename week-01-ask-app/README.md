# Ask App

A small Streamlit app with a FastAPI backend for asking Claude questions, summarising text, and analysing sentiment.

## Features

- `POST /ask` streams newline-delimited JSON (NDJSON): answer chunks, followed by a final event with the answer, model, input/output token counts, total tokens, and estimated cost.
- `POST /summarise` returns a summary and key points. Input must contain at least five words.
- `POST /analyze-sentiment` returns a sentiment, confidence, and explanation.
- Structured Claude output is validated before it is returned. Invalid output is rejected with HTTP 502 rather than passed to the frontend.

## Model and cost

The default model is Claude Haiku 5.5 (`claude-haiku-5-5`), a fast, lower-cost choice for this app's short Q&A and structured tasks. For prompts under 100,000 tokens, the configured rates are $0.10 per million input tokens and $0.50 per million output tokens; `/ask` reports an estimated per-call cost using the actual token counts from Anthropic.

Pricing can change; check [Anthropic's current API pricing](https://platform.claude.com/docs/en/about-claude/pricing). If you change model, also update the configured rates.

## Requirements and setup

- Python 3.9 or later
- An Anthropic API key with API access (API usage may be billed separately)
- Docker and Docker Compose are optional if you prefer the container workflow

From this directory:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # Windows PowerShell: Copy-Item .env.example .env
```

Edit `.env` locally and set `ANTHROPIC_API_KEY` to your key. `.env` is private and must never be committed or shared. Commit `.env.example` only; it contains placeholder/configuration values, not secrets.

## Run locally

Open two terminals in this directory and activate `.venv` in each.

Terminal 1, start the API:

```bash
uvicorn backend:app --reload --port 8000
```

Terminal 2, start the UI:

```bash
streamlit run frontend.py
```

Open <http://localhost:8501> to use the app and <http://localhost:8000/docs> to inspect the API.

### Docker

With `.env` configured, build and run both services:

```bash
docker compose up --build
```

Open <http://localhost:8501>. Stop the containers with `Ctrl+C`, then run `docker compose down`.

## Test the validation guardrail

This demo deliberately injects malformed structured output locally; it does not call Anthropic. It is disabled by default. In `.env`, temporarily set:

```dotenv
ENABLE_GUARDRAIL_DEMO=true
```

Restart the backend (or rebuild/restart Docker), open the **Summarise** tab, and submit text containing at least five words, for example:

```text
force_bad this input tests the structured output validation guardrail
```

The UI should show that Claude returned malformed output and validation blocked the response. The API responds with HTTP 502 instead of returning the invalid object. Without the guardrail, malformed or incomplete fields could reach the UI and cause incorrect output or a rendering failure. Turn the demo back off (`ENABLE_GUARDRAIL_DEMO=false`) after capturing your local screenshot; do not enable this test hook on a public deployment.

## API examples

Run these commands while the local backend is running. `/ask` returns NDJSON: the answer arrives in chunk events, then a final event contains usage and cost.

```bash
curl -N http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"Why do leaves change color in autumn? Answer in two sentences."}'
```

```bash
curl http://localhost:8000/summarise \
  -H 'Content-Type: application/json' \
  -d '{"text":"The community garden opened in May. Volunteers grow vegetables and herbs. The harvest is shared with a neighborhood food pantry."}'
```

```bash
curl http://localhost:8000/analyze-sentiment \
  -H 'Content-Type: application/json' \
  -d '{"text":"The volunteers were delighted with the successful community garden harvest."}'
```

## Deploy to Render

Push the app to GitHub, then create two Render **Web Services** from the repository and branch containing this app. For both services set the root directory to `week-01-ask-app` and select the Docker runtime.

1. **Backend service:** set the start command to `uvicorn backend:app --host 0.0.0.0 --port $PORT`. Add `ANTHROPIC_API_KEY` in Render's environment settings; optionally set the two token-price variables to the current rates. Do not put the key in source control.
2. **Frontend service:** set the start command to `streamlit run frontend.py --server.address 0.0.0.0 --server.port $PORT`. Set `BACKEND_URL` to the backend service's internal URL if both services are in the same Render region, or its public URL otherwise.

Choose the Free plan if available and appropriate for your account; free services can spin down while idle. Anthropic API calls can still incur separate charges. Verify each service's plan and billing settings in Render before deploying.

## Smoke-test notes

For an optional comparison, ask three different questions in the Ask tab. Record the returned answer, `tokens_used`, and `cost_usd` for each; cost varies with prompt and answer length. Do not claim a measured per-call cost until you have run the live calls and captured the displayed values.

## Proof post

For a programme proof post, use only a screenshot or a short screen recording of the app and guardrail test. Credit the programme, and do not include the live application URL, API key, or other secrets.
