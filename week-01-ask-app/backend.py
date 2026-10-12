# backend.py - a tiny FastAPI server that forwards questions to Claude.

import json
import logging
import os
from typing import List, Literal

import anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

# Load variables from the .env file (this is where ANTHROPIC_API_KEY lives).
load_dotenv()

# Set up logging so errors are printed with a time and a level (INFO, ERROR...).
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("ask-app")

MODEL = "claude-haiku-5-5"
MIN_WORDS = 5  # Shorter text isn't worth summarising
INPUT_COST_PER_MILLION_USD = float(os.environ.get("ANTHROPIC_INPUT_COST_PER_MILLION_USD", "1.0"))
OUTPUT_COST_PER_MILLION_USD = float(os.environ.get("ANTHROPIC_OUTPUT_COST_PER_MILLION_USD", "5.0"))

# Create the Claude client. It reads ANTHROPIC_API_KEY from the environment.
# - timeout: give up on a single request after 60 seconds.
# - max_retries: the SDK automatically retries timeouts, rate limits (429),
#   server errors (5xx) and network problems, waiting a bit longer each time.
client = anthropic.Anthropic(timeout=60.0, max_retries=3)

# Create the FastAPI app.
app = FastAPI()


# Raised when Claude answers but we can't use the answer (e.g. it refused).
class UnusableAnswerError(Exception):
    pass


def friendly_error(error):
    """Turn an exception into (HTTP status code, message for the user). Never 500."""
    # APITimeoutError must be checked before APIConnectionError (it's a subclass).
    if isinstance(error, anthropic.APITimeoutError):
        return 504, "Claude took too long to respond. Please try again."
    if isinstance(error, anthropic.RateLimitError):
        return 429, "Too many requests right now. Please wait a moment and try again."
    if isinstance(error, anthropic.APIConnectionError):
        return 503, "Could not connect to Claude. Please try again shortly."
    if isinstance(error, anthropic.BadRequestError):
        return 400, "Claude couldn't process that input. Try shorter or different text."
    if isinstance(error, UnusableAnswerError):
        return 502, str(error)
    # Anything else (bad API key, Claude outage, a bug in our code...).
    return 503, "Something went wrong on our side. Please try again later."


# This middleware wraps every request. If an endpoint raises an error, we log
# it and send back a friendly JSON message instead of a 500 error.
@app.middleware("http")
async def handle_errors(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as error:
        logger.exception("Request to %s failed", request.url.path)
        status, message = friendly_error(error)
        return JSONResponse(status_code=status, content={"error": message})


def too_few_words(text, minimum):
    """Return a 400 error response if text is too short, or None if it's fine."""
    word_count = len(text.split())
    if word_count < minimum:
        word = "word" if minimum == 1 else "words"
        return JSONResponse(
            status_code=400,
            content={"error": f"Please enter at least {minimum} {word} (you entered {word_count})."},
        )
    return None


# This describes the JSON body we expect, e.g. {"question": "What is Python?"}
class Question(BaseModel):
    question: str


@app.post("/ask")
def ask(body: Question):
    # Check the input first, so we don't pay for a call with no question.
    error = too_few_words(body.question, 1)
    if error:
        return error

    # NDJSON lets the answer stream while still sending usage metadata at the end.
    def generate():
        answer_parts = []
        try:
            with client.messages.stream(
                model=MODEL,
                max_tokens=16000,  # High limit so long answers don't get cut off
                messages=[{"role": "user", "content": body.question}],
            ) as stream:
                for text in stream.text_stream:
                    answer_parts.append(text)
                    yield json.dumps({"type": "chunk", "text": text}, ensure_ascii=False) + "\n"

                message = stream.get_final_message()
                input_tokens = message.usage.input_tokens
                output_tokens = message.usage.output_tokens
                cost_usd = (
                    input_tokens * INPUT_COST_PER_MILLION_USD
                    + output_tokens * OUTPUT_COST_PER_MILLION_USD
                ) / 1_000_000
                yield json.dumps(
                    {
                        "type": "done",
                        "answer": "".join(answer_parts),
                        "model": message.model,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "tokens_used": input_tokens + output_tokens,
                        "cost_usd": round(cost_usd, 8),
                    }
                ) + "\n"
        except Exception as error:
            logger.exception("Streaming /ask failed")
            _, message = friendly_error(error)
            yield json.dumps({"type": "error", "message": message}, ensure_ascii=False) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


# --- Structured output endpoints ---
# For these, we give Claude a Pydantic model and it replies with JSON that
# matches it exactly, so we get clean fields instead of free-form text.


# The JSON body both endpoints expect, e.g. {"text": "Some long article..."}
class TextInput(BaseModel):
    text: str


# The shape of the summary Claude must return.
class Summary(BaseModel):
    summary: str  # A short paragraph summing up the text
    key_points: List[str]  # The main points as a list


# The shape of the sentiment result Claude must return.
class Sentiment(BaseModel):
    sentiment: Literal["positive", "negative", "neutral", "mixed"]
    confidence: float  # How sure Claude is, from 0.0 to 1.0
    explanation: str  # One sentence explaining why


def ask_for_structured(prompt, output_model, system=""):
    """Ask Claude and get back an instance of output_model (e.g. Summary)."""
    # messages.parse checks Claude's reply against our model for us.
    response = client.messages.parse(
        model=MODEL,
        max_tokens=16000,
        system=system,
        messages=[{"role": "user", "content": prompt}],
        output_format=output_model,
    )
    # parsed_output is empty if Claude refused or ran out of room.
    if response.parsed_output is None:
        raise UnusableAnswerError(
            f"Claude couldn't produce a result for this text (reason: {response.stop_reason})."
        )
    return response.parsed_output


@app.post("/summarise", response_model=Summary)
def summarise(body: TextInput):
    # Check the input first, so we don't pay for a call with nothing to summarise.
    error = too_few_words(body.text, MIN_WORDS)
    if error:
        return error
    system = (
        "You summarise text. Write a summary of 2-3 sentences and 3-5 key points. "
        "Use only information from the text; don't add facts or opinions."
    )
    return ask_for_structured(f"Summarise this text:\n\n{body.text}", Summary, system)


@app.post("/analyze-sentiment", response_model=Sentiment)
def analyze_sentiment(body: TextInput):
    error = too_few_words(body.text, 1)
    if error:
        return error
    prompt = f"Analyze the sentiment of this text. Give a confidence between 0.0 and 1.0.\n\n{body.text}"
    return ask_for_structured(prompt, Sentiment)
