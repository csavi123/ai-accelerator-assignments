# frontend.py - a simple Streamlit page that talks to our FastAPI backend.

import json
import os

import requests
import streamlit as st

# Where the backend is running. Docker sets BACKEND_URL; locally we use localhost.
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

# (connect timeout, read timeout) in seconds. The read timeout is long because
# the backend may retry Claude a few times before answering.
TIMEOUT = (5, 300)

st.title("Ask Claude")

# One tab per backend endpoint.
ask_tab, summarise_tab, sentiment_tab = st.tabs(["Ask", "Summarise", "Sentiment"])


class BackendError(Exception):
    pass


def check(response):
    # If the backend sent an error, raise it with the backend's friendly message.
    if not response.ok:
        try:
            message = response.json()["error"]
        except (ValueError, KeyError):
            message = f"Unexpected error (status {response.status_code})."
        raise BackendError(message)


def stream_answer(question, result):
    completed = False
    with requests.post(f"{BACKEND_URL}/ask", json={"question": question}, stream=True, timeout=TIMEOUT) as response:
        check(response)
        response.encoding = "utf-8"
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise BackendError("The backend returned an invalid streaming response.") from error

            if event.get("type") == "chunk":
                yield event["text"]
            elif event.get("type") == "done":
                result.update(event)
                completed = True
            elif event.get("type") == "error":
                raise BackendError(event["message"])
            else:
                raise BackendError("The backend returned an unexpected streaming response.")

    if not completed:
        raise BackendError("The backend ended the response before returning usage details.")


def post_text(endpoint, text):
    # Send {"text": ...} to an endpoint and return the JSON it sends back.
    response = requests.post(f"{BACKEND_URL}{endpoint}", json={"text": text}, timeout=TIMEOUT)
    check(response)
    return response.json()


# --- Ask tab: streams a free-form answer ---
with ask_tab:
    question = st.text_input("Your question:")
    if st.button("Ask"):
        if not question.strip():
            st.warning("Please type a question first.")
        else:
            try:
                result = {}
                st.write_stream(stream_answer(question, result))
                st.caption(f"Model: {result['model']}")
                usage_columns = st.columns(2)
                usage_columns[0].metric("Tokens used", result["tokens_used"])
                usage_columns[1].metric("Estimated cost", f"${result['cost_usd']:.6f}")
                st.caption(
                    f"Input: {result['input_tokens']} tokens · "
                    f"Output: {result['output_tokens']} tokens"
                )
            except BackendError as error:
                st.error(str(error))
            except requests.Timeout:
                st.error("The request timed out. Please try again.")
            except requests.RequestException:
                st.error("Could not reach the backend. Is it running?")

# --- Summarise tab: shows a summary and a list of key points ---
with summarise_tab:
    long_text = st.text_area("Text to summarise:", height=200)
    if st.button("Summarise"):
        if not long_text.strip():
            st.warning("Please paste some text first.")
        else:
            try:
                with st.spinner("Summarising..."):
                    result = post_text("/summarise", long_text)
                st.subheader("Summary")
                st.write(result["summary"])
                st.subheader("Key points")
                for point in result["key_points"]:
                    st.markdown(f"- {point}")
            except BackendError as error:
                st.error(str(error))
            except requests.Timeout:
                st.error("The request timed out. Please try again.")
            except requests.RequestException:
                st.error("Could not reach the backend. Is it running?")

# --- Sentiment tab: shows the label, confidence and explanation ---
with sentiment_tab:
    sentiment_text = st.text_area("Text to analyze:", height=150)
    if st.button("Analyze sentiment"):
        if not sentiment_text.strip():
            st.warning("Please type some text first.")
        else:
            try:
                with st.spinner("Analyzing..."):
                    result = post_text("/analyze-sentiment", sentiment_text)
                st.metric("Sentiment", result["sentiment"].capitalize())
                # Keep the value between 0 and 1, since the progress bar needs that.
                confidence = min(max(result["confidence"], 0.0), 1.0)
                st.progress(confidence, text=f"Confidence: {confidence:.0%}")
                st.write(result["explanation"])
            except BackendError as error:
                st.error(str(error))
            except requests.Timeout:
                st.error("The request timed out. Please try again.")
            except requests.RequestException:
                st.error("Could not reach the backend. Is it running?")
