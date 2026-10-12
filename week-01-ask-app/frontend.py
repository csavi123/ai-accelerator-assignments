# frontend.py - a simple Streamlit page that talks to our FastAPI backend.

import json
import os
import re
from pathlib import Path

import requests
import streamlit as st

# Where the backend is running. Docker sets BACKEND_URL; locally we use localhost.
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

# (connect timeout, read timeout) in seconds. The read timeout is long because
# the backend may retry Claude a few times before answering.
TIMEOUT = (5, 300)

# Same rule as the backend: "#" is reserved for chunk numbers like "doc2_expenses#3".
DOCUMENT_ID_PATTERN = r"^[A-Za-z0-9_.-]+$"

st.title("Ask Claude")
st.caption(f"Backend: {BACKEND_URL}")

# One tab per backend endpoint.
ask_tab, ingest_tab, summarise_tab, sentiment_tab = st.tabs(["Ask", "Ingest", "Summarise", "Sentiment"])


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


def post_json(endpoint, body):
    # Send a JSON body to an endpoint and return the JSON it sends back.
    response = requests.post(f"{BACKEND_URL}{endpoint}", json=body, timeout=TIMEOUT)
    check(response)
    return response.json()


def post_text(endpoint, text):
    # Send {"text": ...} to an endpoint and return the JSON it sends back.
    return post_json(endpoint, {"text": text})


def show_sources(result):
    # List the chunks the backend retrieved, marking the ones the answer cites.
    if not result["sources"]:
        return
    cited = set(result["cited_chunk_ids"])
    st.subheader("Sources")
    for source in result["sources"]:
        mark = "✅ cited" if source["chunk_id"] in cited else "not cited"
        st.markdown(
            f"- `{source['chunk_id']}` from **{source['document_id']}** · "
            f"score {source['score']:.2f} · {mark}"
        )


# --- Ask tab: streams an answer grounded in the ingested documents ---
with ask_tab:
    question = st.text_input("Ask a question about the Northwind documents:")
    if st.button("Ask"):
        if not question.strip():
            st.warning("Please type a question first.")
        else:
            try:
                result = {}
                st.write_stream(stream_answer(question, result))
                if result["refused"] and result["model"] is None:
                    st.info("No relevant documents were found, so Claude was not called.")
                elif result["refused"]:
                    st.info("Claude read the closest documents but they don't contain the answer.")
                show_sources(result)
                if result["model"]:
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

# --- Ingest tab: adds a document to the knowledge base ---
with ingest_tab:
    st.write("Upload a text file or paste text. It is split into chunks and stored for the Ask tab.")
    uploaded_file = st.file_uploader("Text file (optional):", type=["txt", "md"])
    # Default the document ID to the file name, e.g. "doc2_expenses.txt" -> "doc2_expenses".
    default_id = Path(uploaded_file.name).stem if uploaded_file else ""
    document_id = st.text_input("Document ID:", value=default_id, help="Letters, numbers, dots, dashes and underscores.")
    pasted_text = st.text_area("Or paste the text here:", height=200, disabled=uploaded_file is not None)
    if st.button("Ingest"):
        text = uploaded_file.getvalue().decode("utf-8") if uploaded_file else pasted_text
        if not re.match(DOCUMENT_ID_PATTERN, document_id):
            st.warning("Please enter a document ID using only letters, numbers, dots, dashes and underscores.")
        elif not text.strip():
            st.warning("Please upload a file or paste some text first.")
        else:
            try:
                with st.spinner("Chunking and storing..."):
                    result = post_json("/ingest", {"document_id": document_id, "text": text})
                st.success(f"Ingested **{result['document_id']}** as {result['chunks']} chunks.")
                st.caption("Chunk IDs: " + ", ".join(result["chunk_ids"]))
            except BackendError as error:
                st.error(str(error))
            except UnicodeDecodeError:
                st.error("That file isn't UTF-8 text. Please upload a plain .txt or .md file.")
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
