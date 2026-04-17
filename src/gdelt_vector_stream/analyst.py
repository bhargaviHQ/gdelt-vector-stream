"""RAG-powered news analyst: answer questions grounded in GDELT events via Hugging Face Inference API."""

import argparse
import logging
import os
import sys
from typing import Any

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from gdelt_vector_stream.query import semantic_search

load_dotenv()

logger = logging.getLogger(__name__)

HF_MODEL = os.getenv("HF_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")


def format_events_as_context(results: list[dict[str, Any]]) -> str:
    """
    Format Pinecone search results into readable context for the LLM.

    Returns a numbered list of event summaries the LLM can reference.
    """
    lines = []
    for i, result in enumerate(results, 1):
        meta = result["metadata"]
        score = result["similarity_score"]

        actor1 = meta.get("actor1_name", "Unknown")
        actor2 = meta.get("actor2_name", "Unknown")
        country = meta.get("country_code", "Unknown")
        event_code = meta.get("event_code", "")
        tone = meta.get("avg_tone", 0)
        date = meta.get("event_date", "Unknown date")
        mentions = meta.get("num_mentions", 0)
        url = meta.get("source_url", "")

        lines.append(
            f"[Event {i}] Date: {date} | Location: {country} | "
            f"Actors: {actor1} → {actor2} | "
            f"Event code: {event_code} | Tone: {tone:.1f} | "
            f"Mentions: {mentions} | Relevance: {score:.2f}\n"
            f"  Source: {url}"
        )

    return "\n\n".join(lines)


def build_messages(question: str, context: str) -> list[dict[str, str]]:
    """
    Build the RAG message list with event context and user question.

    Using a separate system role keeps the analyst instructions isolated from the
    user-supplied question, reducing the risk of prompt injection.
    """
    system_content = (
        "You are a geopolitical news analyst. Answer the user's question using ONLY the "
        "GDELT news events provided. Be concise (3-5 sentences). If the events don't "
        "contain enough information, say so honestly. When referencing events, mention "
        "the actors, locations, and dates. End with a 'Sources' section listing the "
        "relevant source URLs.\n\n"
        f"--- GDELT NEWS EVENTS ---\n{context}\n--- END EVENTS ---"
    )
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": question},
    ]


def call_hf_inference(messages: list[dict[str, str]], model: str = HF_MODEL) -> str:
    """
    Call the Hugging Face Inference API.

    Args:
        messages: List of chat messages (system + user roles)
        model: HF model ID (e.g., "mistralai/Mistral-7B-Instruct-v0.3")

    Returns:
        Generated response text

    Raises:
        RuntimeError: If the API call fails
    """
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN not set. Get a free token at https://huggingface.co/settings/tokens "
            "and add it to your .env file."
        )

    client = InferenceClient(token=hf_token)

    try:
        response = client.chat_completion(
            model=model,
            messages=messages,
            max_tokens=512,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"HF Inference API error: {e}")


def ask(question: str, top_k: int = 5, model: str | None = None) -> dict[str, Any]:
    """
    Ask a question and get an answer grounded in GDELT events.

    Args:
        question: Natural language question (e.g., "What's happening in Ukraine?")
        top_k: Number of events to retrieve from Pinecone
        model: HF model override (default from env/config)

    Returns:
        Dict with "answer", "events", and "model" keys
    """
    model = model or HF_MODEL

    # Step 1: Retrieve relevant events from Pinecone
    logger.info(f"Searching GDELT index for: {question}")
    results = semantic_search(question, top_k=top_k)

    if not results:
        return {
            "answer": "No relevant events found in the GDELT index. Try ingesting more data.",
            "events": [],
            "model": model,
        }

    # Step 2: Format events as context
    context = format_events_as_context(results)
    logger.info(f"Retrieved {len(results)} events, sending to {model}...")

    # Step 3: Build messages and call LLM
    messages = build_messages(question, context)
    answer = call_hf_inference(messages, model=model)

    return {
        "answer": answer,
        "events": results,
        "model": model,
    }


# --- CLI ---

if __name__ == "__main__":
    log_level = os.getenv("LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Ask questions about global events, answered by HF Inference API grounded in GDELT data"
    )
    parser.add_argument("question", nargs="+", help="Your question in natural language")
    parser.add_argument("--top-k", type=int, default=5, help="Number of events to retrieve (default: 5)")
    parser.add_argument("--model", type=str, default=None, help=f"HF model (default: {HF_MODEL})")
    parser.add_argument("--show-context", action="store_true", help="Show the raw events sent to the LLM")

    args = parser.parse_args()
    question = " ".join(args.question)

    print(f"\nQuestion: {question}")
    print("=" * 60)

    try:
        result = ask(question, top_k=args.top_k, model=args.model)
    except RuntimeError as e:
        print(f"\nError: {e}")
        sys.exit(1)

    # Show context if requested
    if args.show_context:
        print(f"\n--- Retrieved Events ({len(result['events'])}) ---")
        context = format_events_as_context(result["events"])
        print(context)
        print("--- End Events ---\n")

    # Show answer
    print(f"\nAnswer (via {result['model']}):\n")
    print(result["answer"])

    # Show sources
    urls = [
        e["metadata"].get("source_url", "")
        for e in result["events"]
        if e["metadata"].get("source_url")
    ]
    if urls:
        print("\n\nSources:")
        for url in urls[:5]:
            print(f"  - {url}")

    print()
