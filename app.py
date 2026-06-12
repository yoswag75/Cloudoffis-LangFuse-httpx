import time
import uuid
import streamlit as st
from dotenv import load_dotenv
from langfuse import get_client

from langfuse_utils import (
    get_prompt_and_config,
    save_turn_to_dataset,
    log_generation,
    ensure_dataset_exists,
)
from llm_httpx import (
    call_chat,
    call_complete,
    call_summarize,
    call_instruct,
)

load_dotenv()
lf = get_client()

st.set_page_config(
    page_title="Gemma 4 E2B — Custom API",
    page_icon="🌐",
    layout="wide",
)
st.title("🌐 Gemma 4 E2B — Custom API")

ensure_dataset_exists()

try:
    data = get_prompt_and_config("simple-qa")
except Exception as e:
    st.error(f"❌ Could not fetch prompt from Langfuse: {e}")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.header("⚙️ Model Parameters")
st.sidebar.caption("Defaults from Langfuse · adjust freely")

temperature = st.sidebar.slider("Temperature", 0.0, 2.0,  float(data["temperature"]), 0.05)
top_p       = st.sidebar.slider("Top P",       0.0, 1.0,  float(data["top_p"]),       0.05)
top_k       = st.sidebar.slider("Top K",       1,   100,  int(data["top_k"]),          1)
max_tokens  = st.sidebar.slider("Max Tokens",  64,  4096, int(data["max_tokens"]),     64)

config = {
    "temperature": temperature,
    "top_p":       top_p,
    "top_k":       top_k,
    "max_tokens":  max_tokens,
}

st.sidebar.divider()
system_prompt = st.sidebar.text_area(
    "System Prompt",
    value="You are a helpful AI assistant.",
    height=100,
)
st.sidebar.divider()
st.sidebar.subheader("📋 Active Config")
st.sidebar.json(config)
st.sidebar.divider()
st.sidebar.subheader("🔗 Available Endpoints")
st.sidebar.code(
    "POST /api/v1/chat\n"
    "POST /api/v1/complete\n"
    "POST /api/v1/summarize\n"
    "POST /api/v1/instruct",
    language="text",
)
st.sidebar.caption("Swagger UI → http://localhost:8000/docs")

# ── Session state ─────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

session_id = st.session_state.session_id
st.sidebar.caption(f"Session: `{session_id[:8]}...`")

# ── 4 Tabs — one per endpoint ─────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "💬 /api/v1/chat",
    "✍️ /api/v1/complete",
    "📝 /api/v1/summarize",
    "🎯 /api/v1/instruct",
])


def show_result(response, req_payload, raw_response, endpoint):
    """Shared result display used by all tabs."""
    st.success("✅ Success")
    st.markdown("**Response**")
    st.info(response)

    meta = {
        "endpoint":              endpoint,
        "model":                 "gemma-4-e2b",
        "backend":               "ctransformers",
        "temperature":           config["temperature"],
        "top_p":                 config["top_p"],
        "top_k":                 config["top_k"],
        "max_tokens":            config["max_tokens"],
        "response_length_words": len(response.split()),
        "prompt_tokens":         raw_response.get("prompt_tokens"),
        "completion_tokens":     raw_response.get("completion_tokens"),
        "total_tokens":          raw_response.get("total_tokens"),
        "generation_time_sec":   raw_response.get("generation_time_sec"),
        "session_id":            session_id,
        "trace_logged":          True,
        "dataset_saved":         True,
    }

    return meta


# ── Tab 1: /api/v1/chat ───────────────────────────────────────
with tab1:
    st.subheader("💬 Multi-turn Chat")
    st.caption("Full conversation history · system prompt · roles")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("meta"):
                with st.expander("📊 Turn stats"):
                    st.json(msg["meta"])

    user_input = st.chat_input("Chat with Gemma...", key="chat_input")

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    t0 = time.time()
                    response, req_payload, raw_response = call_chat(
                        history=st.session_state.chat_history,
                        config=config,
                        system_prompt=system_prompt,
                        user_message=user_input,
                    )
                    latency = round(time.time() - t0, 2)

                    st.markdown(response)
                    meta = show_result(response, req_payload, raw_response, "/api/v1/chat")
                    meta["response_time_sec"] = latency

                    st.session_state.chat_history.append({"role": "user",      "content": user_input})
                    st.session_state.chat_history.append({"role": "assistant", "content": response, "meta": meta})

                    messages_for_trace = (
                        [{"role": "system", "content": system_prompt}]
                        + [{"role": m["role"], "content": m["content"]}
                           for m in st.session_state.chat_history]
                    )

                    log_generation(session_id, messages_for_trace, response, config, latency, req_payload, raw_response, "/api/v1/chat")
                    save_turn_to_dataset(user_input, response, config, session_id, "/api/v1/chat")

                except Exception as e:
                    st.error(f"❌ {e}")

    if st.session_state.chat_history:
        if st.button("🗑️ Clear chat", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()


# ── Tab 2: /api/v1/complete ───────────────────────────────────
with tab2:
    st.subheader("✍️ Raw Completion")
    st.caption("Single prompt string · no roles · no history")

    prompt_input = st.text_area("Enter a prompt", height=150, key="complete_prompt",
                                 placeholder="Once upon a time in a land far away...")

    if st.button("▶ Complete", key="btn_complete"):
        if prompt_input.strip():
            with st.spinner("Generating..."):
                try:
                    t0 = time.time()
                    response, req_payload, raw_response = call_complete(
                        prompt=prompt_input,
                        config=config,
                    )
                    latency = round(time.time() - t0, 2)
                    meta = show_result(response, req_payload, raw_response, "/api/v1/complete")
                    meta["response_time_sec"] = latency

                    log_generation(session_id, [{"role": "user", "content": prompt_input}], response, config, latency, req_payload, raw_response, "/api/v1/complete")
                    save_turn_to_dataset(prompt_input, response, config, session_id, "/api/v1/complete")

                except Exception as e:
                    st.error(f"❌ {e}")
        else:
            st.warning("Enter a prompt first.")


# ── Tab 3: /api/v1/summarize ──────────────────────────────────
with tab3:
    st.subheader("📝 Summarize")
    st.caption("Paste any text · choose style · get summary")

    text_input = st.text_area("Text to summarize", height=200, key="summarize_text",
                               placeholder="Paste a long article, document, or paragraph here...")
    style = st.radio("Summary style", ["concise", "detailed", "bullet"], horizontal=True)

    if st.button("▶ Summarize", key="btn_summarize"):
        if text_input.strip():
            with st.spinner("Summarizing..."):
                try:
                    t0 = time.time()
                    response, req_payload, raw_response = call_summarize(
                        text=text_input,
                        style=style,
                        config=config,
                    )
                    latency = round(time.time() - t0, 2)

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Original words", raw_response.get("original_length_words"))
                    col2.metric("Summary words",  raw_response.get("summary_length_words"))
                    col3.metric("Compression",    f"{raw_response.get('compression_ratio', 0)*100:.0f}%")

                    meta = show_result(response, req_payload, raw_response, "/api/v1/summarize")
                    meta["response_time_sec"] = latency
                    meta["style"] = style

                    log_generation(session_id, [{"role": "user", "content": text_input}], response, config, latency, req_payload, raw_response, "/api/v1/summarize")
                    save_turn_to_dataset(text_input, response, config, session_id, "/api/v1/summarize")

                except Exception as e:
                    st.error(f"❌ {e}")
        else:
            st.warning("Enter text to summarize first.")


# ── Tab 4: /api/v1/instruct ───────────────────────────────────
with tab4:
    st.subheader("🎯 Instruct")
    st.caption("Instruction + optional context · good for extraction, classification, transformation")

    instruction_input = st.text_input(
        "Instruction",
        key="instruct_instruction",
        placeholder="Extract all dates mentioned in the text below",
    )
    context_input = st.text_area(
        "Context (optional)",
        height=150,
        key="instruct_context",
        placeholder="The project started on March 3rd and the deadline is June 15th...",
    )

    if st.button("▶ Run", key="btn_instruct"):
        if instruction_input.strip():
            with st.spinner("Running..."):
                try:
                    t0 = time.time()
                    response, req_payload, raw_response = call_instruct(
                        instruction=instruction_input,
                        context=context_input,
                        config=config,
                    )
                    latency = round(time.time() - t0, 2)
                    meta = show_result(response, req_payload, raw_response, "/api/v1/instruct")
                    meta["response_time_sec"] = latency

                    user_msg = f"Instruction: {instruction_input}\nContext: {context_input}"
                    log_generation(session_id, [{"role": "user", "content": user_msg}], response, config, latency, req_payload, raw_response, "/api/v1/instruct")
                    save_turn_to_dataset(user_msg, response, config, session_id, "/api/v1/instruct")

                except Exception as e:
                    st.error(f"❌ {e}")
        else:
            st.warning("Enter an instruction first.")