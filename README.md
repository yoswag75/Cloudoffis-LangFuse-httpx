# Gemma 3 Custom API & Streamlit UI with Langfuse Observability

![Python](https://img.shields.io/badge/Python-3.13-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue.svg)

A complete, end-to-end containerized application that serves the **Gemma 3 (270M Instruct)** model via a custom **FastAPI** backend, provides an interactive **Streamlit** user interface, and deeply integrates with **Langfuse** for LLM observability, tracing, and dataset management.

---

## 🌟 Features

- **FastAPI Backend:** Serves the Hugging Face `google/gemma-3-270m-it` model on the CPU. It exposes four specialized endpoints to handle multi-turn chat, raw text completion, summarization, and instruction-based text generation.
- **Streamlit Frontend:** A rich, interactive web UI with four dedicated tabs for each backend capability. It includes a dynamic sidebar to adjust model generation parameters (Temperature, Top-P, Top-K, Max Tokens) in real-time.
- **Langfuse Observability:** All LLM requests are traced using Langfuse. Automatically tracks latency, token usage (prompt, completion, total), generation times, and logs inputs/outputs. It automatically saves high-quality conversational turns into a designated dataset (`gemma-httpx-ctransformers-conversations`).
- **Dockerized Architecture:** Completely encapsulated environment using `docker-compose`. Independent containers for the API and the UI, ensuring consistent deployments and dependency management via `uv`.

---

## 📂 Project Structure

```text
.
├── api/
│   └── main.py              # FastAPI application and model loading logic
├── app.py                   # Streamlit frontend application
├── llm_httpx.py             # HTTP client module connecting the UI to the API
├── langfuse_utils.py        # Langfuse tracing, logging, and dataset operations
├── Dockerfile.api           # Docker configuration for the FastAPI service
├── Dockerfile.streamlit     # Docker configuration for the Streamlit service
├── docker-compose.yml       # Docker Compose setup to orchestrate API and UI
├── pyproject.toml           # Project dependencies and Python version requirements
└── .env                     # Environment variables (Langfuse keys, HF token)
```

---

## 🚀 Getting Started

### Prerequisites

- **Docker** and **Docker Compose** installed on your machine.
- A **Hugging Face account** and an [Access Token](https://huggingface.co/settings/tokens) with read permissions (to download the Gemma model).
- A **Langfuse account** with API keys (Public Key, Secret Key, Host).

### 1. Environment Configuration

Create a `.env` file in the root directory and populate it with your credentials:

```env
# Hugging Face Token (Required to pull Gemma-3)
HF_TOKEN=hf_your_hugging_face_token_here

# Langfuse Credentials (Required for Observability)
LANGFUSE_PUBLIC_KEY=pk-lf-your_public_key
LANGFUSE_SECRET_KEY=sk-lf-your_secret_key
LANGFUSE_HOST=https://cloud.langfuse.com

# Optional custom API URL for the UI to talk to (defaults to http://api:8000 inside Docker)
CUSTOM_API_URL=http://api:8000
```

### 2. Build and Run

Deploy both the backend and frontend simultaneously using Docker Compose. The initial build may take a few minutes as it downloads dependencies and the Gemma model.

```bash
docker-compose up --build
```

### 3. Access the Application

Once the containers are running and healthy:
- **Streamlit UI:** `http://localhost:8502`
- **FastAPI Swagger Docs:** `http://localhost:8000/docs`

---

## 📡 API Endpoints

The FastAPI backend exposes the following endpoints (all under `/api/v1/`):

1. **`POST /chat`**: Handles multi-turn conversations. Expects an array of messages with system, user, and assistant roles. Uses the model's native chat template.
2. **`POST /complete`**: Performs raw text completion. Takes a single prompt string without any system roles or conversational formatting.
3. **`POST /summarize`**: A dedicated endpoint for summarizing long texts. Supports `concise`, `detailed`, and `bullet` styles.
4. **`POST /instruct`**: Built for complex instruction-following. Accepts an instruction and optional context text (great for extraction and classification).

All endpoints accept standard model generation parameters (`temperature`, `top_p`, `top_k`, `max_new_tokens`) inside the request body.

---

## 🖥️ User Interface Guide

The Streamlit UI is divided into a sidebar and four main tabs:

- **Sidebar Config:** Tweak generation settings dynamically. Change temperature, top-k, top-p, or max tokens. You can also edit the initial System Prompt.
- **💬 Chat Tab:** A ChatGPT-like multi-turn chat interface. Each turn displays rich metadata and trace links in a toggleable expander.
- **✍️ Raw Completion Tab:** Simply type a prompt and watch the model complete it.
- **📝 Summarize Tab:** Paste large blocks of text, choose a summary style, and get token compression ratios in a visual dashboard.
- **🎯 Instruct Tab:** Provide a specific instruction alongside contextual data to execute targeted NLP tasks.

---

## 📊 Langfuse Observability Integration

This project deeply embeds Langfuse tracing into the request lifecycle via `langfuse_utils.py`:

- **Prompts & Configs:** Pulls default configuration values directly from Langfuse prompt management.
- **Trace Spans:** Tracks the exact time spent in the HTTP request (`http-request`) vs model generation (`llm-call`).
- **Metrics Tracking:** Calculates tokens generated per second and captures exact token counts reported by the FastAPI backend.
- **Dataset Generation:** Automatically saves high-quality turns into a Langfuse dataset, bridging the gap between local testing and ML pipeline curation.

---

## 🛠️ Tech Stack Notes

- **Model Execution:** Runs natively on CPU via HuggingFace `transformers` and PyTorch.
- **Package Management:** Uses `uv` for blazingly fast dependency installation within the Dockerfiles.
- **HTTP Client:** Uses `httpx` to manage network calls between the Streamlit container and the FastAPI container asynchronously.
