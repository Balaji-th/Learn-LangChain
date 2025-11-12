# LangChain Walkthrough

![LangChain Logo](https://github.com/langchain-ai/langchain/raw/master/.github/images/logo-dark.svg)

## Installation

This project uses `uv` for package management.

**1. Install `uv`:**

*   **macOS and Linux:**
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
*   **Windows:**
    ```powershell
    irm https://astral.sh/uv/install.ps1 | iex
    ```

**2. Install Dependencies:**

Once `uv` is installed, you can install the required packages from `requirements.txt`:

```bash
uv pip install -r requirements.txt
```

## Core LangChain Concepts

This repository provides a hands-on walkthrough of several core components in the LangChain library.

### 1. Chat Models

*   **File:** `01_ChatModels.py`
*   **Concept:** This script shows how to initialize and use a chat model from a provider like OpenAI, Google, or Anthropic. The `invoke` method sends a prompt to the model and retrieves the response.

### 2. Prompts

*   **File:** `05_Prompts.py`
*   **Concept:** Prompts are how you guide the output of a language model. This script demonstrates two ways to create prompt templates:
    *   `ChatPromptTemplate.from_template`: For simple prompts with placeholders.
    *   `ChatPromptTemplate.from_messages`: For more complex, chat-style prompts with distinct roles (e.g., "system" and "human").

### 3. Chains

*   **File:** `07_Runnables_Chain.py`
*   **Concept:** Chains allow you to combine multiple components together to create a single, coherent application. This script demonstrates how to create a simple chain that takes a prompt, sends it to a model, and parses the output. LangChain Expression Language (LCEL) is used to pipe components together.

### 4. Retrieval-Augmented Generation (RAG)

*   **Files:** `11_RAGs_Indexing.py`, `12_RAGs_Retriever.py`
*   **Concept:** RAGs allow you to connect LLMs to external data sources. This example shows how to:
    1.  **Index Data:** Load documents, split them into chunks, and store them in a vector store (ChromaDB) using embeddings.
    2.  **Retrieve Data:** Use a retriever to fetch relevant documents from the vector store based on a user's query.
    3.  **Generate Response:** The retrieved documents are then used as context for the LLM to generate a response.

## References and Further Learning

*   **LangChain Official Documentation:** [https://python.langchain.com/](https://python.langchain.com/)
*   **LangChain GitHub Repository:** [https://github.com/langchain-ai/langchain](https://github.com/langchain-ai/langchain)
*   **LangChain Expression Language (LCEL):** [https://python.langchain.com/docs/expression_language/](https://python.langchain.com/docs/expression_language/)