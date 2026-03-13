# goodmem-semantic-kernel

A [GoodMem](https://goodmem.ai) connector for [Microsoft Semantic Kernel](https://github.com/microsoft/semantic-kernel).

Implements Semantic Kernel's `VectorStoreCollection` and `VectorStore` interfaces so agents built on Semantic Kernel can store and retrieve memories from a GoodMem server without having to configure your own data processing pipeline

## What is GoodMem?

GoodMem is a centralized memory API for AI agents and LLMs. The point of GoodMem is so that you can easily and efficiently store and retrieve your data/memories through semantic searching, ai summaries, and context-aware results.

GoodMem stores text memories as semantic embeddings in PostgreSQL (via `pgvector`) and retrieves them by semantic similarity. Because it runs as a shared service, multiple agents can read and write to the same memory spaces simultaneously.

> Embeddings are computed **server-side**, so this connector never needs an `embedding_generator`.

### Conceptual Overview

In GoodMem all data is hosted in a "**Space**", an abstract storage unit in GoodMem.
Each **Space** can be configured with embedders and/or chunking strategies. Each Space holds "**Memories**".

**Memories** are stored content with associated metadata that are automatically chunked and embedded for efficient retrieval. All **Memories** belong to a **Space**.

**Embedders** convert your data into a vectorized format. GoodMem supports multiple embedding models & providers.

---

## Quickstart

1. [installation](#installation)
2. [configuration](#configuration)
3. [run sample files](#running-the-samples)
4. create your own integration

## Installation

### Python (recommended)

**Requirements:** Python 3.10+ and a running GoodMem server.

```bash
pip install goodmem-semantic-kernel
```

To install from source:

```bash
git clone https://github.com/PAIR-Systems-Inc/goodmem-semantic-kernel
cd goodmem-semantic-kernel
pip install -e .
```

### .NET (debian/ubuntu)

```bash
sudo apt install dotnet-sdk-8.0
```

Build the connector from source:

```bash
dotnet build dotnet/GoodMem.SemanticKernel/GoodMem.SemanticKernel.csproj
```

### Java (debian/ubuntu)

**Requirements:** JDK 17+ (JDK 21 recommended) and Maven 3.6+.

Install JDK 21 via SDKMAN (recommended):

```bash
sdk install java 21.0.5-tem
```

Or via apt:

```bash
sudo apt install openjdk-21-jdk
```

Build and install the connector into your local Maven repository:

```bash
mvn install -f java/pom.xml -DskipTests
```

## Configuration

All settings are read from environment variables with the `GOODMEM_` prefix, or passed directly via `GoodMemSettings`.

```bash
export GOODMEM_API_KEY=your_key_here
export GOODMEM_BASE_URL=https://your_goodmem_server:8080
export GOODMEM_VERIFY_SSL=true_or_false
```

| Variable | Required | Default | Description |
|---|---|---|---|
| `GOODMEM_API_KEY` | Yes | — | API key for the GoodMem server |
| `GOODMEM_BASE_URL` | No | `http://localhost:8080` | GoodMem server base URL |
| `GOODMEM_EMBEDDER_ID` | No | auto-detected | UUID of the embedder to use |
| `GOODMEM_VERIFY_SSL` | No | `false` | Set to `false` for self-signed certs, set to `true` if you setup custom TLS certs|

## Running the samples

### Python

```bash
cd samples/python

# Option A — agent with memory tool (also requires OPENAI_API_KEY)
OPENAI_API_KEY=your_openai_key_here
python example_agent.py

# Option B — single collection
python example_single_collection.py

# Option C — store with multiple collections
python example_store.py
```

If a sample fails, double-check [Configuration](#configuration) or run inside a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

### .NET

```bash
cd samples/dotnet/ExampleAgent
dotnet run
```

Each sample lists its required environment variables at the top of `Program.cs`.

### Java

Build the connector once before running any sample:

```bash
mvn install -f java/pom.xml -DskipTests
```

Then run any sample:

```bash
cd samples/java/ExampleAgent
mvn compile exec:java
```

Each sample lists its required environment variables in the file header.

## Testing

```bash
# Unit tests (no server required)
pytest python/tests/unit/

# Integration tests (requires a live GoodMem server)
GOODMEM_API_KEY=your_key_here pytest -m integration
```

### Define a data model

```python
from dataclasses import dataclass
from typing import Annotated
from semantic_kernel.data.vector import VectorStoreField, vectorstoremodel

@vectorstoremodel
@dataclass
class Note:
    id: Annotated[str | None, VectorStoreField("key")] = None
    content: Annotated[str, VectorStoreField("data", type="str")] = ""
    source: Annotated[str | None, VectorStoreField("data")] = None
```

- Exactly one `"key"` field (the memory ID — `None` lets the server generate a UUID).
- One `"data"` field named `content` becomes the embedded text (`originalContent` in GoodMem).
- All other `"data"` fields are stored as metadata and returned on search results.
- `"vector"` fields are accepted for interface compatibility but ignored — GoodMem embeds server-side.

We have three example patterns provided in the samples directory. We recommend option A, but choose what works for you.

Option A (`samples/python/example_agent.py`) is the recommended pattern for production agents since the LLM decides when to call memory and what to search for, rather than the application hardcoding those decisions.

### Option A: Wired into a Semantic Kernel agent

```python
from semantic_kernel.agents import AgentThread, ChatCompletionAgent
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.functions import KernelParameterMetadata, KernelPlugin
from goodmem_semantic_kernel import GoodMemCollection

async def main():
    async with GoodMemCollection(record_type=Note, collection_name="agent-memory") as coll:
        await coll.ensure_collection_exists()
        await coll.upsert([...])  # seed your memories

        memory_plugin = KernelPlugin(
            name="memory",
            functions=[
                coll.create_search_function(
                    function_name="recall",
                    description="Search long-term memory for relevant facts.",
                    string_mapper=lambda r: r.record.content,
                )
            ],
        )

        agent = ChatCompletionAgent(
            name="MemoryAgent",
            service=OpenAIChatCompletion(),
            instructions="Always search memory before answering factual questions.",
            function_choice_behavior=FunctionChoiceBehavior.Auto(),
            plugins=[memory_plugin],
        )

        thread: AgentThread | None = None
        result = await agent.get_response(messages="Where is the Golden Gate Bridge?", thread=thread)
        print(result.content)
```

### Option B: Single collection

see [example_single_collection.py](samples/python/example_single_collection.py)

### Option C: Store (multiple collections, shared connection)

see [example_single_store.py](samples/python/example_single_store.py)

## Behavior notes

- **No local embedding.** Never pass an `embedding_generator` — GoodMem embeds content server-side. The parameter is accepted for interface compatibility and silently ignored.
- **Upsert semantics.** GoodMem memories are immutable. If you `upsert` a record with an existing `id`, the connector deletes the old memory and creates a new one.
- **`content` is write-only in GoodMem.** The server does not return `originalContent` in search responses. Retrieved text comes from `chunkText` (a chunk of the original), which the connector maps back to your `content` field transparently.
- **Score convention.** `relevanceScore` from the GoodMem API is a raw pgvector value where lower means more similar. The connector negates it before returning, so SK's standard convention (higher = more relevant) is preserved.
-  **Filters not supported.** Passing `filter=` to `search()` raises `VectorStoreOperationNotSupportedException`. Post-filter results in application code if needed.
-  **Pre-computed vectors not supported.** Passing `vector=` to `search()` raises the same exception. Pass text only.

## Project structure

```
goodmem-semantic-kernel/           ← repo root
├── python/
│   ├── goodmem_semantic_kernel/   ← importable Python package
│   │   ├── __init__.py        # Public exports: GoodMemCollection, GoodMemStore, GoodMemSettings
│   │   ├── _client.py         # Async HTTP wrapper around the GoodMem REST API
│   │   ├── collection.py      # VectorStoreCollection + VectorSearch implementation
│   │   ├── settings.py        # GoodMemSettings (Pydantic, reads GOODMEM_* env vars)
│   │   └── store.py           # VectorStore implementation
│   └── tests/
│       ├── unit/              # Mocked unit tests (no server required)
│       └── integration/       # Live integration tests (require GoodMem server)
├── dotnet/
│   └── GoodMem.SemanticKernel/    ← .NET connector library
├── java/
│   ├── pom.xml                    ← parent Maven POM
│   └── goodmem-semantic-kernel/   ← Java connector library
│       └── src/main/java/ai/goodmem/semantickernel/
│           ├── GoodMemCollection.java   # Typed CRUD + semantic search (Reactive)
│           ├── GoodMemVectorStore.java  # Factory for multiple collections
│           ├── GoodMemPlugin.java       # SK KernelPlugin: save + recall functions
│           ├── GoodMemSchema.java       # Reflection engine for @GoodMemKey/@GoodMemData
│           ├── GoodMemKey.java          # Annotation: marks the memory ID field
│           ├── GoodMemData.java         # Annotation: marks content/metadata fields
│           ├── GoodMemClient.java       # Async HTTP client (GoodMem REST API)
│           ├── GoodMemOptions.java      # Configuration (reads GOODMEM_* env vars)
│           └── GoodMemException.java    # Runtime exception wrapper
├── samples/
│   ├── python/                    ← Runnable Python samples (Options A, B, C)
│   ├── dotnet/                    ← Runnable .NET samples
│   │   ├── ExampleStore/          # GoodMemVectorStore — multiple collections
│   │   ├── ExampleAgent/          # SK agent with memory plugin (OpenAI)
│   │   ├── ExampleAgentNvidia/    # SK agent with NVIDIA NIM
│   │   ├── ExampleAgentHuggingFace/ # SK agent with Hugging Face Inference API
│   │   └── ExampleAgentAzure/     # SK agent with Azure OpenAI
│   └── java/                      ← Runnable Java samples
│       ├── ExampleStore/          # GoodMemVectorStore — multiple collections
│       ├── ExampleCollection/     # Single GoodMemCollection — CRUD + search
│       ├── ExampleAgent/          # SK agent with GoodMemPlugin (OpenAI)
│       ├── ExampleAgentNvidia/    # SK agent with NVIDIA NIM
│       ├── ExampleAgentHuggingFace/ # SK agent with Hugging Face Inference API
│       └── ExampleAgentAzure/     # SK agent with Azure OpenAI
└── pyproject.toml
```

## API reference

### `GoodMemCollection`

The core class. Implements `VectorStoreCollection[str, TModel]` and `VectorSearch[str, TModel]`.

```python
GoodMemCollection(
    record_type=MyModel,
    collection_name="my-space",    # maps to a GoodMem Space
    settings=GoodMemSettings(),    # optional; reads GOODMEM_* env vars by default
    client=None,                   # optional; inject a pre-built GoodMemAsyncClient
)
```

| Method | Description |
|---|---|
| `ensure_collection_exists()` | Create the GoodMem space if it doesn't exist |
| `ensure_collection_deleted()` | Delete the space and all its memories |
| `collection_exists()` | Return `True` if the space exists |
| `upsert(records)` | Write one or a list of records; returns the memory ID(s) |
| `get(key=...)` / `get(keys=[...])` | Fetch memories by ID |
| `delete(keys=[...])` | Delete memories by ID |
| `search(query, top=5)` | Semantic search; returns `KernelSearchResults` |
| `create_search_function(...)` | Wrap search as a `KernelFunction` for use in agent plugins |

### `GoodMemStore`

Factory for collections. All collections from the same store share one HTTP connection.

```python
GoodMemStore(settings=GoodMemSettings())
```

| Method | Description |
|---|---|
| `get_collection(record_type, collection_name=...)` | Return a `GoodMemCollection` |
| `list_collection_names()` | List all GoodMem spaces visible to this API key |

### `GoodMemSettings`

Pydantic settings class; reads `GOODMEM_*` environment variables.

```python
GoodMemSettings(
    base_url="https://localhost:8080",
    api_key="your_key_here",
    embedder_id=None,   # auto-detected if omitted
    verify_ssl=false,
)
```
