# LlamaIndex

LlamaHub is a registry of hundreds of integrations, agents and tools that you can use within LlamaIndex.

LlamaIndex installation instructions are available as a well-structured overview on LlamaHub. This might be a bit overwhelming at first, but most of the installation commands generally follow an easy-to-remember format:

```shell
pip install llama-index-{component-type}-{framework-name}
```
Let’s try to install the dependencies for an LLM and embedding component using the Hugging Face inference API integration.

```sh
pip install llama-index-llms-huggingface-api llama-index-embeddings-huggingface
```
Once installed, we can see the usage patterns. You’ll notice that the import paths follow the install command! Underneath, we can see an example of the usage of the Hugging Face inference API for an LLM component.

```python
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

llm = HuggingFaceInferenceAPI(
    model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
    temperature=0.7,
    max_tokens=100,
    token="hf_xxx",
)

llm.complete("Hello, how are you?")
# I am good, how can I help you today?
```
Wonderful, we now know how to find, install and use the integrations for the components we need. Let’s dive deeper into the components and see how we can use them to build our own agents.

Below is an updated summary that includes the key examples provided on the page:

---

### Overview

- **Purpose of Components:**  
  LlamaIndex offers a suite of components to help agents (like Alfred, the butler agent) process, retrieve, and use data. The focus here is on the **QueryEngine** component, which is central to Retrieval-Augmented Generation (RAG). RAG augments language models by retrieving up-to-date, relevant data to answer questions or perform tasks.

---

### Key Stages in a RAG Pipeline

1. **Loading:**
   - **Goal:** Ingest data from various sources (e.g., files, PDFs, websites, APIs).  
   - **Tools & Examples:**
     - **SimpleDirectoryReader:**  
       Loads files from a local directory and converts them into `Document` objects.
       ```python
       from llama_index.core import SimpleDirectoryReader

       reader = SimpleDirectoryReader(input_dir="path/to/directory")
       documents = reader.load_data()
       ```
     - **LlamaParse:**  
       Used for PDF parsing via a managed API.
     - **LlamaHub:**  
       A registry offering hundreds of integrations for loading data.

2. **Indexing:**
   - **Goal:** Convert documents into searchable formats by creating vector embeddings.
   - **Process & Examples:**
     - Documents are split into smaller pieces called **Nodes**.
     - **Using an IngestionPipeline:**  
       Combines transformations like splitting text into sentences and embedding.
       ```python
       from llama_index.core import Document
       from llama_index.embeddings.huggingface import HuggingFaceEmbedding
       from llama_index.core.node_parser import SentenceSplitter
       from llama_index.core.ingestion import IngestionPipeline

       pipeline = IngestionPipeline(
           transformations=[
               SentenceSplitter(chunk_overlap=0),
               HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5"),
           ]
       )

       nodes = await pipeline.arun(documents=[Document.example()])
       ```

3. **Storing:**
   - **Goal:** Save the processed data (indexed nodes) for efficient retrieval.
   - **Example with Chroma Vector Store:**
     - Install the vector store:
       ```bash
       pip install llama-index-vector-stores-chroma
       ```
     - Use ChromaDB to store embeddings:
       ```python
       import chromadb
       from llama_index.vector_stores.chroma import ChromaVectorStore

       db = chromadb.PersistentClient(path="./alfred_chroma_db")
       chroma_collection = db.get_or_create_collection("alfred")
       vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

       pipeline = IngestionPipeline(
           transformations=[
               SentenceSplitter(chunk_size=25, chunk_overlap=0),
               HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5"),
           ],
           vector_store=vector_store,
       )
       ```

4. **Querying:**
   - **Goal:** Retrieve relevant Nodes using various interfaces.
   - **Query Interfaces & Example:**
     - **QueryEngine:**  
       Wraps the vector store index and connects to an LLM for generating responses.
       ```python
       from llama_index.core import VectorStoreIndex
       from llama_index.embeddings.huggingface import HuggingFaceEmbedding
       from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

       embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
       index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

       llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")
       query_engine = index.as_query_engine(
           llm=llm,
           response_mode="tree_summarize",
       )
       response = query_engine.query("What is the meaning of life?")
       # Expected output: "The meaning of life is 42"
       ```

5. **Response Processing:**
   - **Goal:** Process and refine the response using customizable strategies.
   - **Strategies:**
     - **Refine:** Iteratively refines the answer with multiple LLM calls.
     - **Compact:** Concatenates text chunks before querying, reducing LLM calls.
     - **Tree_Summarize:** Builds a hierarchical (tree-like) summary from the retrieved chunks.
   - **Low-Level API:**  
     Allows fine-tuning of each step for custom workflows.

6. **Evaluation and Observability:**
   - **Goal:** Assess answer quality and track performance.
   - **Examples & Tools:**
     - **Evaluators:**  
       For example, using the **FaithfulnessEvaluator** to check if the response is supported by context.
       ```python
       from llama_index.core.evaluation import FaithfulnessEvaluator

       evaluator = FaithfulnessEvaluator(llm=llm)
       response = query_engine.query("What battles took place in New York City in the American Revolution?")
       eval_result = evaluator.evaluate_response(response=response)
       print(eval_result.passing)
       ```
     - **Observability with LlamaTrace:**  
       Install and configure tracing to monitor workflow performance.
       ```bash
       pip install -U llama-index-callbacks-arize-phoenix
       ```
       ```python
       import llama_index
       import os

       PHOENIX_API_KEY = "<PHOENIX_API_KEY>"
       os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"api_key={PHOENIX_API_KEY}"
       llama_index.core.set_global_handler(
           "arize_phoenix",
           endpoint="https://llamatrace.com/v1/traces"
       )
       ```

---

### Overview

- **Purpose of Tools:**  
  In LlamaIndex, defining clear and well-documented tool interfaces is essential for ensuring that agents (or LLMs acting as agents) can effectively understand and interact with them. Tools in LlamaIndex serve as bridges between natural language instructions and underlying Python functions or query engines, much like an API does for human engineers.

- **Main Tool Types:**  
  The page categorizes tools into four primary types:
  1. **FunctionTool:** Wraps any Python function into a tool.
  2. **QueryEngineTool:** Exposes query engines as callable tools.
  3. **Toolspecs:** Community-created collections of tools for specific tasks (e.g., Gmail).
  4. **Utility Tools:** Manage large data responses by integrating data loading, indexing, and querying in one step.

---

### 1. Creating a FunctionTool

- **Description:**  
  A FunctionTool allows you to convert a Python function into a tool that an agent can call. It automatically extracts the function’s purpose from its name, signature, and description, which is key for proper usage by LLMs.

- **Example:**
  ```python
  from llama_index.core.tools import FunctionTool

  def get_weather(location: str) -> str:
      """Useful for getting the weather for a given location."""
      print(f"Getting weather for {location}")
      return f"The weather in {location} is sunny"

  tool = FunctionTool.from_defaults(
      get_weather,
      name="my_weather_tool",
      description="Useful for getting the weather for a given location.",
  )
  tool.call("New York")
  ```
  This snippet demonstrates how to define a simple weather-checking function and wrap it as a tool, making it accessible for an agent to call when needed.  
  cite21†this notebook, cite22†Function Calling Guide

---

### 2. Creating a QueryEngineTool

- **Description:**  
  The QueryEngineTool transforms an existing query engine (which is often the backbone for retrieving and synthesizing data) into a tool. This is especially useful when agents need to perform complex queries by leveraging pre-built indexes.

- **Example:**
  ```python
  from llama_index.core import VectorStoreIndex
  from llama_index.core.tools import QueryEngineTool
  from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
  from llama_index.embeddings.huggingface import HuggingFaceEmbedding
  from llama_index.vector_stores.chroma import ChromaVectorStore

  embed_model = HuggingFaceEmbedding("BAAI/bge-small-en-v1.5")

  db = chromadb.PersistentClient(path="./alfred_chroma_db")
  chroma_collection = db.get_or_create_collection("alfred")
  vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

  index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

  llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")
  query_engine = index.as_query_engine(llm=llm)
  tool = QueryEngineTool.from_defaults(query_engine, name="some useful name", description="some useful description")
  ```
  Here, a QueryEngine is created from a vector store index, wrapped as a tool, and made ready for agent usage.  
  cite21†this notebook

---

### 3. Creating Toolspecs

- **Description:**  
  Toolspecs are essentially curated collections of tools designed to work together for specific tasks. For example, a Toolspec for Gmail may include tools for reading, composing, and organizing emails, bundled in a user-friendly way.

- **Example:**
  - **Installation:**
    ```bash
    pip install llama-index-tools-google
    ```
  - **Code:**
    ```python
    from llama_index.tools.google import GmailToolSpec

    tool_spec = GmailToolSpec()
    tool_spec_list = tool_spec.to_tool_list()

    # View the metadata of each tool in the spec
    [(tool.metadata.name, tool.metadata.description) for tool in tool_spec_list]
    ```
  This demonstrates how to install and load a Google Toolspec, converting it to a list of tools with useful metadata.  
  cite24†LlamaHub

---

### 4. Utility Tools

- **Description:**  
  Utility Tools are designed to handle scenarios where a single API call might return too much data. They help manage data by combining loading, indexing, and searching within one unified tool call. Two key utility tools mentioned are:
  - **OnDemandToolLoader:** Turns any LlamaIndex data loader into a tool that loads and processes data on-demand.
  - **LoadAndSearchToolSpec:** Wraps an existing tool to provide two functions—a loader for ingesting data and a search tool for querying the resulting index.

- **Usage:**  
  While explicit code examples are not provided in the text, the description explains that these tools streamline the process of dealing with large data sets and making them agent-friendly by integrating multiple steps (loading, indexing, and querying) into one tool call.


Below is a detailed summary with the key points and practical examples from the "Using Agents in LlamaIndex" page:

---

### Overview

- **What is an Agent?**  
  An agent in LlamaIndex is an AI-powered system that combines reasoning, planning, and the ability to call tools to accomplish user-defined objectives. This includes basic task execution as well as more complex interactions like planning and multi-step reasoning.

- **Types of Agents:**  
  LlamaIndex supports three primary types of agents:
  1. **Function Calling Agents:** Designed for LLMs that support calling functions directly.
  2. **ReAct Agents:** Work with any LLM via text or chat endpoints and provide detailed reasoning steps.
  3. **Advanced Custom Agents:** For more complex workflows that require fine-tuned control over multi-step processes.  
  cite28†Image: Agents

---

### 1. Initializing Agents

- **Setting Up Basic Agents:**  
  Agents are created by supplying them with a set of tools (or functions) that define what actions they can perform. The agent then uses an LLM (e.g., via HuggingFaceInferenceAPI) to generate responses and decide which tool to invoke.

- **Example: Basic Agent with a Function Tool**
  ```python
  from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
  from llama_index.core.agent.workflow import AgentWorkflow
  from llama_index.core.tools import FunctionTool

  # Define a sample tool (multiplication function)
  def multiply(a: int, b: int) -> int:
      """Multiplies two integers and returns the resulting integer"""
      return a * b

  # Initialize the LLM
  llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")

  # Initialize the agent with the function tool
  agent = AgentWorkflow.from_tools_or_functions(
      [FunctionTool.from_defaults(multiply)],
      llm=llm
  )
  ```
  - **Running the Agent:**  
    Agents are asynchronous. You can run them in a stateless manner or use a `Context` to remember previous interactions.
    ```python
    # Stateless execution
    response = await agent.run("What is 2 times 2?")

    # Stateful execution using Context to remember past interactions
    from llama_index.core.workflow import Context

    ctx = Context(agent)
    response = await agent.run("My name is Bob.", ctx=ctx)
    response = await agent.run("What was my name again?", ctx=ctx)
    ```
  cite21†this notebook, cite22†excellent async guide

---

### 2. Creating RAG Agents with QueryEngineTools

- **Agentic RAG:**  
  RAG (Retrieval-Augmented Generation) agents leverage a QueryEngine to fetch and synthesize data from an indexed document store. This allows an agent to answer questions based on up-to-date, context-rich information.

- **Example: Wrapping a QueryEngine as a Tool**
  ```python
  from llama_index.core.tools import QueryEngineTool

  # Create a query engine from an index (as shown in the Components section)
  query_engine = index.as_query_engine(llm=llm, similarity_top_k=3)

  # Wrap the query engine as a tool
  query_engine_tool = QueryEngineTool.from_defaults(
      query_engine=query_engine,
      name="name",
      description="a specific description",
      return_direct=False,
  )

  # Initialize an agent using the query engine tool
  query_engine_agent = AgentWorkflow.from_tools_or_functions(
      [query_engine_tool],
      llm=llm,
      system_prompt="You are a helpful assistant that has access to a database containing persona descriptions. "
  )
  ```
  This approach gives the agent the flexibility to decide when to query data rather than simply retrieving documents automatically.  
  cite29†Image: Agentic RAG

---

### 3. Creating Multi-agent Systems

- **Concept:**  
  Multi-agent systems allow for splitting responsibilities among different agents, each specialized in a narrow domain. This setup can improve overall performance by having agents that focus on specific tasks.

- **Example: Multi-agent Workflow**
  ```python
  from llama_index.core.agent.workflow import (
      AgentWorkflow,
      FunctionAgent,
      ReActAgent,
  )

  # Define basic arithmetic tools
  def add(a: int, b: int) -> int:
      """Add two numbers."""
      return a + b

  def subtract(a: int, b: int) -> int:
      """Subtract two numbers."""
      return a - b

  # Create specialized agent configurations (using ReActAgent for broader compatibility)
  calculator_agent = ReActAgent(
      name="calculator",
      description="Performs basic arithmetic operations",
      system_prompt="You are a calculator assistant. Use your tools for any math operation.",
      tools=[add, subtract],
      llm=llm,
  )

  query_agent = ReActAgent(
      name="info_lookup",
      description="Looks up information about XYZ",
      system_prompt="Use your tool to query a RAG system to answer information about XYZ",
      tools=[query_engine_tool],
      llm=llm
  )

  # Create a multi-agent workflow with a designated root agent
  agent = AgentWorkflow(
      agents=[calculator_agent, query_agent], root_agent="calculator"
  )

  # Run the multi-agent system
  response = await agent.run(user_msg="Can you add 5 and 3?")
  ```
  In this setup:
  - The `calculator_agent` handles arithmetic operations.
  - The `query_agent` is responsible for fetching additional information using the query engine.
  - The system designates a root agent (here, "calculator") that coordinates the overall workflow.
  cite26†< > Update on GitHub
