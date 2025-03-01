# SmolAgents

### 1️⃣ Why Use smolagents

smolagents is one of the many open-source agent frameworks available for application development. Alternative options include LlamaIndex and LangGraph, which are also covered in other modules in this course. smolagents offers several key features that might make it a great fit for specific use cases, but we should always consider all options when selecting a framework. We’ll explore the advantages and drawbacks of using smolagents, helping you make an informed decision based on your project’s requirements.

### 2️⃣ CodeAgents

CodeAgents are the primary type of agent in smolagents. Instead of generating JSON or text, these agents produce Python code to perform actions. This module explores their purpose, functionality, and how they work, along with hands-on examples to showcase their capabilities.

### 3️⃣ ToolCallingAgents

ToolCallingAgents are the second type of agent supported by smolagents. Unlike CodeAgents, which generate Python code, these agents rely on JSON/text blobs that the system must parse and interpret to execute actions. This module covers their functionality, their key differences from CodeAgents, and it provides an example to illustrate their usage.

### 4️⃣ Tools

As we saw in Unit 1, tools are functions that an LLM can use within an agentic system, and they act as the essential building blocks for agent behavior. This module covers how to create tools, their structure, and different implementation methods using the Tool class or the @tool decorator. You’ll also learn about the default toolbox, how to share tools with the community, and how to load community-contributed tools for use in your agents.

### 5️⃣ Retrieval Agents

Retrieval agents allow models access to knowledge bases, making it possible to search, synthesize, and retrieve information from multiple sources. They leverage vector stores for efficient retrieval and implement Retrieval-Augmented Generation (RAG) patterns. These agents are particularly useful for integrating web search with custom knowledge bases while maintaining conversation context through memory systems. This module explores implementation strategies, including fallback mechanisms for robust information retrieval.

### 6️⃣ Multi-Agent Systems

Orchestrating multiple agents effectively is crucial for building powerful, multi-agent systems. By combining agents with different capabilities—such as a web search agent with a code execution agent—you can create more sophisticated solutions. This module focuses on designing, implementing, and managing multi-agent systems to maximize efficiency and reliability.

### 7️⃣ Vision and Browser agents

Vision agents extend traditional agent capabilities by incorporating Vision-Language Models (VLMs), enabling them to process and interpret visual information. This module explores how to design and integrate VLM-powered agents, unlocking advanced functionalities like image-based reasoning, visual data analysis, and multimodal interactions. We will also use vision agents to build a browser agent that can browse the web and extract information from it.

## When to use smolagents?

With these advantages in mind, when should we use smolagents over other frameworks?

smolagents is ideal when:
- You need a lightweight and minimal solution.
- You want to experiment quickly without complex configurations.
-  Your application logic is straightforward.

Unlike other frameworks where agents write actions in JSON, smolagents focuses on tool calls in code, simplifying the execution process. This is because there’s no need to parse the JSON in order to build code that calls the tools: the output can be executed directly.

## Agent Types in smolagents
Agents in smolagents operate as multi-step agents.
Each MultiStepAgent performs:
- One thought
- One tool call and execution
In addition to using CodeAgent as the primary type of agent, smolagents also supports ToolCallingAgent, which writes tool calls in JSON.

## Tools
To interact with a tool, the LLM needs an interface description with these key components:
- Name: What the tool is called
- Tool description: What the tool does
- Input types and descriptions: What arguments the tool accepts
- Output type: What the tool returns

In smolagents, tools can be defined in two ways:
- Using the @tool decorator for simple function-based tools
- Creating a subclass of Tool for more complex functionality

### using the decorator
Using this approach, we define a function with:
- A clear and descriptive function name that helps the LLM understand its purpose.
- Type hints for both inputs and outputs to ensure proper usage.
- A detailed description, including an Args: section where each argument is explicitly described. These descriptions provide valuable context for the LLM, so it’s important to write them carefully.

### using the class
This approach involves creating a subclass of Tool. For complex tools, we can implement a class instead of a Python function. The class wraps the function with metadata that helps the LLM understand how to use it effectively. In this class, we define:
- name: The tool’s name.
- description: A description used to populate the agent’s system prompt.
- inputs: A dictionary with keys type and description, providing information to help the Python interpreter process inputs.
- output_type: Specifies the expected output type.
- forward: The method containing the inference logic to execute.

# default pre-built tools
smolagents comes with a set of pre-built tools that can be directly injected into your agent. The default toolbox includes:
- PythonInterpreterTool
- FinalAnswerTool
- UserInputTool
- DuckDuckGoSearchTool
- GoogleSearchTool
- VisitWebpageTool

### Sharing a tool to the hub 
pushing
```py
party_theme_tool.push_to_hub("{your_username}/party_theme_tool", token="<YOUR_HUGGINGFACEHUB_API_TOKEN>")
```
importing
```py
from smolagents import load_tool, CodeAgent, HfApiModel

image_generation_tool = load_tool(
    "m-ric/text-to-image",
    trust_remote_code=True
)

agent = CodeAgent(
    tools=[image_generation_tool],
    model=HfApiModel()
)

agent.run("Generate an image of a luxurious superhero-themed party at Wayne Manor with made-up superheros.")
```

## Agentic RAG systems
Agentic RAG (Retrieval-Augmented Generation) extends traditional RAG systems by combining autonomous agents with dynamic knowledge retrieval.

## Multi-agent systems
Multi-agent systems enable specialized agents to collaborate on complex tasks, improving modularity, scalability, and robustness. Instead of relying on a single agent, tasks are distributed among agents with distinct capabilities.

A typical setup might include:
- A Manager Agent for task delegation
- A Code Interpreter Agent for code execution
- A Web Search Agent for information retrieval

## Visual agents
https://huggingface.co/learn/agents-course/unit2/smolagents/vision_agents

