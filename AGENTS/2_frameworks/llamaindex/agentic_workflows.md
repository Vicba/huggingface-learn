# Agentic Workflows

### Overview

- **What are Workflows?**  
  Workflows in LlamaIndex provide a structured, event-driven framework to organize code into discrete, manageable steps. They combine clear organization, type-safe communication, and built-in state management—enabling both simple sequential tasks and more complex agentic interactions.

- **Key Benefits:**  
  - Discrete step organization  
  - Event-driven architecture for flexible control flow  
  - Type-safe communication between steps  
  - Built-in state management  
  - Support for both simple workflows and multi-agent systems

---

### 1. Creating Workflows

#### Basic Workflow Creation

- **Concept:**  
  A basic workflow is defined by creating a class that inherits from `Workflow`. Functions are decorated with `@step` and are triggered by special events (`StartEvent` and `StopEvent`).

- **Example: Single-Step Workflow**
  ```python
  from llama_index.core.workflow import StartEvent, StopEvent, Workflow, step

  class MyWorkflow(Workflow):
      @step
      async def my_step(self, ev: StartEvent) -> StopEvent:
          # do something here
          return StopEvent(result="Hello, world!")

  w = MyWorkflow(timeout=10, verbose=False)
  result = await w.run()
  ```
  This example shows how to define a workflow with one step that starts with a `StartEvent` and ends by returning a `StopEvent`.

---

#### Connecting Multiple Steps

- **Concept:**  
  To build multi-step workflows, custom events are created to pass data between steps.

- **Example: Multi-Step Workflow with Data Passing**
  ```python
  from llama_index.core.workflow import StartEvent, StopEvent, Workflow, step, Event

  class ProcessingEvent(Event):
      intermediate_result: str

  class MultiStepWorkflow(Workflow):
      @step
      async def step_one(self, ev: StartEvent) -> ProcessingEvent:
          # Process initial data
          return ProcessingEvent(intermediate_result="Step 1 complete")

      @step
      async def step_two(self, ev: ProcessingEvent) -> StopEvent:
          # Use the intermediate result
          final_result = f"Finished processing: {ev.intermediate_result}"
          return StopEvent(result=final_result)

  w = MultiStepWorkflow(timeout=10, verbose=False)
  result = await w.run()
  ```
  In this workflow, the output of the first step (`ProcessingEvent`) is passed to the second step to form the final result.

---

#### Loops and Branches

- **Concept:**  
  Workflows can include loops and branches using type hints and union operators (`|`) to handle multiple event types. This allows steps to repeat or branch based on runtime conditions.

- **Example: Workflow with Looping**
  ```python
  from llama_index.core.workflow import StartEvent, StopEvent, Workflow, step, Event
  import random

  class ProcessingEvent(Event):
      intermediate_result: str

  class LoopEvent(Event):
      loop_output: str

  class MultiStepWorkflow(Workflow):
      @step
      async def step_one(self, ev: StartEvent | LoopEvent) -> ProcessingEvent | LoopEvent:
          if random.randint(0, 1) == 0:
              print("Bad thing happened")
              return LoopEvent(loop_output="Back to step one.")
          else:
              print("Good thing happened")
              return ProcessingEvent(intermediate_result="First step complete.")

      @step
      async def step_two(self, ev: ProcessingEvent) -> StopEvent:
          # Use the intermediate result
          final_result = f"Finished processing: {ev.intermediate_result}"
          return StopEvent(result=final_result)

  w = MultiStepWorkflow(verbose=False)
  result = await w.run()
  ```
  This snippet demonstrates using a loop—if a "bad thing" occurs, the workflow emits a `LoopEvent` to trigger a retry.

---

#### Drawing Workflows

- **Concept:**  
  LlamaIndex includes a utility to visualize workflows by generating an HTML file that depicts all possible flows.

- **Example:**
  ```python
  from llama_index.utils.workflow import draw_all_possible_flows

  # Assuming 'w' is your workflow instance
  draw_all_possible_flows(w, "flow.html")
  ```
  This command creates an HTML file named `flow.html` that visually represents the workflow’s structure.

---

#### State Management

- **Concept:**  
  Workflows can maintain shared state across steps using the `Context` object. This is useful when subsequent steps need access to previously stored values.

- **Example: Using Context for State Management**
  ```python
  from llama_index.core.workflow import Context, StartEvent, StopEvent, step

  @step
  async def query(self, ctx: Context, ev: StartEvent) -> StopEvent:
      # Store a value in the context
      await ctx.set("query", "What is the capital of France?")

      # Perform operations with context
      val = ...  # Some processing here

      # Retrieve the value from the context
      query = await ctx.get("query")
      return StopEvent(result=val)
  ```
  This example illustrates setting and getting state within a workflow step.

---

### 2. Automating Workflows with Multi-Agent Workflows

- **Concept:**  
  The `AgentWorkflow` class allows creation of multi-agent systems where different agents (each with specific capabilities) collaborate to complete a task. A designated root agent receives the user message and may delegate tasks to other agents.

- **Example: Multi-Agent Workflow**
  ```python
  from llama_index.core.agent.workflow import AgentWorkflow, ReActAgent
  from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

  # Define some tools as simple functions
  def add(a: int, b: int) -> int:
      """Add two numbers."""
      return a + b

  def multiply(a: int, b: int) -> int:
      """Multiply two numbers."""
      return a * b

  llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")

  # Create agents using ReActAgent (works for any LLM)
  multiply_agent = ReActAgent(
      name="multiply_agent",
      description="Can multiply two integers",
      system_prompt="A helpful assistant that multiplies numbers.",
      tools=[multiply],
      llm=llm,
  )

  addition_agent = ReActAgent(
      name="add_agent",
      description="Can add two integers",
      system_prompt="A helpful assistant that adds numbers.",
      tools=[add],
      llm=llm,
  )

  # Create a multi-agent workflow; designate a root agent to start the process
  workflow = AgentWorkflow(
      agents=[multiply_agent, addition_agent],
      root_agent="multiply_agent",
  )

  # Run the system
  response = await workflow.run(user_msg="Can you add 5 and 3?")
  ```
  This demonstrates how multiple agents with different specializations work together in a coordinated workflow.

- **Stateful Multi-Agent Workflow:**  
  Agent workflows can also be initialized with an initial state (e.g., to count function calls) which is shared across agents.
  ```python
  from llama_index.core.workflow import Context

  # Example functions that update state
  async def add(ctx: Context, a: int, b: int) -> int:
      cur_state = await ctx.get("state")
      cur_state["num_fn_calls"] += 1
      await ctx.set("state", cur_state)
      return a + b

  async def multiply(ctx: Context, a: int, b: int) -> int:
      cur_state = await ctx.get("state")
      cur_state["num_fn_calls"] += 1
      await ctx.set("state", cur_state)
      return a * b

  workflow = AgentWorkflow(
      agents=[multiply_agent, addition_agent],
      root_agent="multiply_agent",
      initial_state={"num_fn_calls": 0},
      state_prompt="Current state: {state}. User message: {msg}",
  )

  ctx = Context(workflow)
  response = await workflow.run(user_msg="Can you add 5 and 3?", ctx=ctx)
  state = await ctx.get("state")
  print(state["num_fn_calls"])
  ```
  In this example, the workflow's state tracks how many function calls have been made across agents.
