# What are agents?

an Agent is a system that uses an AI Model (typically an LLM) as its core reasoning engine, to:

- Understand natural language: Interpret and respond to human instructions in a meaningful way.

- Reason and plan: Analyze information, make decisions, and devise strategies to solve problems.

- Interact with its environment: Gather information, take actions, and observe the results of those actions.

### How are LLMs used in AI Agents?

LLMs are a key component of AI Agents, providing the foundation for understanding and generating human language.

They can interpret user instructions, maintain context in conversations, define a plan and decide which tools to use.

We will explore these steps in more detail in this Unit, but for now, what you need to understand is that the LLM is the brain of the Agent.

## chat templates
This is where chat templates come in. They act as the bridge between conversational messages (user and assistant turns) and the specific formatting requirements of your chosen LLM. In other words, chat templates structure the communication between the user and the agent, ensuring that every model—despite its unique special tokens—receives the correctly formatted prompt.

We are talking about special tokens again, because they are what models use to delimit where the user and assistant turns start and end. Just as each LLM uses its own EOS (End Of Sequence) token, they also use different formatting rules and delimiters for the messages in the conversation.

Base Models vs. Instruct Models

Another point we need to understand is the difference between a Base Model vs. an Instruct Model:

A Base Model is trained on raw text data to predict the next token.

An Instruct Model is fine-tuned specifically to follow instructions and engage in conversations. For example, SmolLM2-135M is a base model, while SmolLM2-135M-Instruct is its instruction-tuned variant.

To make a Base Model behave like an instruct model, we need to format our prompts in a consistent way that the model can understand. This is where chat templates come in.

ChatML is one such template format that structures conversations with clear role indicators (system, user, assistant). If you have interacted with some AI API lately, you know that’s the standard practice.

It’s important to note that a base model could be fine-tuned on different chat templates, so when we’re using an instruct model we need to make sure we’re using the correct chat template.

In transformers, chat templates include Jinja2 code that describes how to transform the ChatML list of JSON messages, as presented in the above examples, into a textual representation of the system-level instructions, user messages and assistant responses that the model can understand.

This structure helps maintain consistency across interactions and ensures the model responds appropriately to different types of inputs.

```json
{% for message in messages %}
{% if loop.first and messages[0]['role'] != 'system' %}
<|im_start|>system
You are a helpful AI assistant named SmolLM, trained by Hugging Face
<|im_end|>
{% endif %}
<|im_start|>{{ message['role'] }}
{{ message['content'] }}<|im_end|>
{% endfor %}
```
with this message:
```json
messages = [
    {"role": "system", "content": "You are a helpful assistant focused on technical topics."},
    {"role": "user", "content": "Can you explain what a chat template is?"},
    {"role": "assistant", "content": "A chat template structures conversations between users and AI models..."},
    {"role": "user", "content": "How do I use it ?"},
]
```
gives this string
```json
<|im_start|>system
You are a helpful assistant focused on technical topics.<|im_end|>
<|im_start|>user
Can you explain what a chat template is?<|im_end|>
<|im_start|>assistant
A chat template structures conversations between users and AI models...<|im_end|>
<|im_start|>user
How do I use it ?<|im_end|>
```

## Tools

A good tool should be something that complements the power of an LLM.

For instance, if you need to perform arithmetic, giving a calculator tool to your LLM will provide better results than relying on the native capabilities of the model.

Furthermore, LLMs predict the completion of a prompt based on their training data, which means that their internal knowledge only includes events prior to their training. Therefore, if your agent needs up-to-date data you must provide it through some tool.

LLMs, as we saw, can only receive text inputs and generate text outputs. They have no way to call tools on their own. What we mean when we talk about providing tools to an Agent, is that we teach the LLM about the existence of tools, and ask the model to generate text that will invoke tools when it needs to. For example, if we provide a tool to check the weather at a location from the Internet, and then ask the LLM about the weather in Paris, the LLM will recognize that question as a relevant opportunity to use the “weather” tool we taught it about. The LLM will generate text, in the form of code, to invoke that tool. It is the responsibility of the Agent to parse the LLM’s output, recognize that a tool call is required, and invoke the tool on the LLM’s behalf. The output from the tool will then be sent back to the LLM, which will compose its final response for the user.

The output from a tool call is another type of message in the conversation. Tool calling steps are typically not shown to the user: the Agent retrieves the conversation, calls the tool(s), gets the outputs, adds them as a new conversation message, and sends the updated conversation to the LLM again. From the user’s point of view, it’s like the LLM had used the tool, but in fact it was our application code (the Agent) who did it.

Example code of tool decorator:
```python
class Tool:
    """
    A class representing a reusable piece of code (Tool).
    
    Attributes:
        name (str): Name of the tool.
        description (str): A textual description of what the tool does.
        func (callable): The function this tool wraps.
        arguments (list): A list of argument.
        outputs (str or list): The return type(s) of the wrapped function.
    """
    def __init__(self, 
                 name: str, 
                 description: str, 
                 func: callable, 
                 arguments: list,
                 outputs: str):
        self.name = name
        self.description = description
        self.func = func
        self.arguments = arguments
        self.outputs = outputs

    def to_string(self) -> str:
        """
        Return a string representation of the tool, 
        including its name, description, arguments, and outputs.
        """
        args_str = ", ".join([
            f"{arg_name}: {arg_type}" for arg_name, arg_type in self.arguments
        ])
        
        return (
            f"Tool Name: {self.name},"
            f" Description: {self.description},"
            f" Arguments: {args_str},"
            f" Outputs: {self.outputs}"
        )

    def __call__(self, *args, **kwargs):
        """
        Invoke the underlying function (callable) with provided arguments.
        """
        return self.func(*args, **kwargs)
```

To summarize, we learned:

What Tools Are: Functions that give LLMs extra capabilities, such as performing calculations or accessing external data.

How to Define a Tool: By providing a clear textual description, inputs, outputs, and a callable function.

Why Tools Are Essential: They enable Agents to overcome the limitations of static model training, handle real-time tasks, and perform specialized actions.

Now, we can move on to the Agent Workflow where you’ll see how an Agent observes, thinks, and acts. This brings together everything we’ve covered so far and sets the stage for creating your own fully functional AI Agent.

## Re-act

The Re-Act Approach

A key method is the ReAct approach, which is the concatenation of “Reasoning” (Think) with “Acting” (Act).

ReAct is a simple prompting technique that appends “Let’s think step by step” before letting the LLM decode the next tokens.

Indeed, prompting the model to think “step by step” encourages the decoding process toward next tokens that generate a plan, rather than a final solution, since the model is encouraged to decompose the problem into sub-tasks.

We have recently seen a lot of interest for reasoning strategies. This is what's behind models like Deepseek R1 or OpenAI's o1, which have been fine-tuned to "think before answering".
These models have been trained to always include specific thinking sections (enclosed between <think> and </think> special tokens). This is not just a prompting technique like ReAct, but a training method where the model learns to generate these sections after analyzing thousands of examples that show what we expect it to do.

## Types of agents
- JSON Agent: The Action to take is specified in JSON format.
- CODE Agent: The Agent writes a code block that is interpreted externally.
- Function-calling Agent: It is a subcategory of the JSON Agent which has been fine-tuned to generate a new message for each action.

One crucial part of an agent is the ability to STOP generating new tokens when an action is complete, and that is true for all formats of Agent: JSON, code, or function-calling. This prevents unintended output and ensures that the agent’s response is clear and precise.

The LLM only handles text and uses it to describe the action it wants to take and the parameters to supply to the tool.

The Stop and Parse Approach

One key method for implementing actions is the stop and parse approach. This method ensures that the agent’s output is structured and predictable:

Generation in a Structured Format:
The agent outputs its intended action in a clear, predetermined format (JSON or code).

Halting Further Generation:
Once the action is complete, the agent stops generating additional tokens. This prevents extra or erroneous output.

Parsing the Output:
An external parser reads the formatted action, determines which Tool to call, and extracts the required parameters.

```python
# Code Agent Example: Retrieve Weather Information
def get_weather(city):
    import requests
    api_url = f"https://api.weather.com/v1/location/{city}?apiKey=YOUR_API_KEY"
    response = requests.get(api_url)
    if response.status_code == 200:
        data = response.json()
        return data.get("weather", "No weather information available")
    else:
        return "Error: Unable to fetch weather data."

# Execute the function and prepare the final answer
result = get_weather("New York")
final_answer = f"The current weather in New York is: {result}"
print(final_answer)
```

## Function calling
While here, with function-calling, the Agent is fine-tuned (trained) to use Tools.

If we have one piece of advice now, it’s to try to fine-tune different models. The best way to learn is by trying.

