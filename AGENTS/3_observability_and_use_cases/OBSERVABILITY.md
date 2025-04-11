# Observability
---

## 🔎 What is Observability?

**Observability** refers to the ability to understand the internal state of an AI agent by examining external outputs—such as **logs, metrics, and traces**. For AI agents, this means tracking:
- Actions taken by the agent
- Usage of tools or APIs
- Model invocations
- Responses generated

This data helps developers debug, optimize, and enhance agent performance.

---

## 🔭 Why Agent Observability Matters

Without observability, agents are **black boxes**. Observability provides transparency and is critical for:
- Understanding trade-offs (e.g., cost vs. accuracy)
- Measuring latency and response time
- Detecting issues like **harmful content** or **prompt injections**
- Monitoring **user feedback**

It is essential to transition from a working demo to a **production-ready** AI agent.

---

## 🔨 Observability Tools

Common tools include:
- **Langfuse**
- **Arize**

These platforms:
- Collect detailed **traces**
- Provide **dashboards** for real-time monitoring
- Help detect bugs and performance bottlenecks

### Tool Variations
- Some are **open-source**, encouraging community-driven features
- Some focus only on parts of LLMOps (e.g., **prompt management**, **evaluation**, etc.)
- Others provide end-to-end LLMOps support

Frameworks like **smolagents** use **OpenTelemetry** to expose metadata. Tools also develop **custom instrumentations** for enhanced flexibility.

---

## 🔬 Traces and Spans

These are core concepts in observability:

- **Traces**: Represent the entire process of handling a task or query
- **Spans**: Break down the trace into smaller steps, like an LLM call or a tool execution

---

## 📊 Key Metrics to Monitor

### 1. **Latency**
- Measures response speed
- Can be tracked at task or span level
- Long latencies degrade user experience

### 2. **Costs**
- Tracks cost per agent run
- Important in LLM-based agents where API usage is metered
- Helps identify unnecessary model calls or expensive operations

### 3. **Request Errors**
- Identifies failed API or tool calls
- Can inform the setup of **fallbacks** and **retries**

### 4. **User Feedback**
- Explicit: Ratings (e.g., 👍/👎, ⭐ 1–5)
- Valuable for spotting issues from the user’s perspective

### 5. **Implicit Feedback**
- Behavioral signals (e.g., repeat queries, retries)
- Useful to detect dissatisfaction without formal feedback

### 6. **Accuracy**
- Measures correctness of outputs
- Varies by context (e.g., solving math problems, retrieving accurate info)
- Requires **ground truth definitions** and evaluation labels

### 7. **Automated Evaluation Metrics**
- Use of LLMs or libraries to evaluate:
  - Helpfulness
  - Harmfulness
  - Accuracy
- Tools: **RAGAS**, **LLM Guard**, etc.

A **combination** of all these metrics gives the best insight into the agent’s health.

---

## 👍 Evaluating AI Agents

### Key Idea:
**Observability collects data; evaluation interprets it** to make decisions about performance and improvement.

### Why Evaluate?
- AI agents are **non-deterministic**
- Models can **drift** over time
- Ongoing evaluation is needed to maintain and improve performance

---

## 🥷 Offline Evaluation

Evaluation in a **controlled environment**:
- Use of **test datasets** with known outputs
- Useful for:
  - Development phase
  - Regression testing
  - CI/CD pipelines

**Benefits:**
- Clear metrics (due to known ground truth)
- Repeatable and automatable

**Challenges:**
- Test sets may become outdated
- Must include edge cases and reflect real-world queries
- Use both small ("smoke tests") and large evaluation sets

---

## 🔄 Online Evaluation

Evaluation in **real-time**, during actual usage:
- Captures live **user interactions**
- Detects:
  - Model drift
  - Unexpected inputs
  - Changing usage patterns

**Includes:**
- Success/failure tracking
- User satisfaction scoring
- A/B or shadow testing

**Challenges:**
- Hard to label real-world interactions
- Must infer from feedback or downstream metrics

---

## 🤝 Combining Offline and Online Evaluation

**Best practice** is to use both:
- **Offline** for structured benchmarking
- **Online** for detecting real-world issues

### Continuous Loop:
1. Offline evaluation
2. Deploy updated agent
3. Monitor online metrics
4. Gather failure examples
5. Add failures to offline tests
6. Repeat

This cycle ensures **constant learning and improvement**.

---

Let me know if you'd like this turned into a slide deck, a checklist, or a step-by-step implementation guide!