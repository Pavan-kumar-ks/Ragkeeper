GOLDEN_SET: list[dict] = [
    {
        "question": "How do I create an agent in LangChain?",
        "expected_sources": ["langchain/agents.mdx", "langchain/quickstart.mdx"],
        "reference_answer": "Use create_agent(model, tools=[...]) from langchain.agents, passing a model and a list of tools.",
    },
    {
        "question": "What are tools in LangChain and what can they let an agent do?",
        "expected_sources": ["langchain/tools.mdx"],
        "reference_answer": "Tools extend what agents can do, letting them fetch real-time data, execute code, query external databases, and take actions in the world.",
    },
    {
        "question": "What capabilities do LangChain chat models support beyond plain text generation?",
        "expected_sources": ["langchain/models.mdx"],
        "reference_answer": "Many models support tool calling, structured output, multimodality, and reasoning.",
    },
    {
        "question": "What are the three main components of a LangChain message?",
        "expected_sources": ["langchain/messages.mdx"],
        "reference_answer": "A message has a role (e.g. system, user), content (text, images, audio, documents), and metadata (response info, message IDs, token usage).",
    },
    {
        "question": "Why is streaming important for LLM applications built with LangChain?",
        "expected_sources": ["langchain/streaming.mdx"],
        "reference_answer": "Streaming surfaces real-time updates and improves responsiveness by displaying output progressively instead of waiting for a full response, which matters given LLM latency.",
    },
    {
        "question": "How does structured output work with create_agent in LangChain?",
        "expected_sources": ["langchain/structured-output.mdx"],
        "reference_answer": "You set a response_format schema (JSON schema, Pydantic model, or dataclass) on create_agent; the model's structured output is captured, validated, and returned in the 'structured_response' key of the agent's state.",
    },
    {
        "question": "How do I build a semantic search engine over documents in LangChain?",
        "expected_sources": ["langchain/knowledge-base.mdx"],
        "reference_answer": "Create Document objects from a source like a PDF, generate embeddings, load and split the document, then index the chunks in a vector store to query by similarity.",
    },
    {
        "question": "What is short-term memory in a LangChain agent?",
        "expected_sources": ["langchain/short-term-memory.mdx"],
        "reference_answer": "Short-term memory lets an application remember previous interactions within a single thread or conversation, most commonly as conversation history.",
    },
    {
        "question": "What is long-term memory used for in LangChain agents?",
        "expected_sources": ["langchain/long-term-memory.mdx"],
        "reference_answer": "Long-term memory lets agents store and recall data across different conversations and sessions, not just within a single thread.",
    },
    {
        "question": "What decision types can a human choose from when a LangChain agent's tool call is interrupted?",
        "expected_sources": ["langchain/human-in-the-loop.mdx"],
        "reference_answer": "A human can approve the action as-is, edit it before running, reject it with feedback, or respond directly for ask-user style tools.",
    },
    {
        "question": "How do LangChain agents use tools from an MCP server?",
        "expected_sources": ["langchain/mcp.mdx"],
        "reference_answer": "LangChain agents can use tools defined on Model Context Protocol (MCP) servers via the langchain-mcp-adapters library.",
    },
    {
        "question": "What are guardrails used for in LangChain agents?",
        "expected_sources": ["langchain/guardrails.mdx"],
        "reference_answer": "Guardrails validate and filter content at key points in an agent's execution to prevent PII leakage, block prompt injection, block harmful content, and enforce business rules.",
    },
    {
        "question": "How can I get visibility into how my LangChain agent behaves in production?",
        "expected_sources": ["langchain/observability.mdx"],
        "reference_answer": "Agents built with create_agent automatically support tracing through LangSmith, which records traces of every step of execution including tool calls and model interactions.",
    },
    {
        "question": "What's the first step in the LangChain quickstart before creating an agent?",
        "expected_sources": ["langchain/quickstart.mdx"],
        "reference_answer": "Install the required LangChain packages before creating the agent.",
    },
    {
        "question": "Why would I use a multi-agent system instead of a single agent in LangChain?",
        "expected_sources": ["langchain/multi-agent/index.mdx"],
        "reference_answer": "Multi-agent systems help with context management, distributed development, and parallelization, useful when a single agent has too many tools or needs specialized, isolated context.",
    },
]
