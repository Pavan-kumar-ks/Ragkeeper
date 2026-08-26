from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

JUDGE_SYSTEM_PROMPT = """You are an impartial evaluator of a RAG system's answer. Score on a 1-5 scale:
- faithfulness: is every claim in the answer supported by the retrieved context (5), or does it contain \
unsupported/fabricated claims (1)?
- answer_relevance: does the answer directly address the question (5), or is it off-topic/evasive (1)?
- correctness: how well does the answer match the reference answer's meaning (5 = equivalent, 1 = contradicts it)?
Be strict and concise in your justification."""

JUDGE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", JUDGE_SYSTEM_PROMPT),
        (
            "human",
            "Question: {question}\n\nRetrieved context:\n{context}\n\n"
            "Generated answer: {answer}\n\nReference answer: {reference_answer}",
        ),
    ]
)


class JudgeScores(BaseModel):
    faithfulness: int = Field(ge=1, le=5)
    answer_relevance: int = Field(ge=1, le=5)
    correctness: int = Field(ge=1, le=5)
    justification: str


def build_judge(model: str, api_key) -> ChatGroq:
    return ChatGroq(model=model, temperature=0, api_key=api_key).with_structured_output(JudgeScores)


def judge_answer(judge_llm, question: str, context: str, answer: str, reference_answer: str) -> JudgeScores:
    messages = JUDGE_PROMPT.format_messages(
        question=question, context=context, answer=answer, reference_answer=reference_answer
    )
    return judge_llm.invoke(messages)
