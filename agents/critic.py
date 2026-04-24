# agents/critic.py
"""
Critic Agent — Second node in the LangGraph workflow.

Responsibilities:
- Evaluate research quality from Researcher Agent
- Approve if quality standards are met
- Reject with specific feedback if not
- Track retry history in critic_trace
- Hard stop at 3 retries
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from agents.prompts import CRITIC_SYSTEM_PROMPT
from dotenv import load_dotenv
import os

load_dotenv()

class CriticAgent:
    """
    OOP Agent that evaluates research quality.
    Single responsibility: evaluate only, never search.
    """

    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1,
            api_key=os.getenv("OPENAI_API_KEY")
        )

    async def __call__(self, state: dict) -> dict:
        research = state.get("research_output", "")
        retry_count = state.get("retry_count", 0)
        critic_trace = list(state.get("critic_trace", []))

        if retry_count >= 3:
            hard_stop_verdict = (
                "APPROVED: max retries reached, "
                "returning best available research"
            )
            if not critic_trace or critic_trace[-1]["verdict"] != hard_stop_verdict:
                critic_trace.append({
                    "attempt": retry_count + 1,
                    "verdict": hard_stop_verdict
                })
            return {
                **state,
                "is_verified": True,
                "critic_trace": critic_trace,
                "critic_feedback": ""
            }

        evaluation = await self.llm.ainvoke([
            SystemMessage(content=CRITIC_SYSTEM_PROMPT),
            HumanMessage(
                content=f"Evaluate this research report:\n\n{research}"
            )
        ])

        verdict = evaluation.content.strip()
        is_approved = verdict.upper().startswith("APPROVED")

        critic_trace.append({
            "attempt": retry_count + 1,
            "verdict": verdict
        })

        if is_approved:
            return {
                **state,
                "is_verified": True,
                "critic_feedback": verdict,
                "critic_trace": critic_trace,
                "retry_count": retry_count
            }

        return {
            **state,
            "is_verified": False,
            "critic_feedback": verdict,
            "critic_trace": critic_trace,
            "retry_count": retry_count + 1
        }