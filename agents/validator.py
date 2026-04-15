# agents/validator.py
"""
Validator Agent — Third and final node in the LangGraph workflow.

Responsibilities:
- Check recommendations match seller's actual product
- Reject if suggesting different product or material
- Reject if prices are unrealistic for Indian market
- Reject if keywords don't match seller's product
- 1 retry only — loops back to Researcher with specific feedback
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from agents.prompts import VALIDATOR_SYSTEM_PROMPT
from dotenv import load_dotenv
import os

load_dotenv()

class ValidatorAgent:
    """
    OOP Agent that checks recommendation relevance.
    Single responsibility: relevance check only.
    Never searches, never evaluates research quality.
    """

    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1,
            api_key=os.getenv("OPENAI_API_KEY")
        )

    async def __call__(self, state: dict) -> dict:
        """
        LangGraph node interface.
        Checks relevance of recommendations to seller's product.
        """
        research = state.get("research_output", "")
        seller = state["seller_input"]
        validator_retries = state.get("validator_retries", 0)

        # Hard stop — validator gets 1 retry only
        # After that return best available with warning
        if validator_retries >= 1:
            return {
                **state,
                "is_validated": True,
                "validator_feedback": (
                    "WARNING: recommendations may not be fully "
                    "relevant to your specific product. "
                    "Please review carefully."
                )
            }

        # Build validation prompt with seller context
        # so LLM can compare recommendations against
        # seller's actual product
        validation_prompt = f"""
SELLER'S ACTUAL PRODUCT:
Product Name: {seller['product_name']}
Category: {seller['category']}
Product Details: {seller['product_details']}
Current Price: ₹{seller['current_price']}
Platform: {seller['platform']}

RECOMMENDATIONS TO VALIDATE:
{research}

Check if these recommendations are relevant and implementable
for THIS specific seller's product.
"""

        evaluation = await self.llm.ainvoke([
            SystemMessage(content=VALIDATOR_SYSTEM_PROMPT),
            HumanMessage(content=validation_prompt)
        ])

        verdict = evaluation.content.strip()
        is_validated = verdict.upper().startswith("VALIDATED")

        if is_validated:
            return {
                **state,
                "is_validated": True,
                "validator_feedback": verdict
            }

        # Not validated — send back to Researcher
        # with validator's specific feedback
        # increment validator_retries so we don't
        # loop more than once
        return {
            **state,
            "is_validated": False,
            "validator_feedback": verdict,
            "validator_retries": validator_retries + 1,
            # Inject validator feedback into critic_feedback
            # so Researcher picks it up in _build_queries
            "critic_feedback": (
                f"RETRY: {verdict}. "
                f"Ensure recommendations match seller's "
                f"actual product: {seller['product_details']}"
            ),
            # Reset verification so workflow routes
            # back to Researcher correctly
            "is_verified": False,
            "retry_count": state.get("retry_count", 0)
        }