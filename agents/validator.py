# agents/validator.py
"""
Validator Agent — Third and final node in the LangGraph workflow.

Responsibilities:
- Check recommendations match seller's actual product
- Reject if suggesting different product or material
- Reject if prices are unrealistic for Indian market
- Reject if keywords don't match seller's product
- Uses retry_count as single source of truth
- Hard stop at retry_count >= 2
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from agents.prompts import VALIDATOR_SYSTEM_PROMPT
from dotenv import load_dotenv
import os
import logging
logger = logging.getLogger(__name__)

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

        Uses retry_count as single source of truth.
        No separate validator_retries counter.
        If retry_count >= 2 after validator rejection,
        accept with warning rather than looping indefinitely.
        """
        research = state.get("research_output", "")
        seller = state["seller_input"]
        retry_count = state.get("retry_count", 0)

        # Hard stop — if already retried twice, accept
        # best available rather than looping indefinitely
        if retry_count >= 2:
            return {
                **state,
                "is_validated": True,
                "validator_feedback": (
                    "WARNING: recommendations may not be fully "
                    "relevant to your specific product. "
                    "Please review carefully."
                )
            }

        validation_prompt = f"""
            SELLER'S ACTUAL PRODUCT:
            Product Name: {seller['product_name']}
            Category: {seller['category']}
            Product Details: {seller['product_details']}
            Current Price: ₹{seller['current_price']}
            Platform: {seller['platform']}

            RECOMMENDATIONS TO VALIDATE:
            {research}

            Check if these recommendations are impossible
            for THIS specific seller to implement.
            """

        evaluation = await self.llm.ainvoke([
            SystemMessage(content=VALIDATOR_SYSTEM_PROMPT),
            HumanMessage(content=validation_prompt)
        ])

        verdict = evaluation.content.strip()
        logger.info(f"Validator verdict: {verdict}")
        is_validated = verdict.upper().startswith("VALIDATED")

        if is_validated:
            return {
                **state,
                "is_validated": True,
                "validator_feedback": verdict
            }

        # Rejected — increment retry_count and loop back
        # to Researcher with specific feedback
        return {
            **state,
            "is_validated": False,
            "validator_feedback": verdict,
            "critic_feedback": (
                "RETRY: " + verdict +
                ". Ensure recommendations match seller's "
                "actual product: " + seller["product_details"]
            ),
            "is_verified": False,
            "retry_count": retry_count + 1,
            "critic_trace": list(state.get("critic_trace", []))
        }