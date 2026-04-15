# graph/workflow.py
"""
LangGraph Workflow — connects all three agents into a
stateful, cyclic multi-agent graph.

Flow:
Researcher → Critic → Validator → END
     ↑           |         |
     |←←←←←←←←←←|         |
     |←←←←←←←←←←←←←←←←←←←|
"""

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List, Any
from langchain_core.messages import BaseMessage
import operator

from agents.researcher import ResearcherAgent
from agents.critic import CriticAgent
from agents.validator import ValidatorAgent


# ── State Schema ──────────────────────────────────────────────
class AgentState(TypedDict):
    # Seller's original input — set once, never changed
    seller_input: dict

    # Researcher output
    research_output: str
    sources: list

    # Critic fields
    is_verified: bool
    critic_feedback: str
    critic_trace: list
    retry_count: int

    # Validator fields
    is_validated: bool
    validator_feedback: str
    validator_retries: int


# ── Routing Functions ─────────────────────────────────────────
def route_after_critic(state: AgentState) -> str:
    """
    Conditional edge after Critic.
    Routes to Validator if approved.
    Routes back to Researcher if rejected.
    """
    if state.get("is_verified", False):
        return "validator"
    return "researcher"


def route_after_validator(state: AgentState) -> str:
    """
    Conditional edge after Validator.
    Routes to END if validated.
    Routes back to Researcher if rejected.
    """
    if state.get("is_validated", False):
        return "end"
    return "researcher"


# ── Workflow Class ────────────────────────────────────────────
class MarketResearchWorkflow:
    """
    Builds and manages the LangGraph StateGraph.
    Initialized once at FastAPI startup.
    run() called for every seller request.
    """

    def __init__(self):
        self.graph = None
        self.critic = CriticAgent()
        self.validator = ValidatorAgent()

    async def initialize_tools(self, tools: list):
        """
        Called at FastAPI startup after MCP tools are loaded.
        Builds the graph with tools injected into Researcher.
        """
        researcher = ResearcherAgent(tools=tools)
        self.graph = self._build_graph(
            researcher,
            self.critic,
            self.validator
        )
        print("✅ LangGraph workflow compiled successfully.")

    def _build_graph(
        self,
        researcher: ResearcherAgent,
        critic: CriticAgent,
        validator: ValidatorAgent
    ) -> StateGraph:
        """
        Constructs the StateGraph with nodes,
        edges and conditional routing.
        """
        workflow = StateGraph(AgentState)

        # Register nodes
        workflow.add_node("researcher", researcher)
        workflow.add_node("critic", critic)
        workflow.add_node("validator", validator)

        # Entry point
        workflow.set_entry_point("researcher")

        # Normal edge — Researcher always goes to Critic
        workflow.add_edge("researcher", "critic")

        # Conditional edge after Critic
        # → Validator if approved
        # → Researcher if rejected
        workflow.add_conditional_edges(
            "critic",
            route_after_critic,
            {
                "validator": "validator",
                "researcher": "researcher"
            }
        )

        # Conditional edge after Validator
        # → END if validated
        # → Researcher if rejected
        workflow.add_conditional_edges(
            "validator",
            route_after_validator,
            {
                "end": END,
                "researcher": "researcher"
            }
        )

        return workflow.compile()

    def _parse_research_output(
        self, research: str
    ) -> tuple[dict, dict]:
        """
        Parses the structured text output from Researcher
        into separate diagnosis and recommendations dicts.

        Why parse here and not in Researcher?
        Researcher's job is to produce research.
        Formatting for the API response is workflow's job.
        Separation of concerns.
        """
        diagnosis = {
            "title_issues": "",
            "description_issues": "",
            "pricing_issues": ""
        }
        recommendations = {
            "recommended_title": "",
            "recommended_description": "",
            "recommended_price": "",
            "keywords": []
        }

        lines = research.split("\n")
        for line in lines:
            line = line.strip()

            # Parse diagnosis fields
            if line.upper().startswith("TITLE ISSUES:"):
                diagnosis["title_issues"] = line.split(":", 1)[1].strip()
            elif line.upper().startswith("DESCRIPTION ISSUES:"):
                diagnosis["description_issues"] = line.split(":", 1)[1].strip()
            elif line.upper().startswith("PRICING ISSUES:"):
                diagnosis["pricing_issues"] = line.split(":", 1)[1].strip()

            # Parse recommendation fields
            elif line.upper().startswith("RECOMMENDED TITLE:"):
                recommendations["recommended_title"] = line.split(":", 1)[1].strip()
            elif line.upper().startswith("RECOMMENDED DESCRIPTION:"):
                recommendations["recommended_description"] = line.split(":", 1)[1].strip()
            elif line.upper().startswith("RECOMMENDED PRICE:"):
                recommendations["recommended_price"] = line.split(":", 1)[1].strip()
            elif line.upper().startswith("KEYWORDS:"):
                keywords_str = line.split(":", 1)[1].strip()
                recommendations["keywords"] = [
                    k.strip() for k in keywords_str.split(",")
                ]

        return diagnosis, recommendations

    async def run(self, seller_input: dict) -> dict:
        """
        Executes the full multi-agent workflow.
        Called by app.py for every seller request.
        """
        if self.graph is None:
            raise RuntimeError(
                "Workflow not initialized. "
                "Call initialize_tools() first."
            )

        # Build initial state
        initial_state = {
            "seller_input": seller_input,
            "research_output": "",
            "sources": [],
            "is_verified": False,
            "critic_feedback": "",
            "critic_trace": [],
            "retry_count": 0,
            "is_validated": False,
            "validator_feedback": "",
            "validator_retries": 0
        }

        # Run the graph
        final_state = await self.graph.ainvoke(initial_state)

        # Parse research output into structured format
        diagnosis, recommendations = self._parse_research_output(
            final_state["research_output"]
        )

        # Build and return final clean output
        return {
            "diagnosis": diagnosis,
            "recommendations": recommendations,
            "critic_trace": final_state.get("critic_trace", []),
            "sources": final_state.get("sources", []),
            "validator_feedback": final_state.get(
                "validator_feedback", ""
            ),
            "verified": final_state.get("is_verified", False),
            "validated": final_state.get("is_validated", False),
            "attempts": final_state.get("retry_count", 0) + 1
        }