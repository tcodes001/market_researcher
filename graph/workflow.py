# graph/workflow.py
"""
LangGraph Workflow — connects all three agents into a
stateful, cyclic multi-agent graph.
"""

from langgraph.graph import StateGraph, END
from typing import TypedDict

from agents.researcher import ResearcherAgent
from agents.critic import CriticAgent
from agents.validator import ValidatorAgent

import logging
logger = logging.getLogger(__name__)

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
        Parses structured text output from Researcher.

        Handles both single-line and multi-line field values.
        LLM sometimes puts values on same line as label,
        sometimes on subsequent lines with bullet points.
        This parser handles both formats correctly.

        Approach:
        - Track which field we are currently reading
        - Accumulate all lines belonging to that field
        - Save accumulated content when next label found
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

        # Maps label → (which dict, which key)
        field_labels = {
            "TITLE ISSUES:": ("diagnosis", "title_issues"),
            "DESCRIPTION ISSUES:": ("diagnosis", "description_issues"),
            "PRICING ISSUES:": ("diagnosis", "pricing_issues"),
            "RECOMMENDED TITLE:": ("recommendations", "recommended_title"),
            "RECOMMENDED DESCRIPTION:": ("recommendations", "recommended_description"),
            "RECOMMENDED PRICE:": ("recommendations", "recommended_price"),
            "KEYWORDS:": ("recommendations", "keywords"),
        }

        # Section headers that signal end of a field
        section_headers = {
            "DIAGNOSIS:", "RECOMMENDATIONS:", "SOURCES:",
            "DATA QUALITY:", "IMPORTANT:", "CRITICAL:",
            "COMPETITOR LISTINGS FOUND:", "PRICES FOUND:",
            "RATINGS FOUND:", "REVIEWS FOUND:",
            "COMPETITOR NAMES FOUND:"
        }

        current_dict_name = None
        current_key = None
        accumulated_lines = []

        def save_current_field():
            """Save accumulated lines to the correct dict."""
            if current_dict_name is None or current_key is None:
                return
            if not accumulated_lines:
                return

            value = " ".join(accumulated_lines).strip()

            if current_key == "keywords":
                # Clean keywords — remove parentheses and citations
                raw_keywords = value.split(",")
                clean_keywords = []
                for kw in raw_keywords:
                    # Strip everything after "(" — removes citations
                    kw = kw.split("(")[0].strip()
                    # Strip bullet points and dashes
                    kw = kw.lstrip("- •*").strip()
                    if kw and len(kw) > 1:
                        clean_keywords.append(kw)
                recommendations["keywords"] = clean_keywords

            elif current_dict_name == "diagnosis":
                # Clean bullet points from diagnosis fields
                cleaned = value.replace("- ", " ").replace("• ", " ")
                diagnosis[current_key] = cleaned.strip()

            elif current_dict_name == "recommendations":
                recommendations[current_key] = value

        for line in research.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue

            upper = stripped.upper()

            # Check if this line is a field label
            matched_label = False
            for label, (dict_name, key) in field_labels.items():
                if upper.startswith(label):
                    # Save what we were accumulating
                    save_current_field()

                    # Start new field
                    current_dict_name = dict_name
                    current_key = key
                    accumulated_lines = []

                    # Get value on same line as label if any
                    remainder = stripped[len(label):].strip()
                    if remainder:
                        accumulated_lines.append(remainder)

                    matched_label = True
                    break

            if matched_label:
                continue

            # Check if this is a section header
            is_section = any(
                upper.startswith(h) for h in section_headers
            )
            if is_section:
                save_current_field()
                current_dict_name = None
                current_key = None
                accumulated_lines = []
                continue

            # Accumulate continuation lines for current field
            if current_dict_name is not None:
                accumulated_lines.append(stripped)

        # Save the last field
        save_current_field()

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
            "validator_feedback": ""
        }

        # Run the graph
        final_state = await self.graph.ainvoke(initial_state)
        # TEMPORARY DEBUG
        logger.info(
            f"RAW OUTPUT:\n"
            f"{final_state.get('research_output', 'EMPTY')[:1000]}"
        )
        
        # Parse research output into structured format
        diagnosis, recommendations = self._parse_research_output(
            final_state["research_output"]
        )

        # Build and return final clean output
        return {
            "seller_input": {
                "product_name": seller_input["product_name"],
                "category": seller_input["category"],
                "current_price": seller_input["current_price"],
                "platform": seller_input["platform"],
                "current_title": seller_input["current_title"],
                "current_description": seller_input["current_description"]
            },
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