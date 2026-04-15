# agents/researcher.py
"""
Researcher Agent — First node in the LangGraph workflow.

Responsibilities:
- Dynamically build 3 targeted search queries using LLM
- Execute searches via MCP tools (one tool call per query)
- Extract URLs and format results cleanly
- Validate output structure before returning
- Synthesize into structured diagnosis
- Handle critic feedback on retry

Architecture decisions:
- No hardcoded categories, materials, or product types
- LLM builds queries dynamically from any seller input
- Structured output for query parsing — no brittle string parsing
- All LLM instances created once in __init__
- Output validated before passing to Critic
- Full logging for production visibility
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    ToolMessage
)
from pydantic import BaseModel
from dotenv import load_dotenv
import logging
import os

from agents.prompts import (
    TOOL_USAGE_PROMPT,
    ANALYSIS_SYSTEM_PROMPT,
    QUERY_BUILDER_SYSTEM_PROMPT
)

load_dotenv()

logger = logging.getLogger(__name__)


# ── Structured Output Schema ──────────────────────────────────

class SearchQueries(BaseModel):
    """
    Pydantic model for structured query output.

    Why structured output instead of string parsing?
    - Eliminates brittle string parsing entirely
    - LangChain forces GPT-4o to return validated object
    - No fallback needed for malformed output
    - Type safe — query1/2/3 always strings
    """
    query1: str
    query2: str
    query3: str


# ── Required Output Sections ──────────────────────────────────

REQUIRED_OUTPUT_SECTIONS = [
    "DIAGNOSIS:",
    "TITLE ISSUES:",
    "DESCRIPTION ISSUES:",
    "PRICING ISSUES:",
    "RECOMMENDATIONS:",
    "RECOMMENDED TITLE:",
    "RECOMMENDED DESCRIPTION:",
    "RECOMMENDED PRICE:",
    "KEYWORDS:",
    "DATA QUALITY:",
    "COMPETITOR LISTINGS FOUND:",
]


class ResearcherAgent:
    """
    OOP Agent wrapping GPT-4o with MCP search tools.
    Single responsibility: search and synthesize only.

    LLM instances (created once in __init__):
    - self.llm: synthesis (temperature 0.0, no tools)
    - self.query_llm: query building (temperature 0.1, no tools)
    - self.query_llm_structured: query building with structured output
    - self.llm_with_tools: search execution (temperature 0.0, with tools)
    """

    def __init__(self, tools: list):
        # Synthesis LLM — temperature 0.0 for consistent
        # structured output that Critic can evaluate reliably
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.0,
            api_key=os.getenv("OPENAI_API_KEY")
        )

        # Query building LLM — temperature 0.1 for slight
        # variation across retries while staying focused
        self.query_llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1,
            api_key=os.getenv("OPENAI_API_KEY")
        )

        # Structured output version of query LLM
        # Forces output to match SearchQueries Pydantic model
        # Eliminates all string parsing
        self.query_llm_structured = (
            self.query_llm.with_structured_output(SearchQueries)
        )

        # Search execution LLM — temperature 0.0 for
        # consistent tool call behavior
        self.llm_with_tools = ChatOpenAI(
            model="gpt-4o",
            temperature=0.0,
            api_key=os.getenv("OPENAI_API_KEY")
        ).bind_tools(tools)

        self.tools = {tool.name: tool for tool in tools}

        logger.info(
            f"ResearcherAgent initialized with "
            f"{len(tools)} MCP tools: "
            f"{list(self.tools.keys())}"
        )

    # ── Output Validation ─────────────────────────────────────

    def _validate_output(self, content: str) -> bool:
        """
        Validates LLM synthesis output contains all
        required sections before passing to Critic.

        Why validate here?
        If critical sections are missing, downstream
        _parse_research_output in workflow.py silently
        returns empty strings — no error, no retry signal.
        Validating here catches malformed output early
        and triggers a proper retry with clear signal.
        """
        content_upper = content.upper()
        missing = [
            section for section in REQUIRED_OUTPUT_SECTIONS
            if section not in content_upper
        ]

        if missing:
            logger.warning(
                f"Output validation failed. "
                f"Missing sections: {missing}"
            )
            return False

        return True

    # ── Query Builder ─────────────────────────────────────────

    async def _build_queries(self, state: dict) -> list[str]:
        """
        Uses LLM with structured output to build 3 queries.

        Why structured output?
        Eliminates brittle string parsing entirely.
        LangChain forces GPT-4o to return a validated
        SearchQueries Pydantic object — no parsing needed.

        Why async?
        Makes an LLM network call — must be async to
        avoid blocking FastAPI's event loop.

        Fallback behavior:
        If structured output fails, falls back to
        dynamic queries built from seller's actual input.
        Never crashes, never returns hardcoded category data.
        """
        p = state["seller_input"]
        price = int(p["current_price"])
        lower = max(99, price - 150)
        upper = price + 250
        feedback = state.get("critic_feedback", "")
        retry_count = state.get("retry_count", 0)

        retry_instruction = (
            "Previous research was rejected.\n"
            "Critic feedback: " + feedback + "\n"
            "Adjust ALL 3 queries to address "
            "this feedback specifically."
        ) if retry_count > 0 and feedback else ""

        query_request = f"""
Seller's product details:
Product Name: {p['product_name']}
Category: {p['category']}
Product Details: {p['product_details']}
Current Price: ₹{p['current_price']}
Target Audience: {p['target_audience']}
Platform: {p['platform']}

Generate exactly 3 Amazon India search queries.

Query 1: Find competitor listings with star ratings
Query 2: Find products in price range {lower} to {upper} rupees
Query 3: Find bestseller title patterns for this product

{retry_instruction}
"""

        try:
            result = await self.query_llm_structured.ainvoke([
                SystemMessage(
                    content=QUERY_BUILDER_SYSTEM_PROMPT
                ),
                HumanMessage(content=query_request)
            ])

            queries = [result.query1, result.query2, result.query3]
            logger.info(f"Built queries: {queries}")
            return queries

        except Exception as e:
            # Structured output failed — dynamic fallback
            # Built from seller's actual input, never hardcoded
            logger.warning(
                f"Structured query building failed: {e}. "
                f"Using dynamic fallback."
            )
            product = p["product_name"]
            return [
                f"amazon.in {product} top rated reviews india",
                f"amazon.in {product} india {lower} to {upper} rupees",
                f"amazon.in best selling {product} india 2026"
            ]

    # ── URL and Result Handling ───────────────────────────────

    def _extract_urls(self, result) -> list[str]:
        """
        Extracts clean amazon.in URLs from Tavily response.

        Tavily structure:
        [{'type': 'text', 'text': '...URL: https://...'}]

        URLs are inside 'text' field on lines starting
        with 'URL:' — not as separate dictionary keys.
        """
        urls = []

        try:
            if isinstance(result, list):
                for item in result:
                    if isinstance(item, dict):
                        text = item.get("text", "")
                        for line in text.split("\n"):
                            line = line.strip()
                            if line.startswith("URL:"):
                                url = line.replace(
                                    "URL:", ""
                                ).strip()
                                if "amazon.in" in url:
                                    urls.append(url)
        except Exception as e:
            # Never crash on URL extraction
            # Sources are informational not critical
            logger.warning(f"URL extraction failed: {e}")

        return urls

    def _format_result(self, result) -> str:
        """
        Formats Tavily result into clean string for LLM.

        Extracts from 'text' field — where Tavily puts
        all content including titles, URLs, prices, ratings.

        2000 char limit:
        - Controls token cost
        - Keeps context focused
        - Reduces latency
        - First 2000 chars have highest signal density
        """
        try:
            if isinstance(result, list):
                for item in result:
                    if isinstance(item, dict):
                        text = item.get("text", "")
                        if text and len(text) > 50:
                            return text[:2000]

            return str(result)[:2000]

        except Exception as e:
            logger.warning(f"Result formatting failed: {e}")
            return str(result)[:2000]

    # ── Search Execution ──────────────────────────────────────

    async def _execute_searches(
        self, queries: list[str]
    ) -> tuple[list, list[str]]:
        """
        Executes all searches via MCP tools.

        OpenAI conversation format maintained:
        SystemMessage → HumanMessage → AIMessage(tool_calls)
        → ToolMessage(s) → HumanMessage(synthesize)

        Single tool call constraint enforced via prompt.
        All tool_call_ids responded to — satisfies OpenAI
        requirement that every tool call has a response.
        """
        all_message_pairs = []
        all_sources = []

        # Combined system prompt — tool usage rules first
        # then analysis instructions
        # Tool usage at top = higher compliance
        combined_system = (
            TOOL_USAGE_PROMPT + "\n\n" + ANALYSIS_SYSTEM_PROMPT
        )

        for query in queries:
            logger.info(f"Executing search: {query}")

            try:
                response = await self.llm_with_tools.ainvoke([
                    SystemMessage(content=combined_system),
                    HumanMessage(content=(
                        f"Use tavily_search tool ONCE "
                        f"to search for:\n{query}\n\n"
                        f"IMPORTANT:\n"
                        f"- Call tavily_search exactly ONE time\n"
                        f"- Use the query exactly as given\n"
                        f"- Do not call any other tools\n"
                        f"- Do not make multiple tool calls"
                    ))
                ])

                if response.tool_calls:
                    tool_messages_for_this_response = []

                    for tool_call in response.tool_calls:
                        tool_name = tool_call["name"]
                        tool_args = tool_call["args"]

                        if tool_name in self.tools:
                            result = await self.tools[
                                tool_name
                            ].ainvoke(tool_args)

                            urls = self._extract_urls(result)
                            all_sources.extend(urls)
                            logger.info(
                                f"Search complete. "
                                f"URLs found: {len(urls)}"
                            )

                            result_str = self._format_result(
                                result
                            )

                            tool_messages_for_this_response.append(
                                ToolMessage(
                                    content=result_str,
                                    tool_call_id=tool_call["id"]
                                )
                            )
                        else:
                            # Unknown tool — must still respond
                            # Never leave tool_call without response
                            logger.warning(
                                f"Unknown tool called: {tool_name}"
                            )
                            tool_messages_for_this_response.append(
                                ToolMessage(
                                    content=(
                                        "Tool not available. "
                                        "Skipping this search."
                                    ),
                                    tool_call_id=tool_call["id"]
                                )
                            )

                    all_message_pairs.append((
                        response,
                        tool_messages_for_this_response
                    ))

            except Exception as e:
                logger.error(
                    f"Search failed for query '{query}': {e}"
                )
                # Continue to next query — don't crash
                # Partial results better than no results

        return all_message_pairs, all_sources

    # ── Main Node ─────────────────────────────────────────────

    async def __call__(self, state: dict) -> dict:
        """
        LangGraph node interface.
        Receives full state, returns updated state.

        Flow:
        1. Build 3 dynamic queries via LLM
        2. Execute 3 searches via MCP
        3. Synthesize results into structured diagnosis
        4. Validate output structure
        5. Return updated state

        If synthesis output is malformed → force retry
        by returning INSUFFICIENT DATA signal.
        """
        seller = state["seller_input"]
        attempt = state.get("retry_count", 0) + 1
        logger.info(
            f"Researcher starting — attempt {attempt}"
        )

        # Step 1 — Build queries dynamically via LLM
        queries = await self._build_queries(state)

        # Step 2 — Execute searches via MCP
        message_pairs, sources = await self._execute_searches(
            queries
        )

        # Step 3 — Build synthesis message chain
        # Correct OpenAI format:
        # System → Human → AI(tool_calls) → Tool(s) → Human
        combined_system = (
            TOOL_USAGE_PROMPT + "\n\n" + ANALYSIS_SYSTEM_PROMPT
        )

        synthesis_messages = [
            SystemMessage(content=combined_system),
            HumanMessage(
                content=(
                    "Analyze these Amazon India search results:"
                )
            )
        ]

        # AIMessage MUST precede its ToolMessages
        for ai_message, tool_messages in message_pairs:
            synthesis_messages.append(ai_message)
            for tool_message in tool_messages:
                synthesis_messages.append(tool_message)

        # Build critic feedback instruction separately
        # to avoid nested f-string syntax error
        critic_instruction = (
            "PREVIOUS ATTEMPT REJECTED — "
            "ADDRESS THIS SPECIFICALLY:\n"
            + state["critic_feedback"]
        ) if state.get("critic_feedback") else ""

        synthesis_prompt = f"""
SELLER'S CURRENT LISTING TO DIAGNOSE:
Product: {seller['product_name']}
Category: {seller['category']}
Details: {seller['product_details']}
Current Price: ₹{seller['current_price']}
Platform: {seller['platform']}
Target Audience: {seller['target_audience']}
Current Title: {seller['current_title']}
Current Description: {seller['current_description']}

STEP 1 — Extract all data points from search results:
List every ₹ price found.
List every star rating found.
Count valid finished product listings only.

STEP 2 — Compare seller's listing to competitors:
What do top competitor titles have that seller's lacks?
What do top descriptions mention that seller's doesn't?
How does seller's price compare to the range found?

STEP 3 — Write diagnosis and recommendations:
Be specific. Use actual competitor examples.
Never write vague statements.
Instead write: "Competitors like [name] use fabric type,
fit type, and occasion — your title has none of these."
Follow the exact output format specified.

{critic_instruction}
"""

        synthesis_messages.append(
            HumanMessage(content=synthesis_prompt)
        )

        # Step 4 — Synthesize with plain LLM
        # No tools bound — prevents more searches during synthesis
        final_response = await self.llm.ainvoke(
            synthesis_messages
        )

        # Step 5 — Validate output structure
        if not self._validate_output(final_response.content):
            logger.warning(
                "Synthesis output malformed — forcing retry"
            )
            return {
                **state,
                "research_output": (
                    "INSUFFICIENT DATA: synthesis output "
                    "missing required sections. "
                    "Please retry with more specific searches."
                ),
                "sources": list(set(sources)),
                "is_verified": False,
                "retry_count": state.get("retry_count", 0)
            }

        logger.info(
            f"Researcher complete — "
            f"output length: {len(final_response.content)} chars, "
            f"sources: {len(sources)}"
        )

        return {
            **state,
            "research_output": final_response.content,
            "sources": list(set(sources)),
            "is_verified": False,
            "retry_count": state.get("retry_count", 0)
        }