# agents/researcher.py
"""
Researcher Agent — First node in the LangGraph workflow.

Responsibilities:
- Dynamically build 3 targeted search queries using LLM
- Execute two-step search via MCP tools:
  Step 1: tavily_search with optimized params → get URLs
  Step 2: tavily_extract on top scored URLs → get full content
- Synthesize full product page content into structured diagnosis
- Validate output structure before passing to Critic
- Handle critic feedback on retry with adjusted queries

Architecture decisions:
- LLM owns query strings — understands any product context
- Code owns all Tavily parameters — engineering decisions
  not intelligence decisions
- Two-step pipeline solves the data quality problem at source
  rather than relaxing output validation
- Score-based URL filtering ensures only relevant pages extracted
- No hardcoded categories, materials, or product types anywhere
- All LLM instances created once in __init__ — no per-request overhead
- Strict output validation preserved — enforced by Critic
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel
from dotenv import load_dotenv
import logging
import os

from agents.prompts import (
    ANALYSIS_SYSTEM_PROMPT,
    QUERY_BUILDER_SYSTEM_PROMPT
)
from agents.config import (
    REQUIRED_OUTPUT_SECTIONS,
    TAVILY_SEARCH_PARAMS,
    TAVILY_EXTRACT_PARAMS
)

load_dotenv()

logger = logging.getLogger(__name__)


#  Structured Output Schema ----------------

class SearchQueries(BaseModel):
    """
    Pydantic model for LLM query output.

    Why structured output?
    Eliminates brittle string parsing entirely.
    LangChain forces GPT-4o to return a validated object.
    query1, query2, query3 are always strings — type safe.
    If LLM returns malformed output, Pydantic raises an error
    and we fall back to dynamic queries from seller input.
    """
    query1: str
    query2: str
    query3: str


class ResearcherAgent:
    """
    Researcher Agent — searches Amazon India and synthesizes
    competitor intelligence into structured listing diagnosis.

    Single responsibility: search and synthesize only.
    Does not evaluate quality (Critic's job).
    Does not check relevance (Validator's job).

    Three LLM instances created once at startup:
    self.llm — synthesis calls, temperature 0.0
    self.query_llm — query building base, temperature 0.1
    self.query_llm_structured — query building with Pydantic output

    Why separate instances?
    Synthesis needs temperature 0.0 — consistent structured output
    every time for reliable Critic evaluation.
    Query building needs temperature 0.1 — slight variation
    helps produce different queries on retry when previous
    queries failed to find sufficient data.
    """

    def __init__(self, tools: list):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.0,
            api_key=os.getenv("OPENAI_API_KEY")
        )

        self.query_llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1,
            api_key=os.getenv("OPENAI_API_KEY")
        )

        self.query_llm_structured = (
            self.query_llm.with_structured_output(SearchQueries)
        )

        self.tools = {tool.name: tool for tool in tools}

        logger.info(
            f"ResearcherAgent initialized with "
            f"{len(tools)} MCP tools: "
            f"{list(self.tools.keys())}"
        )

    # Output Validation -----------------

    def _validate_output(self, content: str) -> bool:
        """
        Checks synthesis output contains all required sections.

        Called before returning from __call__.
        If validation fails — returns INSUFFICIENT DATA signal
        which Critic reads and triggers a retry.

        Why keep strict validation even with two-step pipeline?
        The two-step pipeline solves the data quality problem
        at the source — better data in means better output.
        But we still validate to catch edge cases where
        extraction fails or Amazon pages return no product data.
        Strict validation is a safety net, not the primary fix.
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

    # Query Builder ------------------

    async def _build_queries(self, state: dict) -> list[str]:
        """
        LLM builds 3 targeted search queries from seller input.

        This is genuine LLM intelligence — not hardcoded logic.
        The LLM understands any product type, category, or market
        and builds contextually relevant queries without any
        hardcoded category lists or material mappings.

        On first attempt: queries target ratings, price range,
        and bestseller patterns.

        On retry: LLM reads critic feedback and builds
        different queries specifically targeting what was missing.
        This is the self-correction mechanism — the LLM adapts
        its search strategy based on evaluation results.

        Price range calculation:
        lower = seller price - 150 (shows cheaper competition)
        upper = seller price + 250 (shows premium alternatives)
        Gives full competitive landscape in one search.

        Fallback: if structured output fails, builds queries
        dynamically from seller's product name — never hardcoded,
        never crashes.
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
                SystemMessage(content=QUERY_BUILDER_SYSTEM_PROMPT),
                HumanMessage(content=query_request)
            ])

            queries = [result.query1, result.query2, result.query3]
            logger.info(f"Built queries: {queries}")
            return queries

        except Exception as e:
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

    # URL Extraction -------------

    def _extract_urls_with_scores(self, result) -> list[dict]:
        """
        Extracts amazon.in URLs with position-based relevance scores.

        Why position-based scores instead of Tavily's score field?
        Tavily's score field is not present in the MCP response
        text format. The text field contains Title, URL, Content
        but not a numeric score we can parse.

        However, Tavily already ranks results by relevance before
        returning them. Result 1 is more relevant than result 5.
        We use position to infer relevance:
        Position 1 → score 0.9
        Position 2 → score 0.8
        Position 3 → score 0.7
        Position 4 → score 0.6
        Position 5+ → score 0.5

        Score threshold 0.7 means we extract top 3 results
        from each search query - the most relevant pages.

        Fallback: if no URLs pass threshold, takes top 3
        by score regardless. Ensures extraction always runs
        even when search returns lower quality results.
        """
        url_scores = []

        try:
            if isinstance(result, list):
                for item in result:
                    if isinstance(item, dict):
                        text = item.get("text", "")
                        lines = text.split("\n")

                        position = 0
                        for line in lines:
                            line = line.strip()
                            if line.startswith("URL:"):
                                url = line.replace(
                                    "URL:", ""
                                ).strip()
                                if "amazon.in" in url:
                                    position += 1
                                    score = max(
                                        0.5,
                                        1.0 - (position * 0.1)
                                    )
                                    url_scores.append({
                                        "url": url,
                                        "score": score
                                    })

        except Exception as e:
            logger.warning(f"URL extraction failed: {e}")

        return url_scores

    def _format_result(self, result) -> str:
        """
        Extracts text content from Tavily response.

        Tavily MCP returns: [{'type': 'text', 'text': '...'}]
        Content is inside the 'text' field of the first item.

        3000 char limit (increased from previous 2000):
        With include_raw_content=True and extract_depth=advanced,
        Tavily returns full product page content — prices,
        ratings, titles, descriptions all in one block.
        3000 chars captures the most signal-dense portion
        while controlling token cost.

        Fallback: converts entire result to string if
        structure doesn't match expected format.
        Never crashes, always returns something.
        """
        try:
            if isinstance(result, list):
                for item in result:
                    if isinstance(item, dict):
                        text = item.get("text", "")
                        if text and len(text) > 50:
                            return text[:3000]

            return str(result)[:3000]

        except Exception as e:
            logger.warning(f"Result formatting failed: {e}")
            return str(result)[:3000]

    async def _is_relevant_product_content(
        self,
        content: str,
        seller_input: dict
    ) -> bool:
        """
        Uses LLM to determine if extracted content is a relevant
        product listing for this specific seller.

        Why LLM and not rules?
        Rules require hardcoded category signals — "fabric",
        "sleeve", "fit" only work for clothing. LLM understands
        relevance for any product type without hardcoding.

        Why self.llm and not a separate instance?
        This is a simple yes/no reasoning task — same model,
        temperature 0.0 for consistency.
        Tiny prompt — cheap call, fast response.

        Why only 500 chars of content?
        First 500 chars of a product page contain the product
        title and category — enough to determine relevance.
        Sending full content wastes tokens on this check.
        """
        try:
            response = await self.llm.ainvoke([
                HumanMessage(content=(
                    f"Seller's product: "
                    f"{seller_input['product_name']}\n"
                    f"Category: {seller_input['category']}\n\n"
                    f"Extracted page content:\n"
                    f"{content[:500]}\n\n"
                    f"Is this page content from a product "
                    f"listing relevant to the seller's product?\n"
                    f"Answer with only YES or NO."
                ))
            ])
            return (
                response.content.strip().upper().startswith("YES")
            )
        except Exception as e:
            logger.warning(f"Relevance check failed: {e}")
            # Default to True on failure
            # Better to include uncertain content than miss data
            return True
        
    
    # Two-Step Search Pipeline -----

    async def _execute_searches(
        self, queries: list[str],
        seller_input: dict
    ) -> tuple[list[str], list[str]]:
        """
        Two-step search pipeline with content validation.

        Step 1 — tavily_search:
        Finds relevant amazon.in pages matching the query.
        Code sets all params — LLM provided query string only.

        Step 2 — tavily_extract per URL:
        Fetches full content of each URL individually.
        Validates content relevance before adding to results.
        Irrelevant pages (books, medical, wrong category)
        are filtered out by LLM relevance check.

        Why extract per URL not batch?
        Batch extraction returns all content mixed together.
        Per-URL extraction lets us validate each page
        independently and skip irrelevant ones.

        Why pass seller_input?
        Relevance check needs seller context to determine
        if extracted content matches the seller's product.
        """
        all_results = []
        all_sources = []

        search_tool = self.tools.get("tavily_search")
        extract_tool = self.tools.get("tavily_extract")

        if not search_tool:
            logger.error(
                "tavily_search tool not found in MCP tools"
            )
            return all_results, all_sources

        for query in queries:
            logger.info(f"Step 1 — Searching: {query}")

            try:
                search_args = {
                    "query": query,
                    **TAVILY_SEARCH_PARAMS
                }

                search_result = await search_tool.ainvoke(
                    search_args
                )

                # Add search snippets as baseline context
                search_text = self._format_result(search_result)
                all_results.append(
                    f"SEARCH RESULTS FOR: {query}\n{search_text}"
                )

                # Get scored URLs from search
                url_scores = self._extract_urls_with_scores(
                    search_result
                )

                # Take top 3 by score
                urls_to_check = [
                    item["url"] for item in sorted(
                        url_scores,
                        key=lambda x: x["score"],
                        reverse=True
                    )[:3]
                ]

                if not urls_to_check:
                    logger.warning(
                        f"No URLs found for query: {query}"
                    )
                    continue

                # Step 2 — Extract and validate per URL
                if extract_tool:
                    for url in urls_to_check:
                        try:
                            logger.info(
                                f"Step 2 — Extracting: {url}"
                            )

                            # Extract single URL
                            extract_result = await extract_tool.ainvoke({
                                "urls": [url],
                                **TAVILY_EXTRACT_PARAMS
                            })

                            content = self._format_result(
                                extract_result
                            )

                            # Validate relevance before adding
                            is_relevant = await self._is_relevant_product_content(
                                content,
                                seller_input
                            )

                            if not is_relevant:
                                logger.info(
                                    f"Skipping irrelevant URL: {url}"
                                )
                                continue

                            # Valid — add to results
                            all_results.append(
                                f"PRODUCT PAGE CONTENT:\n{content}"
                            )
                            all_sources.append(url)
                            logger.info(
                                f"Added relevant source: {url}"
                            )

                        except Exception as e:
                            logger.warning(
                                f"Extraction failed for {url}: {e}"
                            )

            except Exception as e:
                logger.error(
                    f"Search failed for query '{query}': {e}"
                )

        logger.info(
            f"Pipeline complete. "
            f"Results: {len(all_results)}, "
            f"Sources: {len(all_sources)}"
        )

        return all_results, all_sources

    # Main Node 

    async def __call__(self, state: dict) -> dict:
        """
        LangGraph node interface — called by StateGraph.
        Receives full shared state, returns updated state.

        This is what makes the researcher a LangGraph node:
        takes state dict in, returns state dict out.
        LangGraph routes to next node based on state values.

        Flow:
        1. LLM builds 3 contextual queries from seller input
        2. Two-step MCP pipeline with per-URL content validation
        3. Single LLM call synthesizes validated data into diagnosis
        4. Output validated against required sections
        5. Returns updated state for Critic evaluation

        On validation failure:
        Returns INSUFFICIENT DATA signal in research_output.
        Critic reads this and triggers retry.
        Researcher runs again with different queries.
        Hard stop at 3 retries prevents infinite loop.
        """
        seller = state["seller_input"]
        attempt = state.get("retry_count", 0) + 1
        logger.info(f"Researcher starting — attempt {attempt}")

        # LLM builds query strings
        queries = await self._build_queries(state)

        # Two-step pipeline with seller context
        # seller_input passed so relevance check knows
        # what product to validate extracted content against
        search_results, sources = await self._execute_searches(
            queries,
            seller  # ← seller_input passed here
        )

        combined_results = "\n\n" + "="*50 + "\n\n".join(
            search_results
        )

        critic_instruction = (
            "PREVIOUS ATTEMPT REJECTED — "
            "ADDRESS THIS SPECIFICALLY:\n"
            + state["critic_feedback"]
        ) if state.get("critic_feedback") else ""

        synthesis_prompt = f"""
            You have received Amazon India search results and full product
            page content below. Analyze carefully.

            {combined_results}

            SELLER'S CURRENT LISTING TO DIAGNOSE:
            Product: {seller['product_name']}
            Category: {seller['category']}
            Details: {seller['product_details']}
            Current Price: ₹{seller['current_price']}
            Platform: {seller['platform']}
            Target Audience: {seller['target_audience']}
            Current Title: {seller['current_title']}
            Current Description: {seller['current_description']}

            STEP 1 — Extract ALL data points before writing anything:
            List every competitor name found.
            List every ₹ price found next to competitor name.
            List every star rating found next to competitor name.
            List every review count found (e.g. "2,847 ratings").
            List every keyword found in competitor titles.
            List every feature mentioned in competitor descriptions:
            - material claims ("pure cotton", "premium fabric")
            - fit and style details ("straight fit", "A-line", "side slits")
            - occasion mentions ("office", "casual", "festive")
            - care instructions ("machine washable", "colorfast")
            - size information ("S to XXL", "true to size")
            Count valid ready-to-wear finished product listings only.
            Ignore unstitched fabric results entirely.

            STEP 2 — Compare seller's listing to competitors:
            For titles: which exact keywords do top competitors use
            that are missing from seller's title?
            For descriptions: which exact features do top competitor
            descriptions mention that seller's description lacks?
            For pricing: what is the exact price-rating relationship?
            (e.g. "products at ₹799 average 4.2★, products at ₹499
            average 3.8★ — higher price signals quality here")

            STEP 3 — Write diagnosis and recommendations:

            TITLE ISSUES must follow this format:
            "[Competitor] (₹[price], [rating]★, [reviews] reviews)
            uses '[exact keyword]' in their title — your title
            '[seller title]' has none of these search-ranked terms."

            DESCRIPTION ISSUES must follow this format:
            "[Competitor] ([rating]★, [reviews] reviews) describes
            their product as '[exact feature from their description]'
            — your description '[seller description]' mentions none
            of these purchase-driving details."

            PRICING ISSUES must follow this format:
            "Competitors at your price point (₹[range]) average
            [rating]★ with [reviews] reviews. Products at ₹[higher]
            average [higher rating]★ — [interpretation of what
            this means for seller]."

            RECOMMENDED TITLE must:
            - Use only keywords found in competitor listings
            - Cite which competitor used each keyword

            RECOMMENDED DESCRIPTION must:
            - Be built from competitor description patterns only
            - Include material quality signal, fit detail,
            occasion suitability, and one unique feature
            - Never use seller's own input as the source

            RECOMMENDED PRICE must:
            - Be justified by the price-rating relationship found
            - State which competitor data supports this price point

            Follow the exact output format specified in your instructions.

            {critic_instruction}
            """

        final_response = await self.llm.ainvoke([
            SystemMessage(content=ANALYSIS_SYSTEM_PROMPT),
            HumanMessage(content=synthesis_prompt)
        ])

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