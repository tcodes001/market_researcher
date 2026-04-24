# agents/config.py
"""
Configuration constants for agent search parameters.

Separated from agent logic so parameter tuning
doesn't require touching agent code.
"""

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

TAVILY_SEARCH_PARAMS = {
    "search_depth": "advanced",
    "include_domains": ["amazon.in"],
    "country": "India",
    "max_results": 10,
    "include_raw_content": True,
    "time_range": "month"
}

TAVILY_EXTRACT_PARAMS = {
    "extract_depth": "advanced",
    "query": "price rating title description reviews",
    "format": "markdown"
}