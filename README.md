# Listing Optimizer
# Listing Optimizer

**Live Demo:** [listingoptimizer.techainet.com](https://listingoptimizer.techainet.com)

A structured agentic AI pipeline that helps Indian e-commerce sellers optimize their product listings using real-time Amazon India competitor data — not static LLM knowledge.

---

## What It Does

Sellers input their product details. The system searches Amazon India live, finds real competitors, and returns:

- **Diagnosis** — what is missing from your title, description, and pricing compared to top competitors
- **Recommended Title** — copy-paste ready, built from competitor-validated keywords
- **Recommended Description** — grounded in competitor description patterns
- **Recommended Price** — based on live competitor price-rating data
- **Keywords** — extracted from top-performing competitor titles
- **Sources** — direct links to Amazon India competitor pages used

------------------------------------------------------------------------------------
## Architecture
```
Seller Input → Researcher → Critic → Validator → Output
                   ↑_____________|        |
                   |                      |
                   |______________________|
                (shared retry counter, max 3 total retries)
```
**Researcher** — builds queries, searches Amazon India via Tavily MCP, extracts content per URL, filters irrelevant results using an LLM, synthesises diagnosis and recommendations.

**Critic** — evaluates research quality against 8 criteria (named competitors, specific prices, star ratings, review counts). Returns RETRY with feedback if criteria not met.

**Validator** — checks if recommendations are feasible for the seller's actual product. Routes back to Researcher on rejection — because a relevance failure is a data problem, not a quality problem.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Agent Orchestration | LangGraph 1.1.4 |
| LLM | GPT-4o |
| Live Data | Tavily MCP via stdio transport |
| Structured Output | Pydantic |
| Backend | FastAPI + Uvicorn |
| Containerization | Docker |
| Cloud | AWS (ECR, Elastic Beanstalk, ALB, ACM, Route 53) |
| CI/CD | AWS CodePipeline + CodeBuild |
| Python | 3.11.15 |

---

## Local Setup

```bash
git clone https://github.com/tcodes001/market_researcher.git
cd market_researcher

# Install dependencies
pip install -r requirements.txt

# Add API keys
echo "OPENAI_API_KEY=your_key" >> .env
echo "TAVILY_API_KEY=your_key" >> .env

# Run
uvicorn application:application --reload
```

Open `http://127.0.0.1:8000`

---

## Key Design Decisions

**LangGraph over plain LangChain** — chains are linear and cannot loop back. LangGraph's conditional edges enable the Critic → Researcher retry loop.

**Tavily MCP over direct API** — MCP decouples the data layer. Agents call tools without knowing their implementation. Amazon also blocks major AI crawlers — Tavily's infrastructure reliably accesses Amazon India product pages.

**Validator routes to Researcher not Critic** — a feasibility failure means wrong competitors were found. The Researcher needs new data, not a better critique of bad data.

**Temperature 0.0 for synthesis** — deterministic output ensures consistent section formatting that the parser can reliably extract.

---

## Known Limitations

- Niche products with fewer than 2 rated Amazon India competitors return thin pricing analysis
- Requests take 30–60 seconds due to sequential Tavily API calls and LLM synthesis
- No rate limiting implemented

---

*Built by Tanisha Solanki 
