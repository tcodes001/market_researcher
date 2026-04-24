# application.py
"""
FastAPI entry point for the Multi-Agent Listing Optimizer.

Responsibilities:
- Initialize MCP connection at startup
- Build LangGraph workflow with loaded tools
- Expose REST endpoints for seller input
- Clean up MCP on shutdown
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from graph.workflow import MarketResearchWorkflow
from tools.mcp_clients import MCPClient
import logging

# ── Logging Setup ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ── FastAPI App ───────────────────────────────────────────────
application = FastAPI(
    title="Multi-Agent Listing Optimizer",
    description=(
        "Agentic AI system that diagnoses and optimizes "
        "e-commerce listings using LangGraph + MCP"
    ),
    version="1.0.0"
)

app = application

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── Singletons ────────────────────────────────────────────────
mcp_client = MCPClient()
workflow = MarketResearchWorkflow()


# ── Startup & Shutdown ────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    """
    Runs once when FastAPI starts.
    Initializes MCP connection and builds LangGraph workflow.
    """
    try:
        logger.info("Starting MCP connection...")
        tools = await mcp_client.initialize()
        logger.info(f"MCP tools loaded: {[t.name for t in tools]}")

        logger.info("Building LangGraph workflow...")
        await workflow.initialize_tools(tools)
        logger.info("✅ App ready. Workflow compiled successfully.")

    except Exception as e:
        logger.error(f"❌ Startup failed: {str(e)}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """
    Runs once when FastAPI shuts down.
    Cleans up MCP session and kills npx process.
    """
    logger.info("Shutting down MCP connection...")
    await mcp_client.cleanup()
    logger.info("✅ MCP session closed cleanly.")


# ── Input Model ───────────────────────────────────────────────
class SellerInput(BaseModel):
    product_name: str = Field(
        ...,
        min_length=2,
        description="Name of the product being sold"
    )
    category: str = Field(
        ...,
        min_length=2,
        description="Product category e.g. Women's Ethnic Wear"
    )
    product_details: str = Field(
        ...,
        min_length=10,
        description=(
            "Specific product details e.g. "
            "cotton fabric, straight fit, festive wear"
        )
    )
    current_price: float = Field(
        ...,
        gt=0,
        description="Current listing price in INR"
    )
    platform: str = Field(
        ...,
        description="Platform where product is listed e.g. Meesho"
    )
    target_audience: str = Field(
        ...,
        description="Target audience e.g. Women 18-35"
    )
    current_title: str = Field(
        ...,
        min_length=3,
        description="Seller's current listing title"
    )
    current_description: str = Field(
        ...,
        min_length=10,
        description="Seller's current listing description"
    )

    @field_validator("current_price")
    @classmethod
    def price_must_be_realistic(cls, v):
        if v < 10:
            raise ValueError(
                "Price seems too low. "
                "Please enter price in INR."
            )
        if v > 100000:
            raise ValueError(
                "Price seems too high for an e-commerce listing."
            )
        return v

    @field_validator("platform")
    @classmethod
    def platform_must_be_valid(cls, v):
        valid_platforms = [
            "meesho", "amazon", "flipkart",
            "myntra", "ajio", "nykaa"
        ]
        if v.lower() not in valid_platforms:
            raise ValueError(
                f"Platform must be one of: "
                f"{', '.join(valid_platforms)}"
            )
        return v
    
    @field_validator("product_details")
    @classmethod
    def product_details_must_be_specific(cls, v):
        """
        Rejects vague product details before agents run.
        Prevents hallucination at the source.
        """
        vague_phrases = [
            "good quality", "nice product", "great item",
            "best product", "good product", "nice quality",
            "very good", "excellent product", "good item",
            "nice", "good", "great", "best", "excellent"
        ]

        details_lower = v.lower().strip()

        if len(details_lower) < 20:
            raise ValueError(
                "Product details are too short. "
                "Please describe your product's specific attributes "
                "such as material, size, color, features, or occasion. "
                "The more specific you are, the more accurate "
                "your market research will be."
            )

        words = details_lower.split()
        meaningful_words = [
            w for w in words
            if w not in vague_phrases
            and len(w) > 3
        ]

        if len(meaningful_words) < 3:
            raise ValueError(
                "Product details must include specific attributes. "
                "Avoid generic phrases like 'good quality' or "
                "'nice product'. Describe what makes your product "
                "unique — its material, dimensions, features, "
                "colors, or intended use."
            )

        return v


# ── Endpoints ─────────────────────────────────────────────────
@app.get("/")
def health_check():
    """
    Health check endpoint.
    Used by AWS Elastic Beanstalk to verify app is running.
    """
    return {
        "status": "online",
        "service": "Multi-Agent Listing Optimizer",
        "version": "1.0.0",
        "domain": "techainet.com"
    }


@app.post("/research")
async def run_research(request: SellerInput):
    """
    Main endpoint — runs the full multi-agent workflow.
    Takes seller's listing details and returns diagnosis
    and recommendations.
    """
    if workflow.graph is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Service not ready. "
                "MCP tools still initializing. "
                "Please try again in a moment."
            )
        )

    try:
        logger.info(
            f"New research request: "
            f"{request.product_name} on {request.platform}"
        )

        result = await workflow.run(request.model_dump())

        logger.info(
            f"Research completed: "
            f"{result['attempts']} attempts, "
            f"verified={result['verified']}"
        )

        return result

    except Exception as e:
        logger.error(f"Research failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Research failed: {str(e)}"
        )