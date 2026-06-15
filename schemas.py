from pydantic import BaseModel, Field
from typing import List, Literal, TypedDict, Annotated, Optional
from operator import add

def keep_optional_str(left: Optional[str], right: Optional[str]) -> Optional[str]:
    if left is None:
        return right
    if right is None or left == right:
        return left
    raise ValueError(f"Conflicting values for optional string state: {left} != {right}")

research_categories = {
    "event_risk": (
        "Research material external events that could introduce sudden downside or volatility "
        "for a stock. This includes regulatory actions, legal disputes, government investigations, "
        "product recalls, executive departures, cybersecurity incidents, financing events, major "
        "customer or supplier disruptions, and other company-specific news that may not yet be "
        "fully reflected in technical or fundamental data."
    ),

    "narrative": (
        "Analyze management communication and the evolving company story across earnings calls, "
        "investor presentations, shareholder letters, conference appearances, and interviews. "
        "Focus on tone shifts, guidance credibility, consistency of messaging, evasiveness in Q&A, "
        "changes in strategic priorities, and whether management language suggests improving or "
        "deteriorating business conditions before those changes are visible in reported results."
    ),

    "operational": (
        "Investigate the real-world operating health of the business through supply chain signals, "
        "production status, input cost pressures, shipping and logistics trends, inventory dynamics, "
        "customer demand commentary, hiring trends, and reports of disruptions or delays. The goal "
        "is to identify early signs of operational strength or weakness that could affect future "
        "revenue, margins, or execution."
    ),

    "forensics": (
        "Examine the quality, durability, and credibility of the company's financial reporting. "
        "This includes reviewing footnotes, non-GAAP adjustments, cash flow conversion, receivables, "
        "inventory changes, working capital trends, acquisition accounting, segment reporting changes, "
        "insider transactions, governance issues, and other indicators of aggressive accounting or "
        "weak earnings quality that may make the stock appear stronger than it really is."
    ),

    "competition": (
        "Assess the company's competitive position by monitoring peer activity, pricing changes, "
        "product launches, market-share commentary, customer sentiment, product reviews, app or "
        "web engagement trends, and evidence of shifting demand toward or away from competitors. "
        "The objective is to detect weakening moats, declining product relevance, or strengthening "
        "competitive advantages before they are fully visible in financial performance."
    ),

    "sector": (
        "Evaluate the broader industry and macroeconomic context in which the company operates. "
        "This includes analyzing sector-specific trends, regulatory changes, commodity price movements, "
        "interest rate impacts, consumer behavior shifts, and other external factors that could "
        "influence the company's future performance. Understanding the sector dynamics can provide "
        "valuable context for evaluating the company's prospects and potential risks."
    )
}

class CategoryResearch(BaseModel):
    ticker: str
    category: Literal['event_risk', 'narrative', 'operational', 'forensics', 'competition', 'sector']
    signal: float = Field(
        ..., 
        description=(
            "Score between 0 and 1. Score closer to 1 implies strong buy, closer to 0 implies strong portfolio exclusion."
            " This **must** relate **only** to what the given category implies about stock inclusion "
            " or exclusion from the portfolio."
        )
    )
    confidence: float = Field(
        ...,
        description=(
            "Score between 0 and 1. Score closer to 1 implies high confidence in the signal, closer to 0 implies low confidence. "
            " This should reflect the model's confidence in the strength of the signal it has identified, "
            " and can be influenced by factors such as the quality and quantity of available information"
        )
    )
    qualitative_assessment: str = Field(
        ...,
        description=(
            "A textual description providing a qualitative assessment of the stock based on the given category. "
            "This should complement the numerical signal and confidence scores, offering additional context or insights."
            "This should be a concise summary of the key factors influencing the signal score, and should help explain why the stock received its particular rating in this category."
            "This should read like a report, with line breaks and structure, not just a few sentences. It should be detailed enough to provide a clear rationale for the signal score"
            " , but also concise and focused on the most important points."
        )
    )

class StockResearch(BaseModel):
    research: List[CategoryResearch]
    mean_signal: float = Field(
        ...,
        description=(
            "The average of the signal scores across all research categories for this stock. "
            "This provides an overall indication of the stock's attractiveness based on the combined insights from all categories."
        )
    )
    mean_confidence: float = Field(
        ...,
        description=(
            "The average of the confidence scores across all research categories for this stock. "
            "This provides an overall indication of the reliability of the insights based on the combined confidence levels from all categories."
        )
    )
    qualitative_summary: str = Field(
        ...,
        description=(
            "A concise summary synthesizing the qualitative assessments from all research categories. "
            "This should provide an integrated narrative that captures the key strengths and weaknesses of the stock as identified across the different categories."
        )
    )

class CollationOutput(BaseModel):
    mean_signal: float = Field(
        ...,
        description=(
            "The average of the signal scores across all research categories for this stock. "
            "This provides an overall indication of the stock's attractiveness based on the combined insights from all categories."
        )
    )
    mean_confidence: float = Field(
        ...,
        description=(
            "The average of the confidence scores across all research categories for this stock. "
            "This provides an overall indication of the reliability of the insights based on the combined confidence levels from all categories."
        )
    )
    qualitative_summary: str = Field(
        ...,
        description=(
            "A concise summary synthesizing the qualitative assessments from all research categories. "
            "This should provide an integrated narrative that captures the key strengths and weaknesses of the stock as identified across the different categories."
        )
    )

class StockInfoAndResearch(BaseModel):
    ticker: str
    name: str
    sector: str | None
    companysite: str | None
    prediction_pct_change: float
    research: StockResearch | None = None

class ResearchState(TypedDict):
    candidate_stock: StockInfoAndResearch
    category: Literal['event_risk', 'narrative', 'operational', 'forensics', 'competition', 'sector']
    back_test_date: Annotated[Optional[str], keep_optional_str]

class IndividualStockResearchState(TypedDict):
    candidate_stock: StockInfoAndResearch
    research: Annotated[List[CategoryResearch], add]
    back_test_date: Annotated[Optional[str], keep_optional_str]

class StockResearchSubgraphState(TypedDict):
    candidate_stock: StockInfoAndResearch
    research: Annotated[List[CategoryResearch], add]
    back_test_date: Annotated[Optional[str], keep_optional_str]
    stock_research: Annotated[List[StockInfoAndResearch], add]

class StockSelection(BaseModel):
    ticker: str = Field(..., description="Stock ticker")
    name: str = Field(..., description="Stock name")
    qualitative_risk_level: float = Field(..., description="Anticipated risk, with 0 being no risk and 1 being highly risky.")
    reasoning: str = Field(..., description="Reasoning behind inclusion/exclusion from the finalised portfolio.")

class PortfolioSelection(BaseModel):
    selected_stocks: List[StockSelection] = Field(..., description="Finalised list of stocks to include in the portfolio.")
    excluded_stocks: List[StockSelection] = Field(..., description="Finalised list of exclusions from the portfolio.")
    rationale: str = Field(
        ...,
        description=(
            "A detailed summary of the portfolio, based on the research findings and the anticipated returns. "
            "This should synthesize the insights from all research categories for each stock, and explain how those insights contributed to the decision to include the stock in the portfolio."
            "This should also say how risk was minimised and portfolio was adequately diversified without limiting expected returns too heavily."
        )
    )

class StockSelectionState(TypedDict):
    stocks: List[StockInfoAndResearch]
    stock_research: Annotated[List[StockInfoAndResearch], add]
    number_of_stocks_to_select: int
    portfolio_selection: PortfolioSelection
    back_test_date: Annotated[Optional[str], keep_optional_str]