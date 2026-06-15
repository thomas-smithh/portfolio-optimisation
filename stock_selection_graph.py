import json
from datetime import date
from pathlib import Path
from typing import Literal, Dict, List
import numpy as np
from llms import models
from langchain.tools import tool, ToolRuntime
from langchain.agents.middleware import ToolCallLimitMiddleware
from tavily import TavilyClient

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain.agents.structured_output import ToolStrategy
from langchain.agents import create_agent

from schemas import (
    research_categories,
    StockInfoAndResearch,
    CategoryResearch,
    StockResearch,
    ResearchState,
    StockSelectionState,
    StockResearchSubgraphState,
    CollationOutput, 
    PortfolioSelection
)

with open("api_keys.json", "r", encoding="utf-8") as f:
    api_keys = json.load(f)["api_keys"][0]

MAX_CONCURRENCY = 1

@tool
def web_search(
    query: str,
    runtime: ToolRuntime,
    max_results: int = 5,
    deep: bool = True,
) -> str:
    """
    Execute a web-search query using Tavily.
    """
    
    if runtime.state.get('back_test_date', None):
        end_date = runtime.state.get('back_test_date')
    else:
        end_date = None

    client = TavilyClient(api_keys['tavily'])
    search_output = client.search(
        query=query,
        max_results=max_results,
        search_depth="advanced" if deep else "basic",
        end_date=end_date
    )

    docs = search_output.get("results", [])
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{doc.get("url", "")}"/>\n{doc.get("content", "")}\n</Document>'
            for doc in docs
        ]
    )

    return formatted_search_docs

@tool
def array_calc(
    values: list[float],
    operation: Literal["sum", "mean", "median", "min", "max", "std", "var"]
) -> float:
    """
    Perform a statistical calculation on a list of numeric values.
    """
    arr = np.array(values, dtype=float)

    ops = {
        "sum": np.sum,
        "mean": np.mean,
        "median": np.median,
        "min": np.min,
        "max": np.max,
        "std": np.std,
        "var": np.var,
    }
    return float(ops[operation](arr))

@tool
def basic_calc(
    left_operand: float,
    right_operand: float,
    operation: Literal["add", "subtract", "multiply", "divide"]
) -> float:
    """
    Perform a basic arithmetic calculation on two numeric inputs.
    """

    if operation == "add":
        return left_operand + right_operand
    if operation == "subtract":
        return left_operand - right_operand
    if operation == "multiply":
        return left_operand * right_operand
    if right_operand == 0:
        raise ValueError("Cannot divide by zero.")
    return left_operand / right_operand

def send_by_stock(
    state: StockSelectionState
) -> List[Send]:
    """
    """

    return [
        Send(
            "stock_research",
            {
                "candidate_stock": candidate_stock,
                "back_test_date": state.get("back_test_date"),
            }
        ) for candidate_stock in state.get('stocks', [])
    ]

def send_by_category(
    state: StockResearchSubgraphState
) -> List[Send]:
    """
    """

    return [
        Send(
            "topic_research",
            {
                "candidate_stock": state.get('candidate_stock'),
                "category": research_category,
                "back_test_date": state.get("back_test_date"),
            }
        ) for research_category in research_categories.keys()
    ]

def topic_research(
    state: ResearchState
) -> Dict:
    """
    """

    candidate_stock = state.get('candidate_stock').model_dump_json(
        exclude={"research"},
        indent=2
    )
    research_category = state.get('category')
    category_description = research_categories[research_category]
    current_date = date.today().isoformat() if not state.get('back_test_date', None) else state.get('back_test_date')

    prompt = f"""
        You are a deep stock research analyst tasked with conducting in-depth research on a single stock for one specific research category.

        The investment decision is being made today: {current_date}. Disregard **any** information that would not have been available to a researcher on this date. 
        Assume the portfolio decision and all research framing should be anchored to information available as of this date.

        Candidate stock data from state (JSON):
        {candidate_stock}

        Research category to analyze: {research_category}

        Research category description:
        {category_description}

        Your assignment:
        - Investigate this stock only through the lens of the specified research category.
        - Gather current, relevant, decision-useful evidence.
        - Focus on concrete facts, developments, signals, and implications that could affect whether this stock should be included in the portfolio today.
        - Prioritize recency, source quality, and evidence that changes the investment case.
        - Distinguish verified facts from your interpretation when the evidence is incomplete or mixed.

        Tool guidance:
        - web_search: use this to gather current external information, company-specific developments, industry context, and supporting evidence.
        - basic_calc: use this for any arithmetic such as addition, subtraction, multiplication, division, percent changes, or ratio calculations.
        - array_calc: use this for any list or array statistics such as mean, median, min, max, standard deviation, variance, or sums.
        - For any numerical derivation, aggregation, average, comparison, or percentage-based reasoning, use the calculator tools rather than mental math to preserve accuracy.

        Research expectations:
        - Search broadly enough to capture the most decision-relevant information for this category.
        - Prefer company filings, earnings materials, reputable financial journalism, official announcements, and high-quality industry sources when available.
        - Ignore irrelevant information that does not materially inform this category.
        - Do not analyze other categories unless a fact is necessary for context.

        Output requirements:
        - Return a CategoryResearch result for the specified category only.
        - signal must be a score between 0 and 1, where higher means stronger case for portfolio inclusion from this category alone.
        - confidence must be a score between 0 and 1 reflecting how strong and reliable the evidence is.
        - qualitative_assessment must read like a concise but substantive research note with structure and line breaks, not a short blurb.
        - The qualitative assessment should summarize the evidence, explain the investment implication, and justify both the signal and confidence scores.
    """

    agent = create_agent(
        model=models['medium'],
        tools=[
            array_calc, 
            basic_calc,
            web_search
        ],
        system_prompt=prompt,
        response_format=ToolStrategy(CategoryResearch),
        middleware=[
            ToolCallLimitMiddleware(
                run_limit=100
            )
        ]
    )

    response = agent.invoke({})["structured_response"]

    return {
        "research": [response]
    }

def collate_research(
    state: StockResearchSubgraphState
) -> Dict:
    """
    """

    current_date = date.today().isoformat() if not state.get('back_test_date', None) else state.get('back_test_date')

    prompt = f"""
        You are a research synthesis analyst tasked with collating and synthesizing the research findings across multiple categories for a single stock.

        The investment decision is being made today: {current_date}. Disregard **any** information that would not have been available to a researcher on this date. 
        Assume the portfolio decision and all research framing should be anchored to information available as of this date.

        Candidate stock data from state (JSON):
        {state.get('candidate_stock').model_dump_json(indent=2)}

        Research findings to synthesize:
        {state.get('research')}

        Your assignment:
        - Synthesize the research findings across all categories for this stock.
        - Identify key themes, strengths, weaknesses, and overall investment implications.
        - Provide an integrated qualitative summary that captures the most important insights from the combined research.
        - Derive an overall signal score between 0 and 1 that reflects the aggregated attractiveness of this stock based on all research categories.

        Output requirements:
        - mean_signal must be the average of the signal scores across all categories.
        - qualitative_summary must be a concise but substantive synthesis of the research findings, highlighting the main factors influencing the investment case for this stock. It should read like a research note with structure and line breaks, not a short blurb.
    """

    agent = create_agent(
        model=models['high'],
        tools=[
            array_calc,
            basic_calc,
        ],
        system_prompt=prompt,
        response_format=ToolStrategy(CollationOutput),
        middleware=[
            ToolCallLimitMiddleware(
                run_limit=10
            )
        ]
    )

    response = agent.invoke({})["structured_response"]
    stock_info = state.get('candidate_stock')
    stock_research = StockResearch(
        research=state.get('research'),
        mean_signal=response.mean_signal,
        mean_confidence=response.mean_confidence,
        qualitative_summary=response.qualitative_summary
    )

    stock_info_and_research = StockInfoAndResearch(
        ticker=stock_info.ticker,
        name=stock_info.name,
        sector=stock_info.sector,
        companysite=stock_info.companysite,
        prediction_pct_change=stock_info.prediction_pct_change,
        research=stock_research
    )

    return {
        "stock_research": [stock_info_and_research]
    }

def portfolio_selection(
    state: StockSelectionState
) -> Dict:
    """
    """

    stock_research_json = json.dumps(
        [stock.model_dump() for stock in state.get('stock_research', [])],
        indent=2
    )
    number_of_stocks_to_select = state.get("number_of_stocks_to_select")
    current_date = date.today().isoformat() if not state.get('back_test_date', None) else state.get('back_test_date')

    prompt = f"""
        You are a portfolio construction analyst tasked with selecting a final stock portfolio from a researched candidate set.

        The investment decision is being made today: {current_date}. Disregard **any** information that would not have been available to a researcher on this date. 
        Assume the portfolio decision and all research framing should be anchored to information available as of this date.

        Researched stock universe from state (JSON):
        {stock_research_json}

        Target number of stocks to select: {number_of_stocks_to_select}

        Portfolio construction guidelines:
        - Maximise expected returns while simultaneously minimising risk.
        - Exclude stocks with major red flags uncovered by the research, even if their expected return appears attractive (weigh up risk vs. reward here)
        - Diversify the portfolio as much as possible (by sector or otherwise) across the available opportunity set without heavily sacrificing returns.
        - The portfolio has a fairly high risk appetite, so it is acceptable to include higher-volatility names when the risk is well understood and the upside is compelling.
        - Each stock selected will be given an equal weighting in the final portfolio.

        Decision framework:
        - Use each stock's prediction_pct_change, mean_signal, mean_confidence, qualitative_summary, and category-level research to assess inclusion or exclusion.
        - Weigh upside against downside risk, fragility, and uncertainty.
        - Avoid concentrating the portfolio in stocks that share the same obvious risk drivers when credible alternatives exist.
        - Prefer stocks with strong upside and acceptable risk-adjusted profiles, not simply the highest raw return expectations.
        - Use the calculator tools for any numeric comparison, ranking support, averages, or other derivations rather than mental math.

        Output requirements:
        - Return a PortfolioSelection object.
        - selected_stocks should contain the final portfolio constituents and should align with the target number of stocks unless the research strongly justifies selecting fewer.
        - excluded_stocks should contain the researched names that are not chosen.
        - For each selected or excluded stock, provide a concise but specific reasoning statement grounded in the research.
        - qualitative_risk_level must be between 0 and 1, where 1 is the highest anticipated risk.
        - rationale must explain the overall portfolio construction logic, the main tradeoffs made between return and risk, and how diversification was achieved without sacrificing expected return too heavily.
    """

    agent = create_agent(
        model=models['high'],
        tools=[
            array_calc,
            basic_calc,
        ],
        system_prompt=prompt,
        response_format=ToolStrategy(PortfolioSelection),
        middleware=[
            ToolCallLimitMiddleware(
                run_limit=50
            )
        ]
    )

    response = agent.invoke({})['structured_response']

    return {
        "portfolio_selection": response
    }

def create_and_compile_stock_research_graph():
    """
    """
    graph = StateGraph(StockResearchSubgraphState)
    graph.add_node("topic_research", topic_research)
    graph.add_node("collate_research", collate_research)

    graph.add_conditional_edges(START, send_by_category)
    graph.add_edge("topic_research", "collate_research")
    graph.add_edge("collate_research", END)

    return graph.compile().with_config(
        {
            "max_concurrency": MAX_CONCURRENCY
        }
    )

def create_and_compile_graph():
    """
    """
    graph = StateGraph(StockSelectionState)
    graph.add_node("stock_research", create_and_compile_stock_research_graph())
    graph.add_node("portfolio_selection", portfolio_selection)
    
    graph.add_conditional_edges(START, send_by_stock)
    graph.add_edge("stock_research", "portfolio_selection")
    graph.add_edge("portfolio_selection", END)

    graph = graph.compile()
    return graph.with_config(
        {
            "max_concurrency": MAX_CONCURRENCY
        }
    )