from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import SystemMessage, AnyMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Send
from langgraph.prebuilt import ToolNode
import os
from pydantic import BaseModel, Field
from typing import Dict, TypedDict, Annotated, List
from api_keys import tavily_api_key
from operator import add
from models import gpt5_medium_reasoning, gpt5_high_reasoning

MASTER_SYSTEM_PROMPT = """
    You are a specialised agent in the "Stock Selection Graph".

    Identity:
    - You are not a general assistant; you are part of an AI research and portfolio-construction system.
    - Quantitative analysis of the stock has already been undertaken using the following methodology:
        - Subscription based stock data bought and downloaded
        - Derivation of a range technical indicators based off price action
        - Extraction of financial company metrics
        - Time series analysis of financial metrics derived to show movements over time
        - Sector specific averages of financial metrics derived
            -> Resulting in circa 2000 quarterly ML features 
        - XGBoost model fitted to infer the yearly stock returns in terms of a multiple:
            e.g. 1.5 -> 50 percent increase
            e.g. 0.9 -> 10 percent decrease
    
    Purpose:
    - The system identifies and ranks stocks to maximise long-term return while managing volatility, correlation, and sector diversity.
    - The system utilises upstream XGBoost predictions of ROI and combines then with deep research to optimise portfolio stock selection.

    Behaviour:
    - Base reasoning on financial, technical, sentiment, general sector performance, company specific
     information.
    - The overall goal of the stock selection graph is to support the above quantitative analysis
    with deeply researched semi-quantitative/qualitive information
    - Ultimately, I wish to identify signals that are ultimately not present in the purely quantitative
    model in order to make my selection more robust, limiting risk and maximising returns.

    Style:
    - Analytical, precise, and concise.
    - Explain rationale for each conclusion or metric.

    Objective:
    - Produce insights or decisions that help select a robust, well-diversified, high-ROI stock portfolio.

    All stocks being analysed are traded on the NYSE.
"""

os.environ["TAVILY_API_KEY"] = tavily_api_key

class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Search query for retrieval.")

class ResearchState(TypedDict):
    ticker: str
    company_name: str
    company_sector: str
    expected_returns: float
    volatility_measures: Dict
    messages: Annotated[List[AnyMessage], add_messages]
    initial_context: Annotated[List[str], add]
    
def tavily_search(
    search_query: str,
    max_results: int = 5
) -> List[str]:
    """
    """

    tavily_search = TavilySearchResults(max_results=max_results)
    search_docs = tavily_search.invoke(search_query)

    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
            for doc in search_docs
        ]
    )

    return formatted_search_docs

def initial_context_web_search(
    state: ResearchState
) -> Dict:
    
    prompt = MASTER_SYSTEM_PROMPT + f"""

        Based off the following company information:

        Ticker: {state.get('ticker')}
        Company Name: {state.get('company_name')}
        Company Sector: {state.get('company_sector')}

        Build a brief query that will be used for web search.
        The result will provide downstream agents initial context 
        about the above company. The query should produce results 
        that provide general information about the company itself 
        for downstream grounding context.
    """

    search_query = gpt5_medium_reasoning\
        .with_structured_output(SearchQuery)\
            .invoke(SystemMessage(prompt))
    
if __name__ == '__main__':
    search_result = tavily_search("Verastar company")
    with open('test_search_result.txt', 'wb') as f:
        f.write(search_result.encode('utf-8'))