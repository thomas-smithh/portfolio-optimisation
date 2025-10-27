from langchain_tavily import TavilySearch
from langchain_community.document_loaders import WikipediaLoader
from langchain_core.messages import SystemMessage, AnyMessage, AIMessage, HumanMessage, ToolMessage, RemoveMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Send
from langgraph.prebuilt import ToolNode
import os
from pydantic import BaseModel, Field
from typing import Dict, TypedDict, Annotated, List, Literal
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

class InitialContext(BaseModel):
    initial_context: str = Field(..., description="Summarised web and wikipedia information.")

class RollingMessageSummary(BaseModel):
    summary: str = Field(..., description="Message Summary")

class SearchQuery(BaseModel):
    search_query: str = Field(..., description="Search query for retrieval.")

class CriticResponse(BaseModel):
    response: str = Field(..., description="Questions to pose to the research agent.")
    critic_satisfied: bool = Field(..., description="True if your satisfied with the depth of analysis and a response has been presented to all lines of questioning.")

class ResearchOutput(BaseModel):
    research_output: str = Field(..., description="Final, self-contained analyst brief synthesizing the prior agent-critic exchange.")
    investment_recommendation: Literal["strong_sell", "sell", "hold", "buy", "strong_buy"] = Field(..., description="Portfolio recommendation based off the deep research exchange.")
    investment_recommendation_executive_summary: str = Field(..., description="2-3 sentences describing why the investment recommendation was chosen in non-technical terms.")
    risk_level: Literal["very_low", "low", "medium", "high", "very_high"]
    risk_level_executive_summary: str = Field(..., description="2-3 sentences describing why the risk-level category was chosen in non-technical terms.")

class ResearchState(TypedDict):
    ticker: str
    company_name: str
    company_sector: str
    expected_returns: float
    volatility_measures: Dict
    messages: Annotated[List[AnyMessage], add_messages]
    rolling_message_summary: str
    initial_context: Annotated[List[str], add]
    initial_context_summary: str
    critic_satisfied: bool
    critic_counter: int
    research_output: ResearchOutput

def tavily_search(
    search_query: str,
    max_results: int = 5,
    deep: bool = True
) -> List[str]:
    """
    Perform a web search using the Tavily API and return formatted documents
    suitable for downstream reasoning or summarisation.

    This function queries Tavily's search engine for a given text prompt,
    optionally performing a deeper multi-pass search. It formats the search
    results as XML-like document blocks to make them easy to parse or embed
    in model prompts.

    Args:
        search_query (str): The query or search phrase to retrieve information about.
        max_results (int, optional): The number of documents to return. Defaults to 5.
        deep (bool, optional): Whether to use advanced (multi-layer) search depth.
            - True → uses Tavily's “advanced” mode for richer context.
            - False → uses a faster “basic” search.

    Returns:
        List[str]: A list containing a single concatenated string of formatted
        search results. Each result is delimited by "---" and wrapped in
        <Document> tags with the source URL.
    """

    tavily_search = TavilySearch(max_results=max_results)
    search_output = tavily_search.invoke(
        {
            "query": search_query,
            "search_depth": "advanced" if deep else "basic",
        }
    )

    docs = search_output.get("results", [])
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{doc.get("url", "")}"/>\n{doc.get("content", "")}\n</Document>'
            for doc in docs
        ]
    )

    return formatted_search_docs

tavily_search_tool = StructuredTool.from_function(
    func=tavily_search,
    name="tavily_search",
    description=tavily_search.__doc__
)

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
            .invoke([SystemMessage(prompt)])

    web_search_context = tavily_search(
        search_query.search_query,
        max_results=10
    )

    return {
        "initial_context": [web_search_context]
    }

def initial_context_wikipedia_search(
    state: ResearchState
) -> Dict:
    
    prompt = MASTER_SYSTEM_PROMPT + f"""

        Based off the following company information:

        Ticker: {state.get('ticker')}
        Company Name: {state.get('company_name')}
        Company Sector: {state.get('company_sector')}

        Build a brief query that will be used for wikipedia search.
        The result will provide downstream agents initial context 
        about the above company. The query should produce results 
        that provide general information about the company itself 
        for downstream grounding context.
    """

    search_query = gpt5_medium_reasoning\
        .with_structured_output(SearchQuery)\
            .invoke([SystemMessage(prompt)])

    search_docs = WikipediaLoader(
        query=search_query.search_query, 
        load_max_docs=10
    ).load()

    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ]
    )

    return {"initial_context": [formatted_search_docs]}
    
def initial_context_summariser(
    state: ResearchState
) -> Dict:
    """
    """

    prompt = MASTER_SYSTEM_PROMPT + f"""
        You are the **Initial Context Summariser** node.

        You are summarising information about the following company:

        Ticker: {state.get('ticker')}
        Company Name: {state.get('company_name')}
        Company Sector: {state.get('company_sector')}

        Purpose:
        - Combine and distil the information retrieved from the web search and Wikipedia search nodes.
        - Produce a concise, factual brief that gives downstream agents enough context
        to understand what the business is, what it does, and any notable recent events.

        Instructions:
        - Use only relevant, credible details about the company.
        - Discard clearly irrelevant or incorrect text (e.g., unrelated entities, broken pages, or generic filler).
        - Include, where available:
            • Company overview (core products, services, markets)
            • Scale and sector position
            • Key customers or geographies
            • Recent performance or developments
            • Any major risks, opportunities, or controversies
        - Keep it factual, objective, and 2-5 short paragraphs long.
        - Avoid copying verbatim marketing language.
        - If little valid information exists, summarise what can be inferred and note uncertainty.

        Context to summarise:
        {state.get("initial_context")}
    """

    summary = gpt5_medium_reasoning.with_structured_output(InitialContext)\
        .invoke([SystemMessage(prompt)])\
            .initial_context
    
    return {
        "initial_context_summary": summary
    }

def agent(
    state: ResearchState
) -> Dict:
    """
    """

    prompt = MASTER_SYSTEM_PROMPT + f"""
        You are the **Investment Research Agent** within the Stock Selection Graph.

        You have access to external search tools (e.g., TavilySearch) that allow you
        to perform **deep, multi-source market research** on a specific stock or company.

        Your task:
        - Evaluate the **investment viability** of the following company:
            • Ticker: {state.get('ticker')}
            • Company Name: {state.get('company_name')}
            • Sector: {state.get('company_sector')}

        You are starting with the following **initial grounding context**, summarised
        by previous nodes:

        {state.get('initial_context_summary')}

        This context is just to give an initial impression of who the business is and 
        what they do. There is no need to explore irrelevant information. We care
        **only** about information that may be indicative of a rising/falling stock price
        in the next year,

        Your goals:
        1. Explore all relevant lines of inquiry that could materially influence the stock's 
           performance over the next year. Your analysis must go beyond surface-level facts to 
           identify why the stock price is likely to rise or fall, grounding every conclusion 
           in credible evidence. This is your most critical objective: to deliver a well-reasoned, 
           evidence-based view of the company's near-term investment potential.
        2. **Evaluate fundamentals to a degree** — analyse financial health, earnings trends, 
           debt levels, margins, and sector-relative valuation. As you're providing supplemenary
           analysis to a quantitive traditional ML model, don't focus too heavily on this
           area.
        3. **Assess external environment** — identify key macroeconomic, regulatory, 
           or technological trends affecting the company and its sector.
        4. **Analyse risks and opportunities** — summarise any emerging risks (competition, 
           litigation, geopolitical exposure, supply chain dependencies) and potential 
           upside opportunities (new products, growth markets, innovation).
        5. **Evaluate sentiment and outlook** — synthesise perspectives from analysts, 
           investor commentary, and news coverage.
        6. Take care to **focus on recent and up to date information** to guide your
           analysis.

        Reminder:
        - The goal is **deep research**, the aim is to uncover trends, signals or 
        significant indicators that the rest of the market might not uncover, gaining
        a competetive advantage in my portfolio selection.

        Interaction configuration:
        - You are collaborating with a "human" critic who reviews your reasoning and asks for clarification or deeper exploration.
        - Base **all** findings on verifiable web research using your available tools and always cite your sources. 
        If reliable data cannot be found, state this clearly.
        - Organise your work into **discrete "units of work"**, each following this cycle:
            1. **Propose** a specific research question or focus area relevant to the company (e.g., “Investigating recent revenue drivers”).
            2. **Execute** your research using tool calls (e.g., TavilySearch) to gather supporting evidence.
            3. **Summarise** your findings and reasoning for that unit of work in clear prose, including references.
        - After completing a unit of work and presenting its summary, **stop issuing further tool calls**.
        Wait for the human critic's feedback before continuing.
        - Do **not** chain multiple research cycles together or emit new tool calls in your summary response.
        The critic will review your current findings and direct the next step.
    """

    if state.get('rolling_message_summary'):
        prompt = prompt + "\n\nPrior Message Summary:\n\n" + state.get('rolling_message_summary')

    agent_result = gpt5_medium_reasoning\
        .bind_tools([tavily_search_tool])\
            .invoke(
                [
                    SystemMessage(prompt),
                    *state.get('messages', [])
                ]
            )

    return {
        "messages": agent_result
    }

def post_agent_router(
    state: ResearchState
) -> str:
    
    last_message = state.get('messages')[-1]

    if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
        return 'tools'
    elif not state.get('critic_satisfied', False):
        return "critic"
    else:
        return "summariser"

def critic(
    state: ResearchState
) -> Dict:
    
    critic_counter = state.get('critic_counter')
    if not critic_counter:
        critic_counter = 1
    else:
        critic_counter += 1

    prompt = MASTER_SYSTEM_PROMPT + f"""
        You are the **Human Critic** in the Stock Selection Graph.

        Role:
        - Review the agent's latest investment research below.
        - Provide focused, evidence-driven critique to improve reasoning and ensure depth.
        - Push for clarity and relevance, but recognise when the analysis has reached satisfactory completeness.

        Objectives:
        1. Identify weak or speculative reasoning.
        2. Question unsupported claims or assumptions.
        3. Suggest where more data or context is needed.
        4. Highlight overlooked risks or alternative viewpoints.
        5. Encourage deeper analysis — but stop when answers are sufficient.
        6. Keep all questions relevant to factors that **directly affect the company's future stock price**.
        7. Disregard tangential or immaterial topics (e.g., minor PR events, marketing language, etc.).
        8. After several exchanges (≈3-5 rounds), aim to conclude the discussion by declaring satisfaction if no new material insights remain.

        ---

        **Question format:**
        Divide your response into two sections:
        1. **Deeper Exploration** — 2-3 follow-up questions probing unresolved aspects of the agent's *current research focus* (e.g., financial assumptions, macro context, sector dynamics).
        2. **New or Untouched Areas** — 2-3 thoughtful questions introducing fresh but relevant angles that could materially affect the company's future performance (e.g., regulatory changes, supply chain risks, disruptive technologies).

        ---

        **Termination and Exit Logic (critical):**
        - You should **not** ask more than ~5 total rounds of questions across the conversation.
        - If your previous questions have been answered comprehensively, or if further discussion is repetitive or speculative, set:
        ```
        critic_satisfied = true
        response = "The analysis appears complete and robust. No further critique required."
        ```
        - Only continue if substantial analytical gaps remain that could materially affect investment conclusions.
        - If this is round number ≥ 3 and all major questions have been addressed, treat this as your **final round** and close the loop.

        - Current question round: {critic_counter}
    """

    if state.get('rolling_message_summary'):
        prompt = prompt + "\n\nPrior Message Summary:\n\n" + state.get('rolling_message_summary')

    critic_response = gpt5_medium_reasoning\
        .with_structured_output(CriticResponse)\
            .invoke(
                [
                    SystemMessage(prompt),
                    *state.get('messages', [])
                ]
            )
    
    return {
        "messages": [HumanMessage(content=critic_response.response)],
        "critic_satisfied": critic_response.critic_satisfied if critic_counter < 5 else True,
        "critic_counter": critic_counter
    }

def rolling_message_summary(
    state: ResearchState
) -> Dict:
    """
    Summarise all but the last 5 groups of messages if history exceeds 10.
    Groups are defined by tool_call_id to ensure tool calls and responses stay together.
    Also returns RemoveMessage entries for the messages being dropped.
    """
    messages = state.get("messages", [])
    if len(messages) <= 10:
        return {}  # no change

    # --- Step 1: Group messages by tool_call_id ---
    groups = []
    temp_group = []
    open_tool_ids = set()

    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            # Start a new group for each tool_call
            for call in msg.tool_calls:
                open_tool_ids.add(call["id"])
            temp_group.append(msg)

        elif isinstance(msg, ToolMessage):
            # Match with existing tool_call_id
            if msg.tool_call_id in open_tool_ids:
                temp_group.append(msg)
                open_tool_ids.remove(msg.tool_call_id)

                # If no open calls, close group
                if not open_tool_ids:
                    groups.append(temp_group)
                    temp_group = []
            else:
                # Orphan ToolMessage (unexpected) → treat as its own group
                groups.append([msg])

        else:
            # Regular human/system/ai message
            if temp_group:
                # If we had open tool_calls, close the group here
                groups.append(temp_group)
                temp_group = []
            groups.append([msg])

    if temp_group:
        groups.append(temp_group)

    # --- Step 2: Split into summarise vs keep ---
    if len(groups) <= 5:
        return {}  # not enough to summarise

    to_summarise = groups[:-5]
    recent = groups[-5:]

    # --- Step 3: Summarise older groups ---
    flat_to_summarise = [m for group in to_summarise for m in group]
    summary_prompt = [
        SystemMessage(MASTER_SYSTEM_PROMPT),
        SystemMessage("You are a summarisation assistant. The generated summary can be as verbose as necessary to retain the bulk of significant meaning. Summarise the following conversation:"),
        SystemMessage("\n\n".join([state.get('rolling_message_summary', '')] + [str(m) for m in flat_to_summarise]))
    ]
    summary = gpt5_medium_reasoning.with_structured_output(RollingMessageSummary).invoke(summary_prompt)

    # --- Step 5: Collect IDs to remove ---
    delete_messages = [RemoveMessage(id=m.id) for m in flat_to_summarise if hasattr(m, "id")]

    return {
        "messages": delete_messages,
        "rolling_message_summary": summary.summary
    }

def summariser(
    state: ResearchState
) -> Dict:
    
    prompt = MASTER_SYSTEM_PROMPT + f"""
        **Provide a structured conclusion of the previous critic-agent inteaction** — 
        judging whether the future outlook is good or bad given the context, 
        and justify your assessment.

        Output format:
        - Write in clear, professional, research-analyst style.
        - Organise output as:
            1. Company Overview
            2. Recent Developments
            3. Financial and Market Analysis
            4. Risk and Opportunity Assessment
            5. Sentiment and Analyst Outlook
            6. Summary and Investment View

        Additional guidance:
        - Base all claims on credible, sourced evidence from the previous agent-critic interaction.
        - Avoid speculation without data.
        - Highlight uncertainties or conflicting signals where relevant.
        - Keep length to roughly 6-10 short, information-dense paragraphs.
    """

    if state.get('rolling_message_summary'):
        prompt = prompt + "\n\nPrior Message Summary:\n\n" + state.get('rolling_message_summary')

    research_output = gpt5_medium_reasoning\
        .with_structured_output(ResearchOutput)\
            .invoke([SystemMessage(prompt)] + state.get('messages', []))
    
    return {
        "research_output": research_output
    }

def create_and_compile_graph():

    graph = StateGraph(ResearchState)
    graph.add_node("initial_context_web_search", initial_context_web_search)
    graph.add_node("initial_context_wikipedia_search", initial_context_wikipedia_search)
    graph.add_node("initial_context_summariser", initial_context_summariser)
    graph.add_node("agent", agent)
    graph.add_node("critic", critic)
    graph.add_node(
        "tools",
        ToolNode(
            [
                tavily_search_tool
            ]
        )
    )
    graph.add_node('rolling_message_summary', rolling_message_summary)
    graph.add_node('summariser', summariser)

    graph.add_edge(START, "initial_context_web_search")
    graph.add_edge(START, "initial_context_wikipedia_search")
    graph.add_edge("initial_context_web_search", "initial_context_summariser")
    graph.add_edge("initial_context_wikipedia_search", "initial_context_summariser")
    graph.add_edge("initial_context_summariser", "agent")

    graph.add_conditional_edges(
        "agent",
        post_agent_router,
        {
            "tools": "tools",
            "critic": "critic",
            "summariser": "summariser"
        }
    )

    graph.add_edge("tools", "agent")
    graph.add_edge("critic", "rolling_message_summary")
    graph.add_edge("rolling_message_summary", "agent")
    graph.add_edge("summariser", END)

    return graph.compile()