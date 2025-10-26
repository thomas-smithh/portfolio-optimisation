from langchain_openai import AzureChatOpenAI
from api_keys import azure_open_ai_api_key

azure_open_ai_config_gpt5 = {
    "api_key": azure_open_ai_api_key,
    "azure_endpoint": "https://pg-oai01.openai.azure.com/",
    "azure_deployment": "gpt-5",
    "model": "gpt-5",
    "deployment_name": "gpt-5",
    "api_version": "2025-03-01-preview"
}

gpt5_medium_reasoning = AzureChatOpenAI(
    **azure_open_ai_config_gpt5,
    temperature=0,
    max_retries=3,
    timeout=60*10
)

gpt5_high_reasoning = AzureChatOpenAI(
    **azure_open_ai_config_gpt5,
    temperature=0,
    reasoning={"effort": "high", "summary": "auto"},
    max_retries=3,
    timeout=60*10
)