import os
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.yfinance import YFinanceTools

os.environ["OPENAI_API_KEY"] = "Your_API_Key"

agent = Agent(
    model=OpenAIChat(id="gpt-4o"),  
    tools=[YFinanceTools(stock_price=True)],
    instructions="Display the content in a formatted and tabular way.",
    markdown=True,
)

agent.print_response("Which is the best stock to invest in right now?", stream=True)
