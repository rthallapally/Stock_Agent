import os
import json
import yfinance as yf
import plotly.express as px
import streamlit as st
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_tavily import TavilySearch
from typing_extensions import TypedDict
from typing import Annotated
import json
from langgraph.graph.message import add_messages
import os
from dotenv import load_dotenv

# 1. Setup
st.set_page_config(page_title="💹 Stock Agent", page_icon="📈")

st.title("📈 Stock Agent")

# Load .env file
load_dotenv()

# Get API keys from environment
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")


llm = init_chat_model("groq:llama-3.3-70b-versatile")
tavily = TavilySearch(max_results=3)

# 2. Define Tools

@tool
def get_live_stock_price(symbol: str):
    """Get the real-time market price of the given stock symbol."""
    import yfinance as yf

    stock = yf.Ticker(symbol)
    info = stock.info
    price = info.get('regularMarketPrice', None)

    if price is None:
        return {"output": f"❌ Unable to retrieve live price for `{symbol.upper()}`.", "return_directly": False}

    return {
        "output": f"💹 The current market price of **{symbol.upper()}** is **${price:.2f}**.",
        "return_directly": False
    }



@tool
def get_stock_history(symbol: str, days: int = 5):
    """Get the historical stock prices for the last X days."""
    stock = yf.Ticker(symbol)
    history = stock.history(period=f"{days}d")['Close']
    history_str = "\n".join([f"- {d.date()}: **${p:.2f}**" for d, p in history.items()])
    return {
    "output": f"### 📊 Last {days} days of {symbol.upper()}:\n\n{history_str}",
    "return_directly": True
}



@tool
def plot_stock_history(symbol: str, days: int = 30):
    """Plot stock history and return interactive chart HTML and summary."""
    stock = yf.Ticker(symbol)
    history = stock.history(period=f"{days}d")

    fig = px.line(history, x=history.index, y='Close',
                  title=f"{symbol.upper()} Stock Price (Last {days} Days)")

    # Generate interactive HTML
    html_chart = fig.to_html(full_html=False)

    start_price = history['Close'].iloc[0]
    end_price = history['Close'].iloc[-1]
    change_percent = ((end_price - start_price) / start_price) * 100
    trend = "increased" if end_price > start_price else "decreased"
    
    summary = (
    f"### 📈 {symbol.upper()} Stock Trend\n\n"
    f"- Start Price: **${start_price:.2f}**\n"
    f"- End Price: **${end_price:.2f}**\n"
    f"- Change: **{trend}** ({change_percent:.2f}%) over the last {days} days."
)


    return {
        "output": summary,
        "html": html_chart,
        "return_directly": True  # prevent looping back to LLM
    }



@tool
def get_stock_news(symbol: str):
    """Get the latest stock market news for a given company ticker."""
    query = f"latest stock market news about {symbol}"
    results = tavily.run(query)
    
    if not results or not isinstance(results, dict) or "results" not in results:
        return {
            "output": f"No news found for {symbol.upper()}.",
            "return_directly": True  # ⬅️ Important!
        }

    headlines = []
    for r in results["results"][:3]:
        title = r.get("title", "No Title")
        snippet = r.get("content", "No description available.")
        url = r.get("url", "#")
        headlines.append(f"- **{title}**\n  {snippet}\n  [Read more]({url})")

    summary = f"📰 **Latest News About {symbol.upper()}**:\n\n" + "\n\n".join(headlines)

    return {
        "output": summary,
        "return_directly": True  # ⬅️ This makes it direct to UI, no LLM
    }




@tool
def compare_stocks(symbol1: str, symbol2: str, days: int = 7):
    """Compare two stocks by their closing prices over the last X days and show a trend chart."""
    import pandas as pd
    import plotly.graph_objs as go
    import yfinance as yf

    s1_data = yf.Ticker(symbol1).history(period=f"{days}d")['Close']
    s2_data = yf.Ticker(symbol2).history(period=f"{days}d")['Close']

    s1_change = ((s1_data.iloc[-1] - s1_data.iloc[0]) / s1_data.iloc[0]) * 100
    s2_change = ((s2_data.iloc[-1] - s2_data.iloc[0]) / s2_data.iloc[0]) * 100

    df = pd.DataFrame({
        "Date": s1_data.index,
        symbol1.upper(): s1_data.values,
        symbol2.upper(): s2_data.values,
    })

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df[symbol1.upper()],
        mode='lines', name=symbol1.upper(),
        line=dict(color='#1f77b4')  # Blue
    ))
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df[symbol2.upper()],
        mode='lines', name=symbol2.upper(),
        line=dict(color='#ff7f0e')  # Orange
    ))

    fig.update_layout(
        title=f"{symbol1.upper()} vs {symbol2.upper()} (Last {days} Days)",
        xaxis_title="Date",
        yaxis_title="Stock Price",
        legend_title="Stock Symbol"
    )

    return {
        "output": (
            f"### 📊 {symbol1.upper()} vs {symbol2.upper()} (Last {days} Days)\n\n"
            f"- **{symbol1.upper()}** change: **{s1_change:.2f}%**\n"
            f"- **{symbol2.upper()}** change: **{s2_change:.2f}%**"
        ),
        "html": fig.to_html(full_html=False),
        "return_directly": True
    }


tools = [get_live_stock_price, get_stock_history, plot_stock_history, get_stock_news, compare_stocks]

# 3. Define Graph

class State(TypedDict):
    messages: Annotated[list, add_messages]

llm_with_tools = llm.bind_tools(tools)

def build_graph(llm_with_tools, tools):
    graph_builder = StateGraph(State)

    # Add LLM Node
    def chatbot(state: State):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    graph_builder.add_node("chatbot", chatbot)

    # Add Tool Execution Node
    class BasicToolNode:
        def __init__(self, tools):
            self.tools_by_name = {tool.name: tool for tool in tools}

        def __call__(self, inputs: dict):
            message = inputs.get("messages", [])[-1]
            outputs = []
            for tool_call in getattr(message, "tool_calls", []):
                try:
                    tool_result = self.tools_by_name[tool_call["name"]].invoke(tool_call["args"])
                except Exception as e:
                    tool_result = {
                        "output": f"⚠️ Error running tool `{tool_call['name']}`: {str(e)}",
                        "html": None,
                        "return_directly": True
                    }

                outputs.append(
                    ToolMessage(
                        content=json.dumps({
                            "output": tool_result.get("output", ""),
                            "html": tool_result.get("html"),
                            "return_directly": tool_result.get("return_directly", False)
                        }),
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                    )
                )
            return {"messages": outputs}

    tool_node = BasicToolNode(tools)
    graph_builder.add_node("tools", tool_node)

    # Routing Logic
    def route_tools(state: State):
        ai_message = state["messages"][-1]
        return "tools" if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0 else END

    def should_redirect(state: State):
        tool_response = state["messages"][-1]
        if tool_response.content:
            try:
                tool_data = json.loads(tool_response.content)
                if tool_data.get("return_directly", False):
                    return END
            except Exception as e:
                print("Failed to parse tool content:", e)
        return "chatbot"

    graph_builder.add_conditional_edges("chatbot", route_tools, {"tools": "tools", END: END})
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_conditional_edges("tools", should_redirect, {"chatbot": "chatbot", END: END})

    return graph_builder.compile()

# 4. Streamlit Interface
import streamlit.components.v1 as components


graph = build_graph(llm_with_tools, tools)

user_input = st.text_input("Ask about a stock:", placeholder="e.g. Compare TSLA and AAPL or show 7-day trend of TSLA")

if st.button("Ask") and user_input:
    response_data = {}

    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            final_message = value["messages"][-1]

            print("🔥 Final message type:", final_message.type)
            print("🧾 Final message content:\n", final_message.content)

            try:
                content_str = final_message.content.strip()
                if content_str.startswith("{") and content_str.endswith("}"):
                    # Only try to parse if it's JSON
                    response_data = json.loads(content_str)
                else:
                    response_data = {"output": content_str}
            except Exception as e:
                print("⚠️ JSON decode failed:", e)
                response_data = {"output": final_message.content}

    # 🖥️ Show response
    output_text = response_data.get("output", "").strip()
    if output_text:
        st.markdown(output_text)

    html_chart = response_data.get("html", None)
    if html_chart:
        components.html(html_chart, height=500, scrolling=True)
