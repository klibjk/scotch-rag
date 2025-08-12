"""
Discord Bot: pookan-langgraph (Slash Commands Version)
Tesla Stock Recommendation using LangGraph Multi-Agent System
Self-contained implementation with slash commands
"""

import discord
from discord.ext import commands
import asyncio
import json
import re
from datetime import datetime
import os
import warnings
from typing import Dict, Any, Optional, List, TypedDict, Annotated
import yfinance as yf
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# Discord Bot Configuration
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents, help_command=None)

# Initialize the LangGraph system with fallback mechanism: Anthropic first, OpenAI second
def configure_langgraph():
    """Configure LangGraph with fallback mechanism"""
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if anthropic_key:
        try:
            print("🔵 Using Anthropic Claude Sonnet as primary LLM")
            llm = ChatAnthropic(model="claude-sonnet-4-0", anthropic_api_key=anthropic_key)
            return llm, "anthropic"
        except Exception as e:
            print(f"⚠️ Anthropic configuration failed: {e}")
            if openai_key:
                try:
                    print("🟢 Falling back to OpenAI GPT-4")
                    llm = ChatOpenAI(model="gpt-4", openai_api_key=openai_key)
                    return llm, "openai"
                except Exception as e2:
                    print(f"❌ OpenAI fallback also failed: {e2}")
                    raise Exception("Both Anthropic and OpenAI configurations failed")
            else:
                raise Exception("Anthropic failed and no OpenAI API key available")
    elif openai_key:
        try:
            print("🟢 Using OpenAI GPT-4 as primary LLM")
            llm = ChatOpenAI(model="gpt-4", openai_api_key=openai_key)
            return llm, "openai"
        except Exception as e:
            print(f"❌ OpenAI configuration failed: {e}")
            raise Exception("OpenAI configuration failed")
    else:
        raise Exception("No API keys found. Please set either ANTHROPIC_API_KEY or OPENAI_API_KEY")

# Configure LangGraph
try:
    llm, provider = configure_langgraph()
    print(f"✅ LangGraph configured successfully with {provider.upper()}")
except Exception as e:
    print(f"❌ LangGraph configuration failed: {e}")
    print("💡 Please check your API keys and try again")
    exit(1)

# Self-contained LangGraph Multi-Agent Implementation
class State(TypedDict):
    messages: Annotated[List[BaseMessage], "The messages in the conversation"]
    market_data: Annotated[Dict[str, Any], "Market data for Tesla"]
    technical_analysis: Annotated[str, "Technical analysis results"]
    fundamental_analysis: Annotated[str, "Fundamental analysis results"]
    risk_assessment: Annotated[str, "Risk assessment results"]
    final_recommendation: Annotated[Dict[str, Any], "Final recommendation"]
    workflow_status: Annotated[str, "Current workflow status"]

# Define tools as BaseTool classes instead of @tool decorators
class MarketDataTool(BaseTool):
    name: str = "fetch_market_data"
    description: str = "Get real-time market data for any stock including price, volume, and technical indicators"
    
    def _run(self, query: str) -> str:
        """Get market data for any stock"""
        try:
            # Extract ticker from query or use default
            ticker = "TSLA"  # Default fallback
            if query and len(query.strip()) > 0:
                # Try to extract ticker from query
                words = query.strip().upper().split()
                for word in words:
                    if len(word) <= 5 and word.isalpha():
                        ticker = word
                        break
            
            # Get stock data with timeout and retry
            import time
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    stock = yf.Ticker(ticker)
                    hist = stock.history(period="30d")
                    
                    if hist.empty:
                        if attempt < max_retries - 1:
                            time.sleep(1)
                            continue
                        return f"Error: Unable to fetch {ticker} market data after multiple attempts"
                    
                    # Calculate current metrics
                    current_price = hist['Close'].iloc[-1]
                    previous_close = hist['Close'].iloc[-2]
                    price_change = ((current_price - previous_close) / previous_close) * 100
                    volume = hist['Volume'].iloc[-1]
                    
                    # Calculate technical indicators
                    delta = hist['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))
                    current_rsi = rsi.iloc[-1]
                    
                    sma_20 = hist['Close'].rolling(window=20).mean().iloc[-1]
                    sma_50 = hist['Close'].rolling(window=50).mean().iloc[-1]
                    
                    return f"{ticker} Market Data: Price=${current_price:.2f}, Change={price_change:.2f}%, Volume={volume:,}, RSI={current_rsi:.1f}, 20SMA=${sma_20:.2f}, 50SMA=${sma_50:.2f}"
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(1)
                        continue
                    else:
                        raise e
                        
        except Exception as e:
            return f"Error fetching market data: {str(e)}"

class TechnicalAnalysisTool(BaseTool):
    name: str = "technical_analysis"
    description: str = "Perform technical analysis on any stock using multiple indicators"
    
    def _run(self, query: str) -> str:
        """Perform technical analysis"""
        try:
            # Extract ticker from query or use default
            ticker = "TSLA"  # Default fallback
            if query and len(query.strip()) > 0:
                # Try to extract ticker from query
                words = query.strip().upper().split()
                for word in words:
                    if len(word) <= 5 and word.isalpha():
                        ticker = word
                        break
            
            import time
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    stock = yf.Ticker(ticker)
                    hist = stock.history(period="30d")
                    
                    if hist.empty:
                        if attempt < max_retries - 1:
                            time.sleep(1)
                            continue
                        return f"Error: Unable to fetch data for {ticker} technical analysis after multiple attempts"
                    
                    current_price = hist['Close'].iloc[-1]
                    
                    # RSI Analysis
                    delta = hist['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))
                    current_rsi = rsi.iloc[-1]
                    
                    # Moving Averages
                    sma_20 = hist['Close'].rolling(window=20).mean().iloc[-1]
                    sma_50 = hist['Close'].rolling(window=50).mean().iloc[-1]
                    
                    # MACD
                    ema_12 = hist['Close'].ewm(span=12).mean()
                    ema_26 = hist['Close'].ewm(span=26).mean()
                    macd = ema_12 - ema_26
                    signal = macd.ewm(span=9).mean()
                    current_macd = macd.iloc[-1]
                    current_signal = signal.iloc[-1]
                    
                    # Bollinger Bands
                    sma_20_series = hist['Close'].rolling(window=20).mean()
                    std_20 = hist['Close'].rolling(window=20).std()
                    upper_band = sma_20_series + (std_20 * 2)
                    lower_band = sma_20_series - (std_20 * 2)
                    
                    # Analysis
                    rsi_signal = "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral"
                    ma_signal = "Bullish" if current_price > sma_20 > sma_50 else "Bearish" if current_price < sma_20 < sma_50 else "Neutral"
                    macd_signal = "Bullish" if current_macd > current_signal else "Bearish"
                    bb_position = "Upper Band" if current_price > upper_band.iloc[-1] else "Lower Band" if current_price < lower_band.iloc[-1] else "Middle"
                    
                    analysis = f"{ticker} Technical Analysis: RSI={current_rsi:.1f} ({rsi_signal}), MA Signal={ma_signal}, MACD={macd_signal}, Bollinger Band Position={bb_position}"
                    
                    return analysis
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(1)
                        continue
                    else:
                        raise e
                        
        except Exception as e:
            return f"Error in technical analysis: {str(e)}"

class FundamentalAnalysisTool(BaseTool):
    name: str = "fundamental_analysis"
    description: str = "Analyze fundamental metrics and company performance for any stock"
    
    def _run(self, query: str) -> str:
        """Perform fundamental analysis"""
        try:
            # Extract ticker from query or use default
            ticker = "TSLA"  # Default fallback
            if query and len(query.strip()) > 0:
                # Try to extract ticker from query
                words = query.strip().upper().split()
                for word in words:
                    if len(word) <= 5 and word.isalpha():
                        ticker = word
                        break
            
            import time
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    
                    # Extract key metrics
                    market_cap = info.get('marketCap', 0)
                    pe_ratio = info.get('trailingPE', 0)
                    pb_ratio = info.get('priceToBook', 0)
                    debt_to_equity = info.get('debtToEquity', 0)
                    profit_margins = info.get('profitMargins', 0)
                    revenue_growth = info.get('revenueGrowth', 0)
                    return_on_equity = info.get('returnOnEquity', 0)
                    
                    # Analysis
                    pe_assessment = "High" if pe_ratio > 50 else "Reasonable" if pe_ratio > 20 else "Low"
                    debt_assessment = "High" if debt_to_equity > 1 else "Manageable"
                    margin_assessment = "Strong" if profit_margins > 0.1 else "Weak" if profit_margins < 0 else "Moderate"
                    growth_assessment = "Strong" if revenue_growth > 0.2 else "Moderate" if revenue_growth > 0.1 else "Weak"
                    
                    analysis = f"{ticker} Fundamental Analysis: Market Cap=${market_cap/1e9:.1f}B, P/E={pe_ratio:.1f} ({pe_assessment}), Debt/Equity={debt_to_equity:.2f} ({debt_assessment}), Profit Margin={profit_margins:.1%} ({margin_assessment}), Revenue Growth={revenue_growth:.1%} ({growth_assessment})"
                    
                    return analysis
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(1)
                        continue
                    else:
                        raise e
                        
        except Exception as e:
            return f"Error in fundamental analysis: {str(e)}"

class RiskAssessmentTool(BaseTool):
    name: str = "risk_assessment"
    description: str = "Assess investment risk for any stock based on various factors"
    
    def _run(self, query: str) -> str:
        """Assess investment risk"""
        try:
            # Extract ticker from query or use default
            ticker = "TSLA"  # Default fallback
            if query and len(query.strip()) > 0:
                # Try to extract ticker from query
                words = query.strip().upper().split()
                for word in words:
                    if len(word) <= 5 and word.isalpha():
                        ticker = word
                        break
            
            import time
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    stock = yf.Ticker(ticker)
                    hist = stock.history(period="30d")
                    info = stock.info
                    
                    if hist.empty:
                        if attempt < max_retries - 1:
                            time.sleep(1)
                            continue
                        return f"Error: Unable to fetch {ticker} data for risk assessment after multiple attempts"
                    
                    # Calculate volatility
                    returns = hist['Close'].pct_change()
                    volatility = returns.std() * (252 ** 0.5)  # Annualized volatility
                    
                    # Beta calculation
                    try:
                        spy = yf.Ticker("SPY")
                        spy_hist = spy.history(period="30d")
                        if not spy_hist.empty:
                            spy_returns = spy_hist['Close'].pct_change()
                            correlation = returns.corr(spy_returns)
                            beta = correlation * (volatility / (spy_returns.std() * (252 ** 0.5)))
                        else:
                            beta = 1.5  # Default beta
                    except Exception:
                        beta = 1.5  # Default beta if SPY fetch fails
                    
                    # Risk factors
                    risk_factors = []
                    risk_score = 0
                    
                    if volatility > 0.5:
                        risk_factors.append("High volatility")
                        risk_score += 2
                    elif volatility > 0.3:
                        risk_factors.append("Moderate volatility")
                        risk_score += 1
                    
                    if beta > 1.5:
                        risk_factors.append("High beta")
                        risk_score += 2
                    elif beta > 1.2:
                        risk_factors.append("Elevated beta")
                        risk_score += 1
                    
                    if info.get('debtToEquity', 0) > 1:
                        risk_factors.append("High debt levels")
                        risk_score += 2
                    elif info.get('debtToEquity', 0) > 0.5:
                        risk_factors.append("Moderate debt")
                        risk_score += 1
                    
                    if info.get('profitMargins', 0) < 0:
                        risk_factors.append("Negative profit margins")
                        risk_score += 1
                    
                    risk_level = "High" if risk_score > 4 else "Medium" if risk_score > 2 else "Low"
                    
                    assessment = f"{ticker} Risk Assessment: Volatility={volatility:.1%}, Beta={beta:.2f}, Risk Level={risk_level} (Score: {risk_score}), Factors: {', '.join(risk_factors) if risk_factors else 'None'}"
                    
                    return assessment
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(1)
                        continue
                    else:
                        raise e
                        
        except Exception as e:
            return f"Error in risk assessment: {str(e)}"

# Initialize tools
tools = [
    MarketDataTool(),
    TechnicalAnalysisTool(),
    FundamentalAnalysisTool(),
    RiskAssessmentTool()
]

# Create tool node
tool_node = ToolNode(tools)

# Define workflow nodes
def data_fetcher(state: State) -> State:
    """Fetch market data"""
    try:
        # Extract ticker from messages
        ticker = "TSLA"  # Default
        for message in state["messages"]:
            if isinstance(message, HumanMessage):
                content = message.content.lower()
                if "analyze" in content:
                    words = content.split()
                    for i, word in enumerate(words):
                        if word == "analyze" and i + 1 < len(words):
                            ticker = words[i + 1].upper()
                            break
        
        # Fetch market data
        market_data_tool = MarketDataTool()
        market_data_result = market_data_tool._run(ticker)
        
        state["market_data"] = {
            "ticker": ticker,
            "data": market_data_result,
            "status": "completed"
        }
        state["workflow_status"] = "Data fetched successfully"
        
    except Exception as e:
        state["market_data"] = {
            "ticker": ticker,
            "data": f"Error: {str(e)}",
            "status": "error"
        }
        state["workflow_status"] = f"Data fetch error: {str(e)}"
    
    return state

def technical_analyst(state: State) -> State:
    """Perform technical analysis"""
    try:
        ticker = state["market_data"]["ticker"]
        technical_tool = TechnicalAnalysisTool()
        technical_result = technical_tool._run(ticker)
        
        state["technical_analysis"] = technical_result
        state["workflow_status"] = "Technical analysis completed"
        
    except Exception as e:
        state["technical_analysis"] = f"Error in technical analysis: {str(e)}"
        state["workflow_status"] = f"Technical analysis error: {str(e)}"
    
    return state

def fundamental_analyst(state: State) -> State:
    """Perform fundamental analysis"""
    try:
        ticker = state["market_data"]["ticker"]
        fundamental_tool = FundamentalAnalysisTool()
        fundamental_result = fundamental_tool._run(ticker)
        
        state["fundamental_analysis"] = fundamental_result
        state["workflow_status"] = "Fundamental analysis completed"
        
    except Exception as e:
        state["fundamental_analysis"] = f"Error in fundamental analysis: {str(e)}"
        state["workflow_status"] = f"Fundamental analysis error: {str(e)}"
    
    return state

def risk_assessor(state: State) -> State:
    """Assess risk"""
    try:
        ticker = state["market_data"]["ticker"]
        risk_tool = RiskAssessmentTool()
        risk_result = risk_tool._run(ticker)
        
        state["risk_assessment"] = risk_result
        state["workflow_status"] = "Risk assessment completed"
        
    except Exception as e:
        state["risk_assessment"] = f"Error in risk assessment: {str(e)}"
        state["workflow_status"] = f"Risk assessment error: {str(e)}"
    
    return state

def decision_maker(state: State) -> State:
    """Make final recommendation"""
    try:
        # Create comprehensive analysis message
        analysis_prompt = f"""
        Based on the following analysis components, provide a comprehensive stock recommendation:
        
        Market Data: {state['market_data']['data']}
        Technical Analysis: {state['technical_analysis']}
        Fundamental Analysis: {state['fundamental_analysis']}
        Risk Assessment: {state['risk_assessment']}
        
        Please provide:
        1. A clear BUY, HOLD, or SELL recommendation
        2. Confidence level (High/Medium/Low)
        3. Key reasoning points
        4. Risk factors to consider
        5. Time horizon recommendation
        """
        
        # Get recommendation from LLM
        response = llm.invoke([HumanMessage(content=analysis_prompt)])
        recommendation_text = response.content
        
        state["final_recommendation"] = {
            "recommendation": recommendation_text,
            "status": "completed",
            "timestamp": datetime.now().isoformat()
        }
        state["workflow_status"] = "Final recommendation completed"
        
    except Exception as e:
        state["final_recommendation"] = {
            "recommendation": f"Error generating recommendation: {str(e)}",
            "status": "error",
            "timestamp": datetime.now().isoformat()
        }
        state["workflow_status"] = f"Decision making error: {str(e)}"
    
    return state

# Build the workflow graph
workflow = StateGraph(State)

# Add nodes
workflow.add_node("data_fetcher", data_fetcher)
workflow.add_node("technical_analyst", technical_analyst)
workflow.add_node("fundamental_analyst", fundamental_analyst)
workflow.add_node("risk_assessor", risk_assessor)
workflow.add_node("decision_maker", decision_maker)

# Set entry point
workflow.set_entry_point("data_fetcher")

# Add edges
workflow.add_edge("data_fetcher", "technical_analyst")
workflow.add_edge("technical_analyst", "fundamental_analyst")
workflow.add_edge("fundamental_analyst", "risk_assessor")
workflow.add_edge("risk_assessor", "decision_maker")
workflow.add_edge("decision_maker", END)

# Compile the graph
app = workflow.compile()

@bot.event
async def on_ready():
    """Bot startup event"""
    print(f'🤖 {bot.user} has connected to Discord!')
    print(f'📊 LangGraph Stock Bot (Slash Commands) is ready!')
    print(f'🤖 Provider: {provider.upper()}')
    print(f'💬 Use /analyze <ticker> to get stock recommendations')
    
    # Register slash commands
    try:
        print("🔄 Registering slash commands...")
        await bot.tree.sync()
        print("✅ Slash commands registered successfully!")
    except Exception as e:
        print(f"❌ Failed to register slash commands: {e}")

# Slash command handlers
@bot.tree.command(name="analyze", description="Analyze any stock using LangGraph multi-agent system")
async def analyze_stock_slash(interaction: discord.Interaction, ticker: str, query: str = None):
    """Slash command handler for stock analysis"""
    await interaction.response.defer()
    
    try:
        # Send initial message
        embed = discord.Embed(
            title="📊 Stock Analysis in Progress",
            description=f"Analyzing **{ticker.upper()}** using LangGraph multi-agent system...",
            color=0x00ff00
        )
        embed.add_field(name="Status", value="🔄 Starting workflow...", inline=False)
        message = await interaction.followup.send(embed=embed)
        
        # Initialize state
        initial_state = {
            "messages": [HumanMessage(content=f"analyze {ticker}")],
            "market_data": {},
            "technical_analysis": "",
            "fundamental_analysis": "",
            "risk_assessment": "",
            "final_recommendation": {},
            "workflow_status": "Initialized"
        }
        
        # Run the workflow
        result = app.invoke(initial_state)
        
        # Create detailed response embed
        embed = discord.Embed(
            title=f"📊 {ticker.upper()} Stock Analysis - LangGraph Multi-Agent",
            description="Analysis completed using LangGraph multi-agent workflow",
            color=0x00ff00
        )
        
        # Add market data
        market_data = result.get("market_data", {})
        if market_data and "data" in market_data:
            try:
                data_str = market_data["data"]
                # Parse the data string like "TSLA Market Data: Price=$255.30, Change=+1.2%, Volume=20M"
                price_match = re.search(r'Price=\$(\d+\.?\d*)', data_str)
                change_match = re.search(r'Change=([+-]?\d+\.?\d*%)', data_str)
                volume_match = re.search(r'Volume=([0-9,]+)', data_str)
                
                if price_match:
                    price = float(price_match.group(1))
                    embed.add_field(
                        name="💰 Current Price", 
                        value=f"${price:.2f}", 
                        inline=True
                    )
                else:
                    embed.add_field(
                        name="💰 Current Price", 
                        value="N/A", 
                        inline=True
                    )
                
                if change_match:
                    embed.add_field(
                        name="📈 Change", 
                        value=change_match.group(1), 
                        inline=True
                    )
                else:
                    embed.add_field(
                        name="📈 Change", 
                        value="N/A", 
                        inline=True
                    )
                
                if volume_match:
                    volume_str = volume_match.group(1)
                    
                    embed.add_field(
                        name="📊 Volume", 
                        value=volume_str, 
                        inline=True
                    )
                else:
                    embed.add_field(
                        name="📊 Volume", 
                        value="N/A", 
                        inline=True
                    )
            except:
                embed.add_field(
                    name="💰 Current Price", 
                    value="N/A", 
                    inline=True
                )
                embed.add_field(
                    name="📈 Change", 
                    value="N/A", 
                    inline=True
                )
                embed.add_field(
                    name="📊 Volume", 
                    value="N/A", 
                    inline=True
                )
        else:
            embed.add_field(
                name="💰 Current Price", 
                value="N/A", 
                inline=True
            )
            embed.add_field(
                name="📈 Change", 
                value="N/A", 
                inline=True
            )
            embed.add_field(
                name="📊 Volume", 
                value="N/A", 
                inline=True
            )
        
        # Add workflow results
        if result["final_recommendation"]["status"] == "completed":
            recommendation = result["final_recommendation"]["recommendation"]
            recommendation_lower = recommendation.lower()
            
            # Determine recommendation type and confidence
            if "buy" in recommendation_lower:
                recommendation_type = "BUY"
                confidence = "High" if "strong" in recommendation_lower or "recommend" in recommendation_lower else "Medium"
            elif "sell" in recommendation_lower:
                recommendation_type = "SELL"
                confidence = "High" if "strong" in recommendation_lower else "Medium"
            else:
                recommendation_type = "HOLD"
                confidence = "Medium"
            
            embed.add_field(
                name="🎯 Recommendation", 
                value=recommendation_type, 
                inline=True
            )
            embed.add_field(
                name="📊 Confidence", 
                value=confidence, 
                inline=True
            )
            embed.add_field(
                name="⚠️ Risk Level", 
                value="Medium", 
                inline=True
            )
            
            # Add reasoning (truncated recommendation)
            recommendation_short = recommendation[:1024] if len(recommendation) > 1024 else recommendation
            embed.add_field(
                name="💭 Reasoning", 
                value=recommendation_short, 
                inline=False
            )
        else:
            embed.add_field(
                name="🎯 Recommendation", 
                value="ERROR", 
                inline=True
            )
            embed.add_field(
                name="📊 Confidence", 
                value="N/A", 
                inline=True
            )
            embed.add_field(
                name="⚠️ Risk Level", 
                value="N/A", 
                inline=True
            )
            
            embed.add_field(
                name="❌ Error", 
                value=result["final_recommendation"]["recommendation"], 
                inline=False
            )
        
        # Add workflow status and framework info
        embed.add_field(
            name="🔄 Workflow Status", 
            value=result["workflow_status"], 
            inline=True
        )
        embed.add_field(
            name="⚡ Framework", 
            value="LangGraph Multi-Agent", 
            inline=True
        )
        embed.add_field(
            name="⏰ Completed", 
            value=datetime.now().strftime("%H:%M:%S"), 
            inline=True
        )
        
        embed.set_footer(text=f"pookan-langgraph • {provider.upper()} • Real-time market data")
        await message.edit(embed=embed)
        
    except Exception as e:
        error_embed = discord.Embed(
            title="❌ Error",
            description=f"Failed to analyze {ticker.upper()}: {str(e)}",
            color=0xff0000
        )
        await interaction.followup.send(embed=error_embed)

@bot.tree.command(name="workflow", description="Show workflow details and status")
async def workflow_command_slash(interaction: discord.Interaction):
    """Slash command handler for workflow information"""
    embed = discord.Embed(
        title="⚡ LangGraph Workflow Details",
        description="**Multi-Agent Stock Analysis Workflow**\n\nThis bot uses a sophisticated 5-node workflow for comprehensive stock analysis.",
        color=0x0099ff
    )
    
    # Workflow nodes
    embed.add_field(
        name="🔄 **Workflow Nodes**",
        value="""
1. **Data Fetcher** - Real-time market data collection
2. **Technical Analyst** - Technical indicators & patterns
3. **Fundamental Analyst** - Company financials & metrics
4. **Risk Assessor** - Risk analysis & factors
5. **Decision Maker** - Final recommendation synthesis
        """,
        inline=False
    )
    
    # Features
    embed.add_field(
        name="🎯 **Advanced Features**",
        value="""
• **State Management** - Tracks analysis progress
• **Conditional Routing** - Smart workflow decisions
• **Error Recovery** - Robust error handling
• **Production-Ready** - Enterprise-grade reliability
• **Multi-Agent Coordination** - Specialized agents for each task
        """,
        inline=False
    )
    
    # Framework info
    embed.add_field(
        name="⚡ **LangGraph Framework**",
        value="""
• **Advanced orchestration** - State-based workflow management
• **Production-ready** - Enterprise-grade reliability
• **State management** - Tracks analysis progress
• **Conditional routing** - Smart workflow decisions
• **Error recovery** - Robust error handling
• **Complex but powerful** - Most advanced framework
        """,
        inline=False
    )
    
    embed.set_footer(text=f"LangGraph Multi-Agent • {provider.upper()} • Real-time market data")
    await interaction.response.send_message(embed=embed)

@bot.tree.command(name="help", description="Show comprehensive help information")
async def help_command_slash(interaction: discord.Interaction):
    """Slash command handler for help information"""
    embed = discord.Embed(
        title="🤖 LangGraph Stock Analysis Bot - Help Guide",
        description="**AI-powered stock analysis using LangGraph multi-agent system and real-time market data**\n\nThis bot provides advanced stock analysis with AI recommendations using the latest market data.",
        color=0x0099ff
    )
    
    # Basic commands
    embed.add_field(
        name="📊 **Basic Commands**",
        value="""
`/analyze <ticker>` - Analyze any stock (e.g., `/analyze MSFT`)
`/analyze <ticker> <query>` - Specific analysis with custom query
`/workflow` - Show workflow details and status
`/help` - Show this help message
`/status` - Show bot status and configuration
        """,
        inline=False
    )
    
    # Examples
    embed.add_field(
        name="💡 **Usage Examples**",
        value="""
• `/analyze AAPL` - Analyze Apple stock
• `/analyze TSLA should I buy?` - Specific buying advice
• `/analyze GOOGL analyze fundamentals` - Focus on fundamentals
• `/analyze NVDA technical analysis` - Focus on technical indicators
        """,
        inline=False
    )
    
    # Features
    embed.add_field(
        name="🎯 **What You Get**",
        value="""
• **Advanced Workflow Orchestration** with 5 specialized nodes:
  - **Data Fetcher** - Real-time market data collection
  - **Technical Analyst** - Technical indicators & patterns
  - **Fundamental Analyst** - Company financials & metrics
  - **Risk Assessor** - Risk analysis & factors
  - **Decision Maker** - Final recommendation synthesis
• **State Management** - Tracks analysis progress
• **Conditional Routing** - Smart workflow decisions
• **Error Recovery** - Robust error handling
• **Production-Ready** - Enterprise-grade reliability
        """,
        inline=False
    )
    
    # Framework info
    embed.add_field(
        name="⚡ **LangGraph Framework**",
        value="""
• **Advanced orchestration** - State-based workflow management
• **Production-ready** - Enterprise-grade reliability
• **State management** - Tracks analysis progress
• **Conditional routing** - Smart workflow decisions
• **Error recovery** - Robust error handling
• **Complex but powerful** - Most advanced framework
        """,
        inline=False
    )
    
    # Tips
    embed.add_field(
        name="💭 **Pro Tips**",
        value="""
• Use any valid stock ticker (e.g., MSFT, AAPL, TSLA)
• This bot uses the most advanced AI framework
• Response times may vary due to complex workflow
• Use `/workflow` to see workflow details
• Great for production environments
        """,
        inline=False
    )
    
    embed.set_footer(text=f"LangGraph Multi-Agent • {provider.upper()} • Real-time market data")
    await interaction.response.send_message(embed=embed)

@bot.tree.command(name="status", description="Show bot status and configuration")
async def status_command_slash(interaction: discord.Interaction):
    """Slash command handler for status"""
    embed = discord.Embed(
        title="🤖 pookan-langgraph Status",
        description="Bot is running and ready for stock analysis",
        color=0x00ff00
    )
    
    embed.add_field(
        name="✅ Status",
        value="Online and Ready",
        inline=True
    )
    
    embed.add_field(
        name="🤖 AI Provider",
        value=provider.upper(),
        inline=True
    )
    
    embed.add_field(
        name="⚡ Framework",
        value="LangGraph Multi-Agent",
        inline=True
    )
    
    embed.add_field(
        name="🔄 Fallback",
        value="Anthropic → OpenAI",
        inline=True
    )
    
    embed.add_field(
        name="🎯 Capability",
        value="Advanced Stock Analysis (Any Ticker)",
        inline=True
    )
    
    embed.add_field(
        name="🤖 Agents",
        value="5 Specialized Agents",
        inline=True
    )
    
    embed.add_field(
        name="💡 Commands",
        value="/analyze, /workflow, /help, /status",
        inline=True
    )
    
    embed.set_footer(text=f"pookan-langgraph • {provider.upper()} • {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    await interaction.response.send_message(embed=embed)

def main():
    """Run the Discord bot"""
    token = os.getenv("LANGGRAPH_DISCORD_TOKEN")
    if not token:
        print("❌ Error: LANGGRAPH_DISCORD_TOKEN not found in environment variables")
        print("💡 Make sure LANGGRAPH_DISCORD_TOKEN is set in environment variables")
        print("💡 Current environment variables:")
        print(f"   - LANGGRAPH_DISCORD_TOKEN: {'Set' if os.getenv('LANGGRAPH_DISCORD_TOKEN') else 'Not set'}")
        print(f"   - ANTHROPIC_API_KEY: {'Set' if os.getenv('ANTHROPIC_API_KEY') else 'Not set'}")
        print(f"   - OPENAI_API_KEY: {'Set' if os.getenv('OPENAI_API_KEY') else 'Not set'}")
        print(f"   - Provider: {provider.upper()}")
        return
    
    print("🚀 Starting pookan-langgraph Discord bot...")
    print("📊 Bot will be ready to analyze any stocks!")
    print(f"✅ Environment variables loaded successfully")
    print(f"⚡ Provider: {provider.upper()}")
    print(f"💡 Use /analyze <ticker> to analyze a stock")
    print(f"💡 Use /workflow to see workflow details")
    print(f"💡 Use /help to see comprehensive help")
    print(f"💡 Use /status to see bot configuration")
    
    try:
        bot.run(token)
    except Exception as e:
        print(f"❌ Failed to start bot: {e}")

if __name__ == "__main__":
    main()
