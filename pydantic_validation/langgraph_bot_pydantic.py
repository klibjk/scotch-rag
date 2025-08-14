"""
Discord Bot: pookan-langgraph (Slash Commands Version) - WITH PYDANTIC VALIDATION
Tesla Stock Recommendation using LangGraph Multi-Agent System
Self-contained implementation with slash commands and Pydantic validation
"""

import discord
from discord.ext import commands
import asyncio
import json
import re
from datetime import datetime
import os
import warnings
from typing import Dict, Any, Optional, List, Annotated, Literal
from typing_extensions import TypedDict
import yfinance as yf
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain.tools import BaseTool
from pydantic import BaseModel, Field, validator, ValidationError, field_validator
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

# Pydantic Models for Validation
class WorkflowStateInput(BaseModel):
    """Validate workflow state input"""
    ticker: str = Field(..., min_length=1, max_length=10, description="Stock ticker symbol")
    query: Optional[str] = Field(None, max_length=500, description="Optional analysis query")
    
    @field_validator('ticker')
    @classmethod
    def validate_ticker(cls, v):
        """Validate ticker format"""
        if not v.isalpha():
            raise ValueError('Ticker must contain only letters')
        if len(v) > 5:
            raise ValueError('Ticker must be 5 characters or less')
        return v.upper()

class MarketDataState(BaseModel):
    """Validate market data state"""
    ticker: str
    data: str = Field(..., min_length=1, description="Market data string")
    status: str = Field(..., pattern="^(completed|error)$", description="Data fetch status")

class Step(TypedDict):
    """Individual thinking step"""
    phase: Literal["thought", "action", "observation"]
    content: str
    payload: Dict[str, Any] | None

class WorkflowState(BaseModel):
    """Validate complete workflow state"""
    messages: List[BaseMessage]
    steps: List[Step]
    market_data: Dict[str, Any]
    technical_analysis: str
    fundamental_analysis: str
    risk_assessment: str
    final_recommendation: Dict[str, Any]
    workflow_status: str

# Initialize the LangGraph system with fallback mechanism
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

# Self-contained LangGraph Multi-Agent Implementation with Pydantic Validation
class State(TypedDict):
    messages: Annotated[List[BaseMessage], "The messages in the conversation"]
    steps: Annotated[List[Step], "Execution log with thinking steps"]
    market_data: Annotated[Dict[str, Any], "Market data for stock"]
    technical_analysis: Annotated[str, "Technical analysis results"]
    fundamental_analysis: Annotated[str, "Fundamental analysis results"]
    risk_assessment: Annotated[str, "Risk assessment results"]
    final_recommendation: Annotated[Dict[str, Any], "Final recommendation"]
    workflow_status: Annotated[str, "Current workflow status"]

# Define tools as BaseTool classes with Pydantic validation
class MarketDataTool(BaseTool):
    name: str = "fetch_market_data"
    description: str = "Get real-time market data for any stock including price, volume, and technical indicators"
    
    def _run(self, query: str) -> str:
        """Get market data for any stock with Pydantic validation"""
        try:
            # Validate input
            if not query or not query.strip():
                raise ValidationError("Query cannot be empty")
            
            # Extract ticker from query
            ticker = "TSLA"  # Default fallback
            words = query.strip().upper().split()
            for word in words:
                if len(word) <= 5 and word.isalpha():
                    ticker = word
                    break
            
            # Validate ticker
            validated_ticker = WorkflowStateInput(ticker=ticker)
            ticker = validated_ticker.ticker
            
            # Get stock data with timeout and retry
            import time
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    stock = yf.Ticker(ticker)
                    hist = stock.history(period="60d")  # Increased to 60 days for 50-day SMA
                    
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
                    
                    # Handle NaN values and validate data before returning
                    if pd.isna(current_rsi) or pd.isna(sma_20) or pd.isna(sma_50):
                        return f"Error: Insufficient data for {ticker} - need at least 50 days of trading data"
                    
                    if current_price <= 0:
                        return f"Error: Invalid price data for {ticker} - price must be positive"
                    
                    if volume <= 0:
                        return f"Error: Invalid volume data for {ticker} - volume must be positive"
                    
                    if current_rsi < 0 or current_rsi > 100:
                        return f"Error: Invalid RSI value for {ticker} - RSI must be between 0 and 100"
                    
                    if sma_20 <= 0 or sma_50 <= 0:
                        return f"Error: Invalid moving average data for {ticker} - SMAs must be positive"
                    
                    return f"{ticker} Market Data: Price=${current_price:.2f}, Change={price_change:.2f}%, Volume={volume:,}, RSI={current_rsi:.1f}, 20SMA=${sma_20:.2f}, 50SMA=${sma_50:.2f}"
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(1)
                        continue
                    else:
                        raise e
                        
        except ValidationError as e:
            return f"Validation error: {str(e)}"
        except Exception as e:
            return f"Error fetching market data: {str(e)}"

# Initialize tools
tools = [MarketDataTool()]

# Create tool node
tool_node = ToolNode(tools)

# Define workflow nodes with Pydantic validation
def data_fetcher(state: State) -> State:
    """Fetch market data with Pydantic validation"""
    # Initialize steps if not present
    state["steps"] = state.get("steps", [])
    
    # Add thinking steps
    state["steps"].append({
        "phase": "thought",
        "content": "Need to extract ticker from user message and fetch market data",
        "payload": None
    })
    
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
        
        state["steps"].append({
            "phase": "observation",
            "content": f"Extracted ticker: {ticker}",
            "payload": {"ticker": ticker}
        })
        
        # Validate ticker with Pydantic
        state["steps"].append({
            "phase": "thought",
            "content": "Validating ticker format with Pydantic",
            "payload": None
        })
        
        validated_input = WorkflowStateInput(ticker=ticker)
        ticker = validated_input.ticker
        
        state["steps"].append({
            "phase": "observation",
            "content": f"Ticker validation passed: {ticker}",
            "payload": {"validated_ticker": ticker}
        })
        
        # Fetch market data
        state["steps"].append({
            "phase": "action",
            "content": "fetch_market_data",
            "payload": {"ticker": ticker}
        })
        
        market_data_tool = MarketDataTool()
        market_data_result = market_data_tool._run(ticker)
        
        state["steps"].append({
            "phase": "observation",
            "content": f"Market data fetched successfully",
            "payload": {"result": market_data_result}
        })
        
        # Validate market data state
        state["steps"].append({
            "phase": "thought",
            "content": "Validating market data structure",
            "payload": None
        })
        
        try:
            validated_market_data = MarketDataState(
                ticker=ticker,
                data=market_data_result,
                status="completed" if "Error" not in market_data_result else "error"
            )
            
            state["steps"].append({
                "phase": "observation",
                "content": "Market data validation passed",
                "payload": {"status": validated_market_data.status}
            })
            
        except ValidationError as e:
            state["steps"].append({
                "phase": "observation",
                "content": f"Market data validation failed: {str(e)}",
                "payload": {"error": str(e)}
            })
            validated_market_data = MarketDataState(
                ticker=ticker,
                data=f"Error: {str(e)}",
                status="error"
            )
        
        state["market_data"] = {
            "ticker": validated_market_data.ticker,
            "data": validated_market_data.data,
            "status": validated_market_data.status
        }
        state["workflow_status"] = "Data fetched successfully"
        
        state["steps"].append({
            "phase": "thought",
            "content": "Data fetcher completed successfully, ready for technical analysis",
            "payload": None
        })
        
    except ValidationError as e:
        state["steps"].append({
            "phase": "observation",
            "content": f"Data fetcher validation error: {str(e)}",
            "payload": {"error": str(e)}
        })
        state["market_data"] = {
            "ticker": ticker,
            "data": f"Validation error: {str(e)}",
            "status": "error"
        }
        state["workflow_status"] = f"Data fetch validation error: {str(e)}"
    except Exception as e:
        state["steps"].append({
            "phase": "observation",
            "content": f"Data fetcher error: {str(e)}",
            "payload": {"error": str(e)}
        })
        state["market_data"] = {
            "ticker": ticker,
            "data": f"Error: {str(e)}",
            "status": "error"
        }
        state["workflow_status"] = f"Data fetch error: {str(e)}"
    
    return state

def technical_analyst(state: State) -> State:
    """Perform technical analysis with Pydantic validation"""
    # Initialize steps if not present
    state["steps"] = state.get("steps", [])
    
    # Add thinking steps
    state["steps"].append({
        "phase": "thought",
        "content": "Need to perform technical analysis using market data",
        "payload": None
    })
    
    try:
        # Validate market data exists
        state["steps"].append({
            "phase": "thought",
            "content": "Checking if market data is available",
            "payload": None
        })
        
        if not state["market_data"] or "ticker" not in state["market_data"]:
            raise ValidationError("Market data not available")
        
        ticker = state["market_data"]["ticker"]
        
        state["steps"].append({
            "phase": "observation",
            "content": f"Processing ticker: {ticker}",
            "payload": {"ticker": ticker}
        })
        
        validated_ticker = WorkflowStateInput(ticker=ticker)
        ticker = validated_ticker.ticker
        
        state["steps"].append({
            "phase": "observation",
            "content": f"Ticker validation passed: {ticker}",
            "payload": {"validated_ticker": ticker}
        })
        
        # Get market data for analysis
        market_data = state["market_data"]["data"]
        
        state["steps"].append({
            "phase": "observation",
            "content": f"Using market data for technical analysis",
            "payload": {"market_data": market_data}
        })
        
        # Create technical analysis prompt for LLM
        state["steps"].append({
            "phase": "thought",
            "content": "Creating technical analysis prompt for LLM",
            "payload": None
        })
        
        technical_prompt = f"""
        As a technical analyst, analyze the following market data for {ticker} stock:
        
        Market Data: {market_data}
        
        Please provide a comprehensive technical analysis including:
        1. RSI interpretation (overbought/oversold/neutral)
        2. Moving average signals (bullish/bearish/neutral)
        3. Price action analysis
        4. Support and resistance levels
        5. Technical indicators summary
        6. Short-term price outlook
        
        Format your response as a structured technical analysis.
        """
        
        state["steps"].append({
            "phase": "action",
            "content": "technical_analysis_llm",
            "payload": {"prompt": technical_prompt}
        })
        
        response = llm.invoke([HumanMessage(content=technical_prompt)])
        technical_result = response.content
        
        state["steps"].append({
            "phase": "observation",
            "content": f"LLM technical analysis completed",
            "payload": {"result": technical_result[:200] + "..." if len(technical_result) > 200 else technical_result}
        })
        
        state["technical_analysis"] = technical_result
        state["workflow_status"] = "Technical analysis completed"
        
        state["steps"].append({
            "phase": "thought",
            "content": "Technical analysis completed successfully, ready for fundamental analysis",
            "payload": None
        })
        
    except ValidationError as e:
        state["steps"].append({
            "phase": "observation",
            "content": f"Technical analyst validation error: {str(e)}",
            "payload": {"error": str(e)}
        })
        state["technical_analysis"] = f"Validation error: {str(e)}"
        state["workflow_status"] = f"Technical analysis validation error: {str(e)}"
    except Exception as e:
        state["steps"].append({
            "phase": "observation",
            "content": f"Technical analyst error: {str(e)}",
            "payload": {"error": str(e)}
        })
        state["technical_analysis"] = f"Error in technical analysis: {str(e)}"
        state["workflow_status"] = f"Technical analysis error: {str(e)}"
    
    return state

def fundamental_analyst(state: State) -> State:
    """Perform fundamental analysis with Pydantic validation"""
    # Initialize steps if not present
    state["steps"] = state.get("steps", [])
    
    # Add thinking steps
    state["steps"].append({
        "phase": "thought",
        "content": "Need to perform fundamental analysis using market data",
        "payload": None
    })
    
    try:
        # Validate market data exists
        state["steps"].append({
            "phase": "thought",
            "content": "Checking if market data is available",
            "payload": None
        })
        
        if not state["market_data"] or "ticker" not in state["market_data"]:
            raise ValidationError("Market data not available")
        
        ticker = state["market_data"]["ticker"]
        
        state["steps"].append({
            "phase": "observation",
            "content": f"Processing ticker: {ticker}",
            "payload": {"ticker": ticker}
        })
        
        validated_ticker = WorkflowStateInput(ticker=ticker)
        ticker = validated_ticker.ticker
        
        state["steps"].append({
            "phase": "observation",
            "content": f"Ticker validation passed: {ticker}",
            "payload": {"validated_ticker": ticker}
        })
        
        # Get market data for analysis
        market_data = state["market_data"]["data"]
        
        state["steps"].append({
            "phase": "observation",
            "content": f"Using market data for fundamental analysis",
            "payload": {"market_data": market_data}
        })
        
        # Create fundamental analysis prompt for LLM
        state["steps"].append({
            "phase": "thought",
            "content": "Creating fundamental analysis prompt for LLM",
            "payload": None
        })
        
        fundamental_prompt = f"""
        As a fundamental analyst, analyze the following market data for {ticker} stock:
        
        Market Data: {market_data}
        
        Please provide a comprehensive fundamental analysis including:
        1. Company valuation assessment
        2. Financial health indicators
        3. Growth prospects analysis
        4. Industry position and competitive advantages
        5. Risk factors assessment
        6. Long-term investment outlook
        
        Format your response as a structured fundamental analysis.
        """
        
        state["steps"].append({
            "phase": "action",
            "content": "fundamental_analysis_llm",
            "payload": {"prompt": fundamental_prompt}
        })
        
        response = llm.invoke([HumanMessage(content=fundamental_prompt)])
        fundamental_result = response.content
        
        state["steps"].append({
            "phase": "observation",
            "content": f"LLM fundamental analysis completed",
            "payload": {"result": fundamental_result[:200] + "..." if len(fundamental_result) > 200 else fundamental_result}
        })
        
        state["fundamental_analysis"] = fundamental_result
        state["workflow_status"] = "Fundamental analysis completed"
        
        state["steps"].append({
            "phase": "thought",
            "content": "Fundamental analysis completed successfully, ready for risk assessment",
            "payload": None
        })
        
    except ValidationError as e:
        state["steps"].append({
            "phase": "observation",
            "content": f"Fundamental analyst validation error: {str(e)}",
            "payload": {"error": str(e)}
        })
        state["fundamental_analysis"] = f"Validation error: {str(e)}"
        state["workflow_status"] = f"Fundamental analysis validation error: {str(e)}"
    except Exception as e:
        state["steps"].append({
            "phase": "observation",
            "content": f"Fundamental analyst error: {str(e)}",
            "payload": {"error": str(e)}
        })
        state["fundamental_analysis"] = f"Error in fundamental analysis: {str(e)}"
        state["workflow_status"] = f"Fundamental analysis error: {str(e)}"
    
    return state

def risk_assessor(state: State) -> State:
    """Assess risk with Pydantic validation"""
    print("\n🤖 [LANGGRAPH] Risk Assessor Agent Starting...")
    
    try:
        # Validate market data exists
        print(f"🔍 Validating market data availability...")
        if not state["market_data"] or "ticker" not in state["market_data"]:
            raise ValidationError("Market data not available")
        
        ticker = state["market_data"]["ticker"]
        print(f"📊 Processing ticker: {ticker}")
        
        validated_ticker = WorkflowStateInput(ticker=ticker)
        ticker = validated_ticker.ticker
        print(f"✅ Ticker validation passed: {ticker}")
        
        # Get market data for analysis
        market_data = state["market_data"]["data"]
        print(f"📊 Using market data: {market_data}")
        
        # Create risk assessment prompt for LLM
        print(f"🤔 Creating risk assessment prompt...")
        risk_prompt = f"""
        As a risk analyst, assess the investment risk for {ticker} stock based on the following market data:
        
        Market Data: {market_data}
        
        Please provide a comprehensive risk assessment including:
        1. Volatility analysis and risk level
        2. Market risk factors
        3. Company-specific risk factors
        4. Liquidity risk assessment
        5. Sector and macroeconomic risks
        6. Risk mitigation strategies
        7. Overall risk score (Low/Medium/High)
        
        Format your response as a structured risk assessment.
        """
        
        print(f"🧠 Sending prompt to LLM for risk assessment...")
        response = llm.invoke([HumanMessage(content=risk_prompt)])
        risk_result = response.content
        print(f"📊 LLM risk assessment result: {risk_result}")
        
        state["risk_assessment"] = risk_result
        state["workflow_status"] = "Risk assessment completed"
        print(f"✅ Risk Assessor completed successfully")
        
    except ValidationError as e:
        print(f"❌ Risk Assessor validation error: {e}")
        state["risk_assessment"] = f"Validation error: {str(e)}"
        state["workflow_status"] = f"Risk assessment validation error: {str(e)}"
    except Exception as e:
        print(f"❌ Risk Assessor error: {e}")
        state["risk_assessment"] = f"Error in risk assessment: {str(e)}"
        state["workflow_status"] = f"Risk assessment error: {str(e)}"
    
    return state

def decision_maker(state: State) -> State:
    """Make final recommendation with Pydantic validation"""
    print("\n🤖 [LANGGRAPH] Decision Maker Agent Starting...")
    
    try:
        # Validate all analysis components exist
        print(f"🔍 Validating all analysis components...")
        required_fields = ["market_data", "technical_analysis", "fundamental_analysis", "risk_assessment"]
        for field in required_fields:
            if not state.get(field):
                raise ValidationError(f"Missing required field: {field}")
        print(f"✅ All analysis components validated")
        
        # Get all analysis results
        market_data = state['market_data']['data']
        technical_analysis = state['technical_analysis']
        fundamental_analysis = state['fundamental_analysis']
        risk_assessment = state['risk_assessment']
        
        print(f"📊 Market Data: {market_data}")
        print(f"📈 Technical Analysis: {technical_analysis[:100]}...")
        print(f"🏢 Fundamental Analysis: {fundamental_analysis[:100]}...")
        print(f"⚠️ Risk Assessment: {risk_assessment[:100]}...")
        
        # Create comprehensive analysis message
        print(f"🤔 Creating final decision prompt...")
        analysis_prompt = f"""
        As a senior investment analyst, synthesize all the following analysis components to provide a comprehensive stock recommendation:
        
        MARKET DATA:
        {market_data}
        
        TECHNICAL ANALYSIS:
        {technical_analysis}
        
        FUNDAMENTAL ANALYSIS:
        {fundamental_analysis}
        
        RISK ASSESSMENT:
        {risk_assessment}
        
        Based on this comprehensive analysis, please provide:
        1. A clear BUY, HOLD, or SELL recommendation
        2. Confidence level (High/Medium/Low)
        3. Key reasoning points that led to this decision
        4. Risk factors to consider
        5. Time horizon recommendation (short-term/medium-term/long-term)
        6. Price targets or expectations
        7. Summary of the most important factors
        
        Format your response as a structured investment recommendation.
        """
        
        # Get recommendation from LLM
        print(f"🧠 Sending comprehensive analysis to LLM for final decision...")
        response = llm.invoke([HumanMessage(content=analysis_prompt)])
        recommendation_text = response.content
        print(f"📊 LLM final recommendation: {recommendation_text[:200]}...")
        
        state["final_recommendation"] = {
            "recommendation": recommendation_text,
            "status": "completed",
            "timestamp": datetime.now().isoformat()
        }
        state["workflow_status"] = "Final recommendation completed"
        print(f"✅ Decision Maker completed successfully")
        
    except ValidationError as e:
        print(f"❌ Decision Maker validation error: {e}")
        state["final_recommendation"] = {
            "recommendation": f"Validation error: {str(e)}",
            "status": "error",
            "timestamp": datetime.now().isoformat()
        }
        state["workflow_status"] = f"Decision making validation error: {str(e)}"
    except Exception as e:
        print(f"❌ Decision Maker error: {e}")
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

# Add edges with transition logging
def log_transition(from_node: str, to_node: str, state: State) -> State:
    """Log workflow transitions"""
    print(f"\n🔄 [LANGGRAPH] Workflow Transition: {from_node} → {to_node}")
    print(f"📊 Current workflow status: {state.get('workflow_status', 'Unknown')}")
    return state

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
    print(f'📊 LangGraph Stock Bot (Slash Commands) with Pydantic Validation is ready!')
    print(f'🤖 Provider: {provider.upper()}')
    print(f'💬 Use /analyze <ticker> to get stock recommendations')
    
    # Register slash commands
    try:
        print("🔄 Registering slash commands...")
        await bot.tree.sync()
        print("✅ Slash commands registered successfully!")
    except Exception as e:
        print(f"❌ Failed to register slash commands: {e}")

# Slash command handlers with Pydantic validation
@bot.tree.command(name="analyze", description="Analyze any stock using LangGraph multi-agent system with Pydantic validation")
async def analyze_stock_slash(interaction: discord.Interaction, ticker: str, query: str = None):
    """Slash command handler for stock analysis with Pydantic validation"""
    await interaction.response.defer()
    
    try:
        # Validate input with Pydantic
        try:
            validated_input = WorkflowStateInput(ticker=ticker, query=query)
            ticker = validated_input.ticker
        except ValidationError as e:
            error_embed = discord.Embed(
                title="❌ Validation Error",
                description=f"Invalid input: {str(e)}",
                color=0xff0000
            )
            await interaction.followup.send(embed=error_embed)
            return
        
        # Send initial message
        embed = discord.Embed(
            title="📊 Stock Analysis in Progress",
            description=f"Analyzing **{ticker}** using LangGraph multi-agent system with Pydantic validation...",
            color=0x00ff00
        )
        embed.add_field(name="Status", value="🔄 Starting workflow...", inline=False)
        message = await interaction.followup.send(embed=embed)
        
        # Initialize state
        initial_state = {
            "messages": [HumanMessage(content=f"analyze {ticker}")],
            "steps": [],
            "market_data": {},
            "technical_analysis": "",
            "fundamental_analysis": "",
            "risk_assessment": "",
            "final_recommendation": {},
            "workflow_status": "Initialized"
        }
        
        # Run the workflow and show thinking process
        print(f"\n🚀 [LANGGRAPH] Starting workflow execution for {ticker}...")
        print(f"📊 Initial state: {initial_state}")
        
        # Create initial embed for thinking process
        thinking_embed = discord.Embed(
            title=f"🤖 LangGraph Thinking Process - {ticker}",
            description="Executing multi-agent workflow...",
            color=0x0099ff
        )
        thinking_embed.add_field(name="🔄 Status", value="Starting workflow...", inline=False)
        thinking_message = await interaction.followup.send(embed=thinking_embed)
        
        # Execute workflow and capture steps
        result = app.invoke(initial_state)
        
        # Show the thinking process from the result
        steps = result.get("steps", [])
        if steps:
            # Create a summary of the thinking process
            thinking_embed.clear_fields()
            thinking_embed.add_field(
                name="🔄 Workflow Completed", 
                value=f"Executed {len(steps)} thinking steps", 
                inline=False
            )
            
            # Show recent steps
            recent_steps = steps[-8:]  # Show last 8 steps
            steps_text = ""
            for step in recent_steps:
                emoji = "💭" if step['phase'] == 'thought' else "⚡" if step['phase'] == 'action' else "👁️"
                steps_text += f"{emoji} **{step['phase'].title()}**: {step['content'][:80]}{'...' if len(step['content']) > 80 else ''}\n"
            
            thinking_embed.add_field(
                name="📝 Thinking Process", 
                value=steps_text if steps_text else "No steps recorded", 
                inline=False
            )
            
            thinking_embed.add_field(
                name="🔄 Final Status", 
                value=result.get("workflow_status", "Completed"), 
                inline=True
            )
            
            await thinking_message.edit(embed=thinking_embed)
        
        print(f"✅ [LANGGRAPH] Workflow execution completed")
        
        # Create detailed response embed
        embed = discord.Embed(
            title=f"📊 {ticker} Stock Analysis - LangGraph Multi-Agent (Pydantic)",
            description="Analysis completed using LangGraph multi-agent workflow with Pydantic validation",
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
            value="LangGraph Multi-Agent + Pydantic", 
            inline=True
        )
        embed.add_field(
            name="⏰ Completed", 
            value=datetime.now().strftime("%H:%M:%S"), 
            inline=True
        )
        
        embed.set_footer(text=f"pookan-langgraph-pydantic • {provider.upper()} • Real-time market data")
        await message.edit(embed=embed)
        
    except Exception as e:
        error_embed = discord.Embed(
            title="❌ Error",
            description=f"Failed to analyze {ticker}: {str(e)}",
            color=0xff0000
        )
        await interaction.followup.send(embed=error_embed)

def main():
    """Run the Discord bot"""
    token = os.getenv("DISCORD_TOKEN")
    if not token:
        print("❌ Error: DISCORD_TOKEN not found in environment variables")
        print("💡 Make sure DISCORD_TOKEN is set in environment variables")
        return
    
    print("🚀 Starting pookan-langgraph Discord bot with Pydantic validation...")
    print("📊 Bot will be ready to analyze any stocks!")
    print(f"✅ Environment variables loaded successfully")
    print(f"⚡ Provider: {provider.upper()}")
    print(f"🔒 Pydantic validation enabled")
    print(f"💡 Use /analyze <ticker> to analyze a stock")
    
    try:
        bot.run(token)
    except Exception as e:
        print(f"❌ Failed to start bot: {e}")

if __name__ == "__main__":
    main()
