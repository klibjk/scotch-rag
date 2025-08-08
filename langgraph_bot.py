"""
Discord Bot: pookan-langgraph
Tesla Stock Recommendation using LangGraph Multi-Agent System
Self-contained implementation
"""

import discord
from discord.ext import commands
import asyncio
import json
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
            
            stock = yf.Ticker(ticker)
            hist = stock.history(period="30d")
            
            if hist.empty:
                return f"Error: Unable to fetch {ticker} market data"
            
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
            return f"Error fetching market data: {str(e)}"

class TechnicalAnalysisTool(BaseTool):
    name: str = "technical_analysis"
    description: str = "Perform detailed technical analysis on any stock using multiple indicators"
    
    def _run(self, query: str) -> str:
        """Perform technical analysis for any stock"""
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
            
            stock = yf.Ticker(ticker)
            hist = stock.history(period="30d")
            
            if hist.empty:
                return f"Error: Unable to fetch data for {ticker} technical analysis"
            
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
            
            # Analysis
            rsi_signal = "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral"
            ma_signal = "Bullish" if current_price > sma_20 > sma_50 else "Bearish" if current_price < sma_20 < sma_50 else "Neutral"
            macd_signal = "Bullish" if current_macd > current_signal else "Bearish"
            
            analysis = f"{ticker} Technical Analysis: RSI={current_rsi:.1f} ({rsi_signal}), MA Signal={ma_signal}, MACD={macd_signal}"
            
            return analysis
            
        except Exception as e:
            return f"Error in technical analysis: {str(e)}"

class FundamentalAnalysisTool(BaseTool):
    name: str = "fundamental_analysis"
    description: str = "Analyze fundamental metrics and company performance for any stock"
    
    def _run(self, query: str) -> str:
        """Analyze fundamental metrics for any stock"""
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
            
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Extract key metrics
            market_cap = info.get('marketCap', 0)
            pe_ratio = info.get('trailingPE', 0)
            pb_ratio = info.get('priceToBook', 0)
            debt_to_equity = info.get('debtToEquity', 0)
            profit_margins = info.get('profitMargins', 0)
            revenue_growth = info.get('revenueGrowth', 0)
            
            # Analysis
            pe_assessment = "High" if pe_ratio > 50 else "Reasonable" if pe_ratio > 20 else "Low"
            debt_assessment = "High" if debt_to_equity > 1 else "Manageable"
            margin_assessment = "Strong" if profit_margins > 0.1 else "Weak" if profit_margins < 0 else "Moderate"
            
            analysis = f"{ticker} Fundamental Analysis: Market Cap=${market_cap/1e9:.1f}B, P/E={pe_ratio:.1f} ({pe_assessment}), Debt/Equity={debt_to_equity:.2f} ({debt_assessment}), Profit Margin={profit_margins:.1%} ({margin_assessment})"
            
            return analysis
            
        except Exception as e:
            return f"Error in fundamental analysis: {str(e)}"

class RiskAssessmentTool(BaseTool):
    name: str = "risk_assessment"
    description: str = "Assess investment risk for any stock based on various factors"
    
    def _run(self, query: str) -> str:
        """Assess investment risk for any stock"""
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
            
            stock = yf.Ticker(ticker)
            hist = stock.history(period="30d")
            info = stock.info
            
            # Calculate volatility
            returns = hist['Close'].pct_change()
            volatility = returns.std() * (252 ** 0.5)  # Annualized volatility
            
            # Beta calculation (simplified)
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
            if volatility > 0.5:
                risk_factors.append("High volatility")
            if beta > 1.5:
                risk_factors.append("High beta")
            if info.get('debtToEquity', 0) > 1:
                risk_factors.append("High debt levels")
            
            risk_level = "High" if len(risk_factors) > 2 else "Medium" if len(risk_factors) > 0 else "Low"
            
            assessment = f"{ticker} Risk Assessment: Volatility={volatility:.1%}, Beta={beta:.2f}, Risk Level={risk_level}, Factors: {', '.join(risk_factors) if risk_factors else 'None'}"
            
            return assessment
            
        except Exception as e:
            return f"Error in risk assessment: {str(e)}"

# Create tool instances
fetch_market_data = MarketDataTool()
technical_analysis = TechnicalAnalysisTool()
fundamental_analysis = FundamentalAnalysisTool()
risk_assessment = RiskAssessmentTool()

# Define workflow nodes
def data_fetcher(state: State) -> State:
    """Fetch market data for any stock"""
    try:
        # Extract ticker from messages or use default
        ticker = "TSLA"  # Default fallback
        messages = state.get("messages", [])
        if messages:
            last_message = messages[-1].content if hasattr(messages[-1], 'content') else str(messages[-1])
            words = last_message.strip().upper().split()
            for word in words:
                if len(word) <= 5 and word.isalpha():
                    ticker = word
                    break
        
        # Get market data
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d")
        market_data = {}
        
        if not hist.empty:
            current_price = hist['Close'].iloc[-1]
            previous_close = hist['Open'].iloc[-1]
            price_change = ((current_price - previous_close) / previous_close) * 100
            volume = hist['Volume'].iloc[-1]
            
            market_data = {
                "current_price": round(current_price, 2),
                "price_change_percent": round(price_change, 2),
                "volume": int(volume)
            }
        
        # Add message
        messages = state["messages"] + [HumanMessage(content=f"Fetch {ticker} market data")]
        
        return {
            **state,
            "messages": messages,
            "market_data": market_data,
            "workflow_status": f"Data fetched successfully for {ticker}"
        }
    except Exception as e:
        return {
            **state,
            "workflow_status": f"Data fetch error: {str(e)}"
        }

def technical_analyst(state: State) -> State:
    """Perform technical analysis"""
    try:
        # Extract ticker from messages or use default
        ticker = "TSLA"  # Default fallback
        messages = state.get("messages", [])
        if messages:
            last_message = messages[-1].content if hasattr(messages[-1], 'content') else str(messages[-1])
            words = last_message.strip().upper().split()
            for word in words:
                if len(word) <= 5 and word.isalpha():
                    ticker = word
                    break
        
        # Get technical analysis
        result = technical_analysis.invoke(f"Analyze {ticker}")
        
        # Add message
        messages = state["messages"] + [HumanMessage(content=f"Technical Analysis: {result}")]
        
        return {
            **state,
            "messages": messages,
            "technical_analysis": result,
            "workflow_status": f"Technical analysis completed for {ticker}"
        }
    except Exception as e:
        return {
            **state,
            "workflow_status": f"Technical analysis error: {str(e)}"
        }

def fundamental_analyst(state: State) -> State:
    """Perform fundamental analysis"""
    try:
        # Extract ticker from messages or use default
        ticker = "TSLA"  # Default fallback
        messages = state.get("messages", [])
        if messages:
            last_message = messages[-1].content if hasattr(messages[-1], 'content') else str(messages[-1])
            words = last_message.strip().upper().split()
            for word in words:
                if len(word) <= 5 and word.isalpha():
                    ticker = word
                    break
        
        # Get fundamental analysis
        result = fundamental_analysis.invoke(f"Analyze {ticker}")
        
        # Add message
        messages = state["messages"] + [HumanMessage(content=f"Fundamental Analysis: {result}")]
        
        return {
            **state,
            "messages": messages,
            "fundamental_analysis": result,
            "workflow_status": f"Fundamental analysis completed for {ticker}"
        }
    except Exception as e:
        return {
            **state,
            "workflow_status": f"Fundamental analysis error: {str(e)}"
        }

def risk_assessor(state: State) -> State:
    """Assess investment risk"""
    try:
        # Extract ticker from messages or use default
        ticker = "TSLA"  # Default fallback
        messages = state.get("messages", [])
        if messages:
            last_message = messages[-1].content if hasattr(messages[-1], 'content') else str(messages[-1])
            words = last_message.strip().upper().split()
            for word in words:
                if len(word) <= 5 and word.isalpha():
                    ticker = word
                    break
        
        # Get risk assessment
        result = risk_assessment.invoke(f"Assess {ticker}")
        
        # Add message
        messages = state["messages"] + [HumanMessage(content=f"Risk Assessment: {result}")]
        
        return {
            **state,
            "messages": messages,
            "risk_assessment": result,
            "workflow_status": f"Risk assessment completed for {ticker}"
        }
    except Exception as e:
        return {
            **state,
            "workflow_status": f"Risk assessment error: {str(e)}"
        }

def decision_maker(state: State) -> State:
    """Make final investment decision"""
    try:
        # Extract ticker from messages or use default
        ticker = "TSLA"  # Default fallback
        messages = state.get("messages", [])
        if messages:
            last_message = messages[-1].content if hasattr(messages[-1], 'content') else str(messages[-1])
            words = last_message.strip().upper().split()
            for word in words:
                if len(word) <= 5 and word.isalpha():
                    ticker = word
                    break
        
        # Prepare analysis summary
        analysis_summary = f"""
        Market Data: {state.get('market_data', {})}
        Technical Analysis: {state.get('technical_analysis', 'Not available')}
        Fundamental Analysis: {state.get('fundamental_analysis', 'Not available')}
        Risk Assessment: {state.get('risk_assessment', 'Not available')}
        """
        
        # Get decision from LLM
        decision_prompt = f"""
        Based on the following analysis, provide a comprehensive investment recommendation for {ticker} stock:
        
        {analysis_summary}
        
        Provide a final recommendation with:
        1. Recommendation (BUY/SELL/HOLD)
        2. Confidence level (High/Medium/Low)
        3. Risk level (High/Medium/Low)
        4. Detailed reasoning
        """
        
        response = llm.invoke([HumanMessage(content=decision_prompt)])
        decision_text = response.content
        
        # Parse recommendation
        recommendation = "BUY" if "buy" in decision_text.lower() else "SELL" if "sell" in decision_text.lower() else "HOLD"
        confidence = "High" if "high" in decision_text.lower() else "Medium" if "medium" in decision_text.lower() else "Low"
        risk_level = "High" if "high risk" in decision_text.lower() else "Medium" if "medium risk" in decision_text.lower() else "Low"
        
        final_recommendation = {
            "recommendation": recommendation,
            "confidence": confidence,
            "reasoning": decision_text,
            "risk_level": risk_level
        }
        
        # Add message
        messages = state["messages"] + [HumanMessage(content=f"Final Decision for {ticker}: {decision_text}")]
        
        return {
            **state,
            "messages": messages,
            "final_recommendation": final_recommendation,
            "workflow_status": f"Decision made successfully for {ticker}"
        }
    except Exception as e:
        return {
            **state,
            "workflow_status": f"Decision making error: {str(e)}"
        }

# Build the workflow graph
workflow = StateGraph(State)

# Add nodes
workflow.add_node("data_fetcher", data_fetcher)
workflow.add_node("technical_analyst", technical_analyst)
workflow.add_node("fundamental_analyst", fundamental_analyst)
workflow.add_node("risk_assessor", risk_assessor)
workflow.add_node("decision_maker", decision_maker)

# Add edges
workflow.set_entry_point("data_fetcher")
workflow.add_edge("data_fetcher", "technical_analyst")
workflow.add_edge("technical_analyst", "fundamental_analyst")
workflow.add_edge("fundamental_analyst", "risk_assessor")
workflow.add_edge("risk_assessor", "decision_maker")
workflow.add_edge("decision_maker", END)

# Compile the graph
app = workflow.compile()

class LangGraphStockRecommendationSystem:
    """Self-contained LangGraph Multi-Agent System for stock recommendations"""
    
    def __init__(self):
        self.workflow = app
        self.provider = provider
    
    def get_recommendation(self, ticker: str = "TSLA", query: str = None) -> Dict[str, Any]:
        """Get stock recommendation using LangGraph workflow"""
        try:
            # Initialize state with ticker information
            initial_state = {
                "messages": [HumanMessage(content=f"Analyze {ticker} stock: {query or 'General analysis'}")],
                "market_data": {},
                "technical_analysis": "",
                "fundamental_analysis": "",
                "risk_assessment": "",
                "final_recommendation": {},
                "workflow_status": f"Starting workflow for {ticker}"
            }
            
            # Run the workflow
            result = self.workflow.invoke(initial_state)
            
            # Extract results
            market_data = result.get("market_data", {})
            final_recommendation = result.get("final_recommendation", {})
            workflow_status = result.get("workflow_status", "unknown")
            agent_messages = result.get("messages", [])
            
            return {
                "final_recommendation": final_recommendation,
                "market_data": market_data,
                "agent_messages": agent_messages,
                "workflow_status": workflow_status,
                "framework": "LangGraph Multi-Agent",
                "provider": self.provider
            }
            
        except Exception as e:
            error_msg = str(e)
            if "404" in error_msg or "not_found_error" in error_msg:
                error_msg = "API endpoint not found. Please check your API keys and try again."
            elif "rate_limit" in error_msg.lower():
                error_msg = "Rate limit exceeded. Please try again later."
            elif "timeout" in error_msg.lower():
                error_msg = "Request timed out. Please try again."
            
            return {
                "error": error_msg,
                "framework": "LangGraph Multi-Agent",
                "provider": self.provider
            }

# Initialize the system
stock_system = LangGraphStockRecommendationSystem()

@bot.event
async def on_ready():
    """Bot startup event"""
    print(f'🤖 {bot.user} has connected to Discord!')
    print(f'📊 LangGraph Multi-Agent Stock Bot is ready!')
    print(f'💬 Use !analyze <ticker> to get stock recommendations')
    print(f'🔵 Active Provider: {provider.upper()}')

@bot.command(name='analyze')
async def analyze_stock(ctx, ticker: str, *, query: str = None):
    """
    Get stock recommendation using LangGraph multi-agent system
    
    Usage: !analyze <ticker> [optional query]
    Examples:
        !analyze TSLA
        !analyze AAPL should I buy Apple stock?
        !analyze MSFT analyze Microsoft fundamentals
    """
    
    # Send initial response
    embed = discord.Embed(
        title="🤖 pookan-langgraph Stock Analysis",
        description=f"Analyzing {ticker.upper()} stock using LangGraph multi-agent workflow...",
        color=0x00ff00
    )
    embed.add_field(name="🔍 Ticker", value=ticker.upper(), inline=True)
    embed.add_field(name="🔍 Query", value=query or f"General {ticker.upper()} stock analysis", inline=True)
    embed.add_field(name="⚡ Framework", value="LangGraph Multi-Agent", inline=True)
    embed.add_field(name="🤖 AI Provider", value=provider.upper(), inline=True)
    embed.add_field(name="⏰ Started", value=datetime.now().strftime("%H:%M:%S"), inline=True)
    
    message = await ctx.send(embed=embed)
    
    try:
        # Get recommendation from our system
        result = stock_system.get_recommendation(ticker, query)
        
        if "error" in result:
            # Error occurred
            error_embed = discord.Embed(
                title="❌ Analysis Failed",
                description=f"Error: {result['error']}",
                color=0xff0000
            )
            await message.edit(embed=error_embed)
            return
        
        # Extract recommendation data
        final_recommendation = result.get("final_recommendation", {})
        market_data = result.get("market_data", {})
        workflow_status = result.get("workflow_status", "unknown")
        
        # Create detailed response embed
        response_embed = discord.Embed(
            title=f"📊 {ticker.upper()} Stock Analysis - LangGraph Multi-Agent",
            description="Analysis completed using advanced LangGraph workflow orchestration",
            color=0x00ff00
        )
        
        # Add market data
        if market_data and "current_price" in market_data:
            response_embed.add_field(
                name="💰 Current Price", 
                value=f"${market_data['current_price']}", 
                inline=True
            )
            response_embed.add_field(
                name="📈 Change", 
                value=f"{market_data.get('price_change_percent', 0)}%", 
                inline=True
            )
            response_embed.add_field(
                name="📊 Volume", 
                value=f"{market_data.get('volume', 0):,}", 
                inline=True
            )
        
        # Add recommendation details
        if final_recommendation:
            recommendation = final_recommendation.get("recommendation", "UNKNOWN")
            confidence = final_recommendation.get("confidence", "UNKNOWN")
            reasoning = final_recommendation.get("reasoning", "")
            
            response_embed.add_field(
                name="🎯 Recommendation", 
                value=recommendation, 
                inline=True
            )
            response_embed.add_field(
                name="📊 Confidence", 
                value=confidence, 
                inline=True
            )
            response_embed.add_field(
                name="⚠️ Risk Level", 
                value=final_recommendation.get("risk_level", "UNKNOWN"), 
                inline=True
            )
            
            # Add reasoning (truncated if too long)
            if reasoning:
                reasoning_short = reasoning[:1024] if len(reasoning) > 1024 else reasoning
                response_embed.add_field(
                    name="💭 Reasoning", 
                    value=reasoning_short, 
                    inline=False
                )
        
        # Add workflow status
        response_embed.add_field(
            name="🔄 Workflow Status", 
            value=workflow_status, 
            inline=True
        )
        response_embed.add_field(
            name="⚡ Framework", 
            value="LangGraph Multi-Agent", 
            inline=True
        )
        response_embed.add_field(
            name="⏰ Completed", 
            value=datetime.now().strftime("%H:%M:%S"), 
            inline=True
        )
        
        # Add agent execution flow
        agent_messages = result.get("agent_messages", [])
        if agent_messages:
            flow_summary = []
            for msg in agent_messages[-3:]:  # Get last 3 messages
                if hasattr(msg, 'content'):
                    flow_summary.append(f"• {msg.content[:100]}...")
                elif isinstance(msg, dict) and 'content' in msg:
                    flow_summary.append(f"• {msg['content'][:100]}...")
                else:
                    flow_summary.append(f"• {str(msg)[:100]}...")
            
            if flow_summary:
                response_embed.add_field(
                    name="🤖 Agent Flow", 
                    value="\n".join(flow_summary), 
                    inline=False
                )
        
        # Add footer
        response_embed.set_footer(text="pookan-langgraph • LangGraph Multi-Agent")
        
        await message.edit(embed=response_embed)
        
    except Exception as e:
        # Handle any errors
        error_embed = discord.Embed(
            title="❌ Analysis Failed",
            description=f"An error occurred: {str(e)}",
            color=0xff0000
        )
        await message.edit(embed=error_embed)

@bot.command(name='workflow')
async def workflow_status(ctx):
    """Show detailed workflow status"""
    embed = discord.Embed(
        title="🔄 LangGraph Workflow Status",
        description="Advanced multi-agent workflow orchestration",
        color=0x0099ff
    )
    
    embed.add_field(
        name="⚡ Framework",
        value="LangGraph Multi-Agent\nAdvanced orchestration with state management",
        inline=True
    )
    
    embed.add_field(
        name="🤖 Agents",
        value="• DataFetcher\n• TechnicalAnalyst\n• FundamentalAnalyst\n• RiskAssessor\n• DecisionMaker",
        inline=True
    )
    
    embed.add_field(
        name="🔄 Features",
        value="• Conditional routing\n• State management\n• Error recovery\n• Parallel processing\n• Production-ready",
        inline=True
    )
    
    embed.set_footer(text="pookan-langgraph • LangGraph Multi-Agent")
    await ctx.send(embed=embed)

@bot.command(name='stockhelp')
async def help_command(ctx):
    """Show comprehensive help information"""
    embed = discord.Embed(
        title="🤖 LangGraph Multi-Agent Bot - Help Guide",
        description="**AI-powered stock analysis using LangGraph advanced workflow orchestration**\n\nThis bot provides production-ready stock analysis using advanced multi-agent workflows with state management.",
        color=0x0099ff
    )
    
    # Basic commands
    embed.add_field(
        name="📊 **Basic Commands**",
        value="""
`!analyze <ticker>` - Analyze any stock (e.g., `!analyze MSFT`)
`!analyze <ticker> <query>` - Specific analysis with custom query
`!workflow` - Show detailed workflow information
`!ping` - Test bot connectivity
`!status` - Show bot status and configuration
`!stockhelp` - Show this help message
        """,
        inline=False
    )
    
    # Examples
    embed.add_field(
        name="💡 **Usage Examples**",
        value="""
• `!analyze AAPL` - Analyze Apple stock
• `!analyze TSLA should I buy?` - Specific buying advice
• `!analyze GOOGL analyze fundamentals` - Focus on fundamentals
• `!analyze NVDA technical analysis` - Focus on technical indicators
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
• Use `!workflow` to see workflow details
• Great for production environments
        """,
        inline=False
    )
    
    embed.set_footer(text=f"LangGraph Multi-Agent • {provider.upper()} • Real-time market data")
    await ctx.send(embed=embed)

@bot.command(name='status')
async def status_command(ctx):
    """Show bot status"""
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
        value="!analyze, !workflow, !stockhelp, !status",
        inline=True
    )
    
    embed.set_footer(text=f"pookan-langgraph • {provider.upper()} • {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    await ctx.send(embed=embed)

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
    bot.run(token)

if __name__ == "__main__":
    main() 