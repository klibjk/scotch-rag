"""
Discord Bot: pookan-langchain-multi
Tesla Stock Recommendation using LangChain Multi-Agent System
Self-contained implementation
"""

import discord
from discord.ext import commands
import asyncio
import json
from datetime import datetime
import os
import warnings
from typing import Dict, Any, Optional, List
import yfinance as yf
import pandas as pd
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, SystemMessage
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

# Initialize the LangChain system with fallback mechanism: Anthropic first, OpenAI second
def configure_langchain():
    """Configure LangChain with fallback mechanism"""
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

# Configure LangChain
try:
    llm, provider = configure_langchain()
    print(f"✅ LangChain configured successfully with {provider.upper()}")
except Exception as e:
    print(f"❌ LangChain configuration failed: {e}")
    print("💡 Please check your API keys and try again")
    exit(1)

# Self-contained LangChain Multi-Agent Implementation
class DataFetcherTool(BaseTool):
    name: str = "fetch_market_data"
    description: str = "Fetch comprehensive market data for any stock including price, volume, and technical indicators"
    
    def _run(self, query: str) -> str:
        """Fetch market data for any stock"""
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

class TechnicalAnalystTool(BaseTool):
    name: str = "technical_analysis"
    description: str = "Perform detailed technical analysis on any stock using multiple indicators"
    
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

class FundamentalAnalystTool(BaseTool):
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

class RiskAssessorTool(BaseTool):
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

class DecisionMakerTool(BaseTool):
    name: str = "make_decision"
    description: str = "Make final investment decision based on all analysis results"
    
    def _run(self, query: str) -> str:
        """Make investment decision"""
        try:
            # This tool will be used by the decision maker agent
            # It receives the combined analysis and makes the final decision
            return "Decision: Based on comprehensive analysis, this tool provides the final investment recommendation."
            
        except Exception as e:
            return f"Error in decision making: {str(e)}"

# Initialize tools for each agent
data_fetcher_tools = [DataFetcherTool()]
technical_analyst_tools = [TechnicalAnalystTool()]
fundamental_analyst_tools = [FundamentalAnalystTool()]
risk_assessor_tools = [RiskAssessorTool()]
decision_maker_tools = [DecisionMakerTool()]

# Initialize agents with modern approach
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate

# Create the prompt template
template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Question: {input}
Thought: {agent_scratchpad}"""

prompt = PromptTemplate.from_template(template)

# Create agents
data_fetcher_agent = AgentExecutor(
    agent=create_react_agent(llm, data_fetcher_tools, prompt),
    tools=data_fetcher_tools,
    verbose=True,
    handle_parsing_errors=True
)

technical_analyst_agent = AgentExecutor(
    agent=create_react_agent(llm, technical_analyst_tools, prompt),
    tools=technical_analyst_tools,
    verbose=True,
    handle_parsing_errors=True
)

fundamental_analyst_agent = AgentExecutor(
    agent=create_react_agent(llm, fundamental_analyst_tools, prompt),
    tools=fundamental_analyst_tools,
    verbose=True,
    handle_parsing_errors=True
)

risk_assessor_agent = AgentExecutor(
    agent=create_react_agent(llm, risk_assessor_tools, prompt),
    tools=risk_assessor_tools,
    verbose=True,
    handle_parsing_errors=True
)

decision_maker_agent = AgentExecutor(
    agent=create_react_agent(llm, decision_maker_tools, prompt),
    tools=decision_maker_tools,
    verbose=True,
    handle_parsing_errors=True
)

class MultiAgentStockRecommendationSystem:
    """Self-contained LangChain Multi-Agent System for stock recommendations"""
    
    def __init__(self):
        self.provider = provider
        self.agents = {
            'data_fetcher': data_fetcher_agent,
            'technical_analyst': technical_analyst_agent,
            'fundamental_analyst': fundamental_analyst_agent,
            'risk_assessor': risk_assessor_agent,
            'decision_maker': decision_maker_agent
        }
    
    def get_recommendation(self, ticker: str = "TSLA", query: str = None) -> Dict[str, Any]:
        """Get stock recommendation using multi-agent system"""
        try:
            # Extract ticker from query if provided
            if query and len(query.strip()) > 0:
                words = query.strip().upper().split()
                for word in words:
                    if len(word) <= 5 and word.isalpha():
                        ticker = word
                        break
            
            agent_messages = []
            
            # Step 1: Data Fetcher Agent
            data_query = f"Fetch comprehensive market data for {ticker} stock"
            data_result = self.agents['data_fetcher'].invoke({"input": data_query})
            agent_messages.append({
                'agent': 'DataFetcher',
                'message': data_result.get("output", str(data_result))
            })
            
            # Step 2: Technical Analyst Agent
            tech_query = f"Perform detailed technical analysis on {ticker} stock"
            tech_result = self.agents['technical_analyst'].invoke({"input": tech_query})
            agent_messages.append({
                'agent': 'TechnicalAnalyst',
                'message': tech_result.get("output", str(tech_result))
            })
            
            # Step 3: Fundamental Analyst Agent
            fund_query = f"Analyze {ticker}'s fundamental metrics and company performance"
            fund_result = self.agents['fundamental_analyst'].invoke({"input": fund_query})
            agent_messages.append({
                'agent': 'FundamentalAnalyst',
                'message': fund_result.get("output", str(fund_result))
            })
            
            # Step 4: Risk Assessor Agent
            risk_query = f"Assess investment risk for {ticker} stock"
            risk_result = self.agents['risk_assessor'].invoke({"input": risk_query})
            agent_messages.append({
                'agent': 'RiskAssessor',
                'message': risk_result.get("output", str(risk_result))
            })
            
            # Step 5: Decision Maker Agent
            decision_query = f"""
            Based on the following analysis, provide a comprehensive investment recommendation for {ticker} stock:
            
            Market Data: {data_result}
            Technical Analysis: {tech_result}
            Fundamental Analysis: {fund_result}
            Risk Assessment: {risk_result}
            
            User Query: {query or f'General {ticker} stock analysis'}
            
            Provide a final recommendation with confidence level and reasoning.
            """
            
            decision_result = self.agents['decision_maker'].invoke({"input": decision_query})
            agent_messages.append({
                'agent': 'DecisionMaker',
                'message': decision_result.get("output", str(decision_result))
            })
            
            # Extract market data
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
            
            # Parse final recommendation
            decision_text = decision_result.get("output", str(decision_result))
            risk_text = risk_result.get("output", str(risk_result))
            final_recommendation = {
                "recommendation": "BUY" if "buy" in decision_text.lower() else "SELL" if "sell" in decision_text.lower() else "HOLD",
                "confidence": "High" if "high" in decision_text.lower() else "Medium" if "medium" in decision_text.lower() else "Low",
                "reasoning": decision_text,
                "risk_level": "High" if "high risk" in risk_text.lower() else "Medium" if "medium risk" in risk_text.lower() else "Low"
            }
            
            return {
                "final_recommendation": final_recommendation,
                "market_data": market_data,
                "agent_messages": agent_messages,
                "workflow_execution": "Completed successfully",
                "framework": "LangChain Multi-Agent",
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
                "framework": "LangChain Multi-Agent",
                "provider": self.provider
            }

# Initialize the system
stock_system = MultiAgentStockRecommendationSystem()

@bot.event
async def on_ready():
    """Bot startup event"""
    print(f'🤖 {bot.user} has connected to Discord!')
    print(f'📊 LangChain Multi-Agent Stock Bot is ready!')
    print(f'💬 Use !analyze <ticker> to get stock recommendations')

@bot.command(name='analyze')
async def analyze_stock(ctx, ticker: str, *, query: str = None):
    """
    Get stock recommendation using LangChain multi-agent system
    
    Usage: !analyze <ticker> [optional query]
    Examples:
        !analyze TSLA
        !analyze AAPL should I buy Apple stock?
        !analyze MSFT analyze Microsoft fundamentals
    """
    
    # Send initial response
    embed = discord.Embed(
        title="🤖 pookan-langchain-multi Stock Analysis",
        description=f"Analyzing {ticker.upper()} stock using LangChain multi-agent system...",
        color=0x00ff00
    )
    embed.add_field(name="🔍 Ticker", value=ticker.upper(), inline=True)
    embed.add_field(name="🔍 Query", value=query or f"General {ticker.upper()} stock analysis", inline=True)
    embed.add_field(name="⚡ Framework", value="LangChain Multi-Agent", inline=True)
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
        workflow_status = result.get("workflow_execution", "unknown")
        
        # Create detailed response embed
        response_embed = discord.Embed(
            title=f"📊 {ticker.upper()} Stock Analysis - LangChain Multi-Agent",
            description="Analysis completed using LangChain multi-agent orchestration",
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
            value="LangChain Multi-Agent", 
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
            flow_summary = "\n".join([f"• {msg.get('agent', 'Unknown')}: {msg.get('message', str(msg))[:100]}..." for msg in agent_messages[:3]])
            response_embed.add_field(
                name="🤖 Agent Flow", 
                value=flow_summary, 
                inline=False
            )
        
        # Add footer
        response_embed.set_footer(text="pookan-langchain-multi • LangChain Multi-Agent")
        
        await message.edit(embed=response_embed)
        
    except Exception as e:
        # Handle any errors
        error_embed = discord.Embed(
            title="❌ Analysis Failed",
            description=f"An error occurred: {str(e)}",
            color=0xff0000
        )
        await message.edit(embed=error_embed)

@bot.command(name='agents')
async def agents_info(ctx):
    """Show LangChain multi-agent information"""
    embed = discord.Embed(
        title="🤖 LangChain Multi-Agent System",
        description="LangChain-based multi-agent orchestration",
        color=0x0099ff
    )
    
    embed.add_field(
        name="⚡ Framework",
        value="LangChain Multi-Agent\nTool-based agent orchestration",
        inline=True
    )
    
    embed.add_field(
        name="🤖 Agents",
        value="• DataFetcher\n• TechnicalAnalyst\n• FundamentalAnalyst\n• RiskAssessor\n• DecisionMaker",
        inline=True
    )
    
    embed.add_field(
        name="🛠️ Features",
        value="• LangChain tools\n• Agent orchestration\n• Tool-based analysis\n• Structured workflow\n• Multi-agent coordination",
        inline=True
    )
    
    embed.set_footer(text="pookan-langchain-multi • LangChain Multi-Agent")
    await ctx.send(embed=embed)

@bot.command(name='stockhelp')
async def help_command(ctx):
    """Show comprehensive help information"""
    embed = discord.Embed(
        title="🤖 LangChain Multi-Agent Bot - Help Guide",
        description="**AI-powered stock analysis using LangChain multi-agent orchestration**\n\nThis bot provides the most detailed stock analysis using multiple specialized agents working together.",
        color=0x0099ff
    )
    
    # Basic commands
    embed.add_field(
        name="📊 **Basic Commands**",
        value="""
`!analyze <ticker>` - Analyze any stock (e.g., `!analyze MSFT`)
`!analyze <ticker> <query>` - Specific analysis with custom query
`!agents` - Show detailed agent system information
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
• **5 Specialized Agents** working together:
  - **Data Fetcher** - Real-time market data
  - **Technical Analyst** - Technical indicators & patterns
  - **Fundamental Analyst** - Company financials & metrics
  - **Risk Assessor** - Risk analysis & factors
  - **Decision Maker** - Final recommendation synthesis
• **Most detailed analysis** of all bots
• **Comprehensive coverage** of all analysis types
• **Structured workflow** with agent coordination
        """,
        inline=False
    )
    
    # Framework info
    embed.add_field(
        name="⚡ **LangChain Multi-Agent**",
        value="""
• **Multi-agent orchestration** - 5 specialized agents
• **Tool-based analysis** - Each agent has specific tools
• **Most detailed analysis** - Comprehensive coverage
• **Structured workflow** - Coordinated agent execution
• **Slower but thorough** - Trade depth for speed
        """,
        inline=False
    )
    
    # Tips
    embed.add_field(
        name="💭 **Pro Tips**",
        value="""
• Use any valid stock ticker (e.g., MSFT, AAPL, TSLA)
• This bot provides the most detailed analysis
• Response times are slower but more comprehensive
• Use `!agents` to see agent system details
• Great for thorough investment research
        """,
        inline=False
    )
    
    embed.set_footer(text=f"LangChain Multi-Agent • {provider.upper()} • Real-time market data")
    await ctx.send(embed=embed)

@bot.command(name='status')
async def status_command(ctx):
    """Show bot status"""
    embed = discord.Embed(
        title="🤖 pookan-langchain-multi Status",
        description="Bot is running and ready for stock analysis",
        color=0x00ff00
    )
    
    embed.add_field(
        name="✅ Status",
        value="Online and Ready",
        inline=True
    )
    
    embed.add_field(
        name="⚡ Framework",
        value="LangChain Multi-Agent",
        inline=True
    )
    
    embed.add_field(
        name="🎯 Capability",
        value="Multi-Agent Stock Analysis (Any Ticker)",
        inline=True
    )
    
    embed.add_field(
        name="🤖 Provider",
        value=provider.upper(),
        inline=True
    )
    
    embed.add_field(
        name="💡 Commands",
        value="!analyze, !agents, !stockhelp, !status",
        inline=True
    )
    
    embed.set_footer(text=f"pookan-langchain-multi • {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    await ctx.send(embed=embed)

def main():
    """Run the Discord bot"""
    token = os.getenv("MULTI_LANGCHAIN_DISCORD_TOKEN")
    if not token:
        print("❌ Error: MULTI_LANGCHAIN_DISCORD_TOKEN not found in environment variables")
        print("💡 Make sure MULTI_LANGCHAIN_DISCORD_TOKEN is set in environment variables")
        print("💡 Current environment variables:")
        print(f"   - MULTI_LANGCHAIN_DISCORD_TOKEN: {'Set' if os.getenv('MULTI_LANGCHAIN_DISCORD_TOKEN') else 'Not set'}")
        print(f"   - ANTHROPIC_API_KEY: {'Set' if os.getenv('ANTHROPIC_API_KEY') else 'Not set'}")
        print(f"   - OPENAI_API_KEY: {'Set' if os.getenv('OPENAI_API_KEY') else 'Not set'}")
        return
    
    print("🚀 Starting pookan-langchain-multi Discord bot...")
    print("📊 Bot will be ready to analyze any stocks!")
    print(f"✅ Environment variables loaded successfully")
    print(f"🤖 Provider: {provider.upper()}")
    bot.run(token)

if __name__ == "__main__":
    main() 