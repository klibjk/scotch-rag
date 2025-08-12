"""
Discord Bot: pookan-langchain-multi (Slash Commands Version)
Tesla Stock Recommendation using LangChain Multi-Agent System
Self-contained implementation with slash commands
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

# Slash command configuration
GUILD_ID = None  # Set to your guild ID for faster command registration, or None for global

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
                        else:
                            return f"No data available for {ticker}"
                    
                    # Calculate basic metrics
                    current_price = hist['Close'].iloc[-1]
                    previous_price = hist['Close'].iloc[-2]
                    price_change = current_price - previous_price
                    price_change_percent = (price_change / previous_price) * 100
                    volume = hist['Volume'].iloc[-1]
                    
                    # Calculate technical indicators
                    # RSI
                    delta = hist['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))
                    current_rsi = rsi.iloc[-1]
                    
                    # Moving averages
                    sma_5 = hist['Close'].rolling(window=5).mean().iloc[-1]
                    sma_20 = hist['Close'].rolling(window=20).mean().iloc[-1]
                    sma_50 = hist['Close'].rolling(window=50).mean().iloc[-1]
                    
                    # MACD
                    ema_12 = hist['Close'].ewm(span=12).mean()
                    ema_26 = hist['Close'].ewm(span=26).mean()
                    macd_line = ema_12 - ema_26
                    signal_line = macd_line.ewm(span=9).mean()
                    current_macd = macd_line.iloc[-1]
                    current_signal = signal_line.iloc[-1]
                    
                    # Bollinger Bands
                    bb_sma = hist['Close'].rolling(window=20).mean()
                    bb_std = hist['Close'].rolling(window=20).std()
                    upper_band = bb_sma + (bb_std * 2)
                    lower_band = bb_sma - (bb_std * 2)
                    
                    analysis = f"{ticker} Market Data: Price=${current_price:.2f} ({price_change_percent:+.2f}%), Volume={volume:,.0f}, RSI={current_rsi:.1f}, SMA20=${sma_20:.2f}, MACD={current_macd:.2f}"
                    
                    return analysis
                    
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
    description: str = "Perform detailed technical analysis including RSI, MACD, Bollinger Bands, and moving averages"
    
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
                        else:
                            return f"No data available for {ticker}"
                    
                    current_price = hist['Close'].iloc[-1]
                    
                    # RSI
                    delta = hist['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))
                    current_rsi = rsi.iloc[-1]
                    
                    # Moving averages
                    sma_5 = hist['Close'].rolling(window=5).mean().iloc[-1]
                    sma_20 = hist['Close'].rolling(window=20).mean().iloc[-1]
                    sma_50 = hist['Close'].rolling(window=50).mean().iloc[-1]
                    
                    # MACD
                    ema_12 = hist['Close'].ewm(span=12).mean()
                    ema_26 = hist['Close'].ewm(span=26).mean()
                    macd_line = ema_12 - ema_26
                    signal_line = macd_line.ewm(span=9).mean()
                    current_macd = macd_line.iloc[-1]
                    current_signal = signal_line.iloc[-1]
                    
                    # Bollinger Bands
                    bb_sma = hist['Close'].rolling(window=20).mean()
                    bb_std = hist['Close'].rolling(window=20).std()
                    upper_band = bb_sma + (bb_std * 2)
                    lower_band = bb_sma - (bb_std * 2)
                    
                    # Generate signals
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
                        else:
                            return f"No data available for {ticker}"
                    
                    # Calculate volatility
                    returns = hist['Close'].pct_change().dropna()
                    volatility = returns.std() * (252 ** 0.5)  # Annualized volatility
                    
                    # Beta calculation (simplified)
                    market_returns = yf.Ticker("^GSPC").history(period="30d")['Close'].pct_change().dropna()
                    if len(returns) > 0 and len(market_returns) > 0:
                        min_len = min(len(returns), len(market_returns))
                        correlation = returns.iloc[-min_len:].corr(market_returns.iloc[-min_len:])
                        beta = correlation * (volatility / (market_returns.std() * (252 ** 0.5)))
                    else:
                        beta = 1.0
                    
                    # Risk factors
                    debt_to_equity = info.get('debtToEquity', 0)
                    current_ratio = info.get('currentRatio', 0)
                    profit_margins = info.get('profitMargins', 0)
                    
                    # Risk assessment
                    volatility_risk = "High" if volatility > 0.4 else "Medium" if volatility > 0.2 else "Low"
                    beta_risk = "High" if abs(beta) > 1.5 else "Medium" if abs(beta) > 1.0 else "Low"
                    debt_risk = "High" if debt_to_equity > 1 else "Medium" if debt_to_equity > 0.5 else "Low"
                    liquidity_risk = "High" if current_ratio < 1 else "Medium" if current_ratio < 2 else "Low"
                    
                    analysis = f"{ticker} Risk Assessment: Volatility={volatility:.1%} ({volatility_risk}), Beta={beta:.2f} ({beta_risk}), Debt Risk={debt_risk}, Liquidity Risk={liquidity_risk}"
                    
                    return analysis
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(1)
                        continue
                    else:
                        raise e
                        
        except Exception as e:
            return f"Error in risk assessment: {str(e)}"

class DecisionMakerTool(BaseTool):
    name: str = "make_recommendation"
    description: str = "Synthesize all analysis results and provide final investment recommendation"
    
    def _run(self, query: str) -> str:
        """Make final recommendation"""
        try:
            # This tool will be used by the LLM to synthesize all previous analysis
            # The LLM will have access to all previous tool outputs
            return "Final recommendation will be synthesized from all analysis results"
        except Exception as e:
            return f"Error in decision making: {str(e)}"

# Create tool groups for different agents
data_fetcher_tools = [DataFetcherTool()]
technical_analyst_tools = [TechnicalAnalystTool()]
fundamental_analyst_tools = [FundamentalAnalystTool()]
risk_assessor_tools = [RiskAssessorTool()]
decision_maker_tools = [DecisionMakerTool()]

# Create agents using LangChain's ReAct framework
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate

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
            
            # Step 5: Decision Maker Agent - Synthesize all results
            decision_prompt = f"""
            Based on the following analysis from multiple specialized agents, provide a comprehensive investment recommendation for {ticker}:
            
            Data Analysis: {agent_messages[0]['message']}
            Technical Analysis: {agent_messages[1]['message']}
            Fundamental Analysis: {agent_messages[2]['message']}
            Risk Assessment: {agent_messages[3]['message']}
            
            User Query: {query or f"General {ticker} stock analysis"}
            
            Please provide:
            1. A clear BUY/HOLD/SELL recommendation
            2. Confidence level (High/Medium/Low)
            3. Risk level assessment
            4. Detailed reasoning based on all the analysis above
            5. Key factors that influenced your decision
            
            Format your response as a structured analysis.
            """
            
            decision_result = self.agents['decision_maker'].invoke({"input": decision_prompt})
            agent_messages.append({
                'agent': 'DecisionMaker',
                'message': decision_result.get("output", str(decision_result))
            })
            
            # Parse the final recommendation
            final_recommendation = self._parse_recommendation(decision_result.get("output", ""))
            
            # Get market data for response
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="30d")
                current_price = hist['Close'].iloc[-1] if not hist.empty else 0
                volume = hist['Volume'].iloc[-1] if not hist.empty else 0
                
                # Calculate price change
                if len(hist) > 1:
                    price_change = hist['Close'].iloc[-1] - hist['Close'].iloc[-2]
                    price_change_percent = (price_change / hist['Close'].iloc[-2]) * 100
                else:
                    price_change_percent = 0
                
                market_data = {
                    "current_price": current_price,
                    "price_change_percent": price_change_percent,
                    "volume": volume
                }
            except:
                market_data = {}
            
            return {
                "ticker": ticker,
                "final_recommendation": final_recommendation,
                "market_data": market_data,
                "agent_messages": agent_messages,
                "workflow_execution": "Multi-Agent Orchestration Completed",
                "provider": self.provider
            }
            
        except Exception as e:
            return {
                "error": f"Error in multi-agent analysis: {str(e)}",
                "ticker": ticker
            }
    
    def _parse_recommendation(self, decision_text: str) -> Dict[str, Any]:
        """Parse the decision text to extract structured recommendation"""
        try:
            # Simple parsing - in a real implementation, you might use more sophisticated parsing
            decision_text_lower = decision_text.lower()
            
            # Extract recommendation
            if "buy" in decision_text_lower:
                recommendation = "BUY"
            elif "sell" in decision_text_lower:
                recommendation = "SELL"
            else:
                recommendation = "HOLD"
            
            # Extract confidence
            if "high confidence" in decision_text_lower or "high" in decision_text_lower:
                confidence = "High"
            elif "low confidence" in decision_text_lower or "low" in decision_text_lower:
                confidence = "Low"
            else:
                confidence = "Medium"
            
            # Extract risk level
            if "high risk" in decision_text_lower:
                risk_level = "High"
            elif "low risk" in decision_text_lower:
                risk_level = "Low"
            else:
                risk_level = "Medium"
            
            return {
                "recommendation": recommendation,
                "confidence": confidence,
                "risk_level": risk_level,
                "reasoning": decision_text
            }
            
        except Exception as e:
            return {
                "recommendation": "UNKNOWN",
                "confidence": "UNKNOWN",
                "risk_level": "UNKNOWN",
                "reasoning": f"Error parsing recommendation: {str(e)}"
            }

# Initialize the stock recommendation system
stock_system = MultiAgentStockRecommendationSystem()

@bot.event
async def on_ready():
    """Bot startup event"""
    print(f'🤖 {bot.user} has connected to Discord!')
    print(f'📊 LangChain Multi-Agent Stock Bot (Slash Commands) is ready!')
    print(f'💬 Use /analyze <ticker> to get stock recommendations')

# Slash command handlers
@bot.tree.command(name="analyze", description="Analyze any stock using LangChain multi-agent system")
async def analyze_stock_slash(interaction: discord.Interaction, ticker: str, query: str = None):
    """Slash command handler for stock analysis"""
    await interaction.response.defer()
    
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
    
    message = await interaction.followup.send(embed=embed)
    
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
            await interaction.followup.send(embed=error_embed)
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
                value=f"${market_data['current_price']:.2f}", 
                inline=True
            )
            response_embed.add_field(
                name="📈 Change", 
                value=f"{market_data.get('price_change_percent', 0):.2f}%", 
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
        
        response_embed.set_footer(text=f"pookan-langchain-multi • {provider.upper()} • Real-time market data")
        await message.edit(embed=response_embed)
        
    except Exception as e:
        error_embed = discord.Embed(
            title="❌ Analysis Failed",
            description=f"An error occurred during analysis: {str(e)}",
            color=0xff0000
        )
        await interaction.followup.send(embed=error_embed)

@bot.tree.command(name="agents", description="Show detailed agent system information")
async def agents_command_slash(interaction: discord.Interaction):
    """Slash command handler for agents information"""
    embed = discord.Embed(
        title="🤖 LangChain Multi-Agent System",
        description="**5 Specialized Agents Working Together**\n\nThis bot uses a sophisticated multi-agent system where each agent has a specific role in the analysis process.",
        color=0x0099ff
    )
    
    # Agent details
    embed.add_field(
        name="🔍 **Data Fetcher Agent**",
        value="• Fetches real-time market data\n• Retrieves historical price data\n• Collects volume and trading information\n• Provides data foundation for analysis",
        inline=False
    )
    
    embed.add_field(
        name="📊 **Technical Analyst Agent**",
        value="• Calculates technical indicators (RSI, MACD, Bollinger Bands)\n• Analyzes price patterns and trends\n• Identifies support/resistance levels\n• Provides technical signals",
        inline=False
    )
    
    embed.add_field(
        name="💰 **Fundamental Analyst Agent**",
        value="• Analyzes company financial metrics\n• Evaluates P/E ratios, debt levels\n• Assesses growth potential and profitability\n• Reviews company fundamentals",
        inline=False
    )
    
    embed.add_field(
        name="⚠️ **Risk Assessor Agent**",
        value="• Evaluates investment risk factors\n• Analyzes volatility and market conditions\n• Assesses company-specific risks\n• Provides risk level assessment",
        inline=False
    )
    
    embed.add_field(
        name="🎯 **Decision Maker Agent**",
        value="• Synthesizes all analysis results\n• Weighs technical vs fundamental factors\n• Considers risk vs reward\n• Provides final recommendation",
        inline=False
    )
    
    # Workflow
    embed.add_field(
        name="🔄 **Analysis Workflow**",
        value="1. **Data Collection** → 2. **Technical Analysis** → 3. **Fundamental Analysis** → 4. **Risk Assessment** → 5. **Decision Synthesis**",
        inline=False
    )
    
    # Benefits
    embed.add_field(
        name="✨ **Benefits of Multi-Agent System**",
        value="• **Specialized Expertise** - Each agent focuses on specific analysis\n• **Comprehensive Coverage** - All aspects of stock analysis covered\n• **Reduced Bias** - Multiple perspectives reduce single-agent bias\n• **Structured Process** - Systematic approach to analysis\n• **Detailed Insights** - Most thorough analysis available",
        inline=False
    )
    
    embed.set_footer(text="pookan-langchain-multi • LangChain Multi-Agent")
    await interaction.response.send_message(embed=embed)

@bot.tree.command(name="help", description="Show comprehensive help information")
async def help_command_slash(interaction: discord.Interaction):
    """Slash command handler for help information"""
    embed = discord.Embed(
        title="🤖 LangChain Multi-Agent Bot - Help Guide",
        description="**AI-powered stock analysis using LangChain multi-agent orchestration**\n\nThis bot provides the most detailed stock analysis using multiple specialized agents working together.",
        color=0x0099ff
    )
    
    # Basic commands
    embed.add_field(
        name="📊 **Basic Commands**",
        value="""
`/analyze <ticker>` - Analyze any stock (e.g., `/analyze MSFT`)
`/analyze <ticker> <query>` - Specific analysis with custom query
`/agents` - Show detailed agent system information
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
• Use `/agents` to see agent system details
• Great for thorough investment research
        """,
        inline=False
    )
    
    embed.set_footer(text=f"LangChain Multi-Agent • {provider.upper()} • Real-time market data")
    await interaction.response.send_message(embed=embed)

@bot.tree.command(name="status", description="Show bot status and configuration")
async def status_command_slash(interaction: discord.Interaction):
    """Slash command handler for status information"""
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
        value="/analyze, /agents, /help, /status",
        inline=True
    )
    
    embed.set_footer(text=f"pookan-langchain-multi • {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    await interaction.response.send_message(embed=embed)

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
    
    print(f"🚀 Starting LangChain Multi-Agent Discord Bot...")
    print(f"🤖 Provider: {provider.upper()}")
    print(f"💡 Use /analyze <ticker> to analyze a stock")
    print(f"💡 Use /help to see comprehensive help")
    print(f"💡 Use /agents to see agent system details")
    print(f"💡 Use /status to see bot configuration")
    
    try:
        bot.run(token)
    except Exception as e:
        print(f"❌ Failed to start bot: {e}")

if __name__ == "__main__":
    main()
