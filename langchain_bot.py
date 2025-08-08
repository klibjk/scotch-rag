"""
Discord Bot: pookan-langchain
Tesla Stock Recommendation using LangChain Single Agent
Self-contained implementation
"""

import discord
from discord.ext import commands
import asyncio
import json
from datetime import datetime
import os
import warnings
from typing import Dict, Any, Optional
import yfinance as yf
import pandas as pd
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage
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

# Self-contained LangChain Single Agent Implementation
class MarketDataTool(BaseTool):
    name: str = "get_market_data"
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
                    
                    # Calculate RSI
                    delta = hist['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))
                    current_rsi = rsi.iloc[-1]
                    
                    # Calculate moving averages
                    sma_20 = hist['Close'].rolling(window=20).mean().iloc[-1]
                    sma_50 = hist['Close'].rolling(window=50).mean().iloc[-1]
                    
                    market_data = {
                        "current_price": round(current_price, 2),
                        "price_change_percent": round(price_change, 2),
                        "volume": int(volume),
                        "rsi": round(current_rsi, 2),
                        "sma_20": round(sma_20, 2),
                        "sma_50": round(sma_50, 2)
                    }
                    
                    return f"{ticker} Market Data: Price=${market_data['current_price']}, Change={market_data['price_change_percent']}%, Volume={market_data['volume']:,}, RSI={market_data['rsi']}, 20SMA=${market_data['sma_20']}, 50SMA=${market_data['sma_50']}"
                    
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
    description: str = "Perform technical analysis on any stock using various indicators"
    
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
                    
                    # Analysis
                    rsi_signal = "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral"
                    ma_signal = "Bullish" if current_price > sma_20 > sma_50 else "Bearish" if current_price < sma_20 < sma_50 else "Neutral"
                    macd_signal = "Bullish" if current_macd > current_signal else "Bearish"
                    
                    analysis = f"{ticker} Technical Analysis: RSI={current_rsi:.1f} ({rsi_signal}), Price vs 20SMA={'Above' if current_price > sma_20 else 'Below'}, MACD Signal={macd_signal}"
                    
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
                    
                    # Analysis
                    pe_assessment = "High" if pe_ratio > 50 else "Reasonable" if pe_ratio > 20 else "Low"
                    debt_assessment = "High" if debt_to_equity > 1 else "Manageable"
                    margin_assessment = "Strong" if profit_margins > 0.1 else "Weak" if profit_margins < 0 else "Moderate"
                    
                    analysis = f"{ticker} Fundamental Analysis: Market Cap=${market_cap/1e9:.1f}B, P/E={pe_ratio:.1f} ({pe_assessment}), Debt/Equity={debt_to_equity:.2f} ({debt_assessment}), Profit Margin={profit_margins:.1%} ({margin_assessment})"
                    
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

# Initialize agent with modern approach
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

# Create the agent
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

class SingleStockRecommendationAgent:
    """Self-contained LangChain Single Agent for stock recommendations"""
    
    def __init__(self):
        self.agent = agent_executor
        self.provider = provider
    
    def get_recommendation(self, ticker: str = "TSLA", query: str = None) -> Dict[str, Any]:
        """Get stock recommendation using unified LangChain agent"""
        try:
            # Extract ticker from query if provided
            if query and len(query.strip()) > 0:
                words = query.strip().upper().split()
                for word in words:
                    if len(word) <= 5 and word.isalpha():
                        ticker = word
                        break
            
            # Prepare the query
            if not query:
                query = f"Analyze {ticker} stock and provide a comprehensive investment recommendation"
            else:
                query = f"Analyze {ticker} stock based on this query: {query}. Provide a comprehensive investment recommendation."
            
            # Get market data first
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
            
            # Get recommendation from agent
            result = self.agent.invoke({"input": query})
            
            return {
                "output": result.get("output", str(result)),
                "market_data": market_data,
                "framework": "LangChain Single Agent",
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
                "framework": "LangChain Single Agent",
                "provider": self.provider
            }

# Initialize the system
stock_system = SingleStockRecommendationAgent()

@bot.event
async def on_ready():
    """Bot startup event"""
    print(f'🤖 {bot.user} has connected to Discord!')
    print(f'📊 LangChain Single Agent Stock Bot is ready!')
    print(f'💬 Use !analyze <ticker> to get stock recommendations')

@bot.command(name='analyze')
async def analyze_stock(ctx, ticker: str, *, query: str = None):
    """
    Get stock recommendation using LangChain single agent
    
    Usage: !analyze <ticker> [optional query]
    Examples:
        !analyze TSLA
        !analyze AAPL should I buy Apple stock?
        !analyze MSFT analyze Microsoft fundamentals
    """
    
    # Send initial response
    embed = discord.Embed(
        title="🤖 pookan-langchain Stock Analysis",
        description=f"Analyzing {ticker.upper()} stock using LangChain single agent...",
        color=0x00ff00
    )
    embed.add_field(name="🔍 Ticker", value=ticker.upper(), inline=True)
    embed.add_field(name="🔍 Query", value=query or f"General {ticker.upper()} stock analysis", inline=True)
    embed.add_field(name="⚡ Framework", value="LangChain Single Agent", inline=True)
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
        recommendation = result.get("output", "")
        market_data = result.get("market_data", {})
        
        # Create detailed response embed
        response_embed = discord.Embed(
            title=f"📊 {ticker.upper()} Stock Analysis - LangChain Single Agent",
            description="Analysis completed using unified LangChain agent",
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
        
        # Add recommendation
        response_embed.add_field(
            name="🎯 Recommendation", 
            value=recommendation[:1024] if len(recommendation) > 1024 else recommendation, 
            inline=False
        )
        
        # Add metadata
        response_embed.add_field(
            name="⚡ Framework", 
            value="LangChain Single Agent", 
            inline=True
        )
        response_embed.add_field(
            name="⏰ Completed", 
            value=datetime.now().strftime("%H:%M:%S"), 
            inline=True
        )
        
        # Add footer
        response_embed.set_footer(text="pookan-langchain • LangChain Single Agent")
        
        await message.edit(embed=response_embed)
        
    except Exception as e:
        # Handle any errors
        error_embed = discord.Embed(
            title="❌ Analysis Failed",
            description=f"An error occurred: {str(e)}",
            color=0xff0000
        )
        await message.edit(embed=error_embed)

@bot.command(name='stockhelp')
async def help_command(ctx):
    """Show comprehensive help information"""
    embed = discord.Embed(
        title="🤖 LangChain Single Agent Bot - Help Guide",
        description="**AI-powered stock analysis using LangChain unified agent approach**\n\nThis bot provides comprehensive stock analysis with a single intelligent agent that handles all aspects of analysis.",
        color=0x0099ff
    )
    
    # Basic commands
    embed.add_field(
        name="📊 **Basic Commands**",
        value="""
`!analyze <ticker>` - Analyze any stock (e.g., `!analyze MSFT`)
`!analyze <ticker> <query>` - Specific analysis with custom query
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
• **Unified agent approach** - Single intelligent agent handles everything
• **Real-time market data** from Yahoo Finance
• **Technical analysis** (RSI, MACD, Moving Averages)
• **Fundamental analysis** (P/E ratios, debt levels, margins)
• **Risk assessment** (volatility, beta, risk factors)
• **Comprehensive recommendations** with detailed reasoning
        """,
        inline=False
    )
    
    # Framework info
    embed.add_field(
        name="⚡ **LangChain Single Agent**",
        value="""
• **Unified reasoning** - One agent handles all analysis types
• **Tool integration** - Seamless access to market data tools
• **Balanced approach** - Good balance of speed and detail
• **Reliable performance** - Consistent analysis quality
• **Error handling** - Graceful handling of invalid inputs
        """,
        inline=False
    )
    
    # Tips
    embed.add_field(
        name="💭 **Pro Tips**",
        value="""
• Use any valid stock ticker (e.g., MSFT, AAPL, TSLA)
• Add specific queries for targeted analysis
• This bot provides balanced analysis depth
• Response times are moderate but reliable
• Great for general stock analysis needs
        """,
        inline=False
    )
    
    embed.set_footer(text=f"LangChain Single Agent • {provider.upper()} • Real-time market data")
    await ctx.send(embed=embed)

@bot.command(name='status')
async def status_command(ctx):
    """Show bot status"""
    embed = discord.Embed(
        title="🤖 pookan-langchain Status",
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
        value="LangChain Single Agent",
        inline=True
    )
    
    embed.add_field(
        name="🎯 Capability",
        value="Stock Analysis (Any Ticker)",
        inline=True
    )
    
    embed.add_field(
        name="🤖 Provider",
        value=provider.upper(),
        inline=True
    )
    
    embed.add_field(
        name="💡 Commands",
        value="!analyze, !stockhelp, !status",
        inline=True
    )
    
    embed.set_footer(text=f"pookan-langchain • {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    await ctx.send(embed=embed)

def main():
    """Run the Discord bot"""
    token = os.getenv("LANGCHAIN_DISCORD_TOKEN")
    if not token:
        print("❌ Error: LANGCHAIN_DISCORD_TOKEN not found in environment variables")
        print("💡 Make sure LANGCHAIN_DISCORD_TOKEN is set in environment variables")
        print("💡 Current environment variables:")
        print(f"   - LANGCHAIN_DISCORD_TOKEN: {'Set' if os.getenv('LANGCHAIN_DISCORD_TOKEN') else 'Not set'}")
        print(f"   - ANTHROPIC_API_KEY: {'Set' if os.getenv('ANTHROPIC_API_KEY') else 'Not set'}")
        print(f"   - OPENAI_API_KEY: {'Set' if os.getenv('OPENAI_API_KEY') else 'Not set'}")
        return
    
    print("🚀 Starting pookan-langchain Discord bot...")
    print("📊 Bot will be ready to analyze any stocks!")
    print(f"✅ Environment variables loaded successfully")
    print(f"🤖 Provider: {provider.upper()}")
    bot.run(token)

if __name__ == "__main__":
    main() 