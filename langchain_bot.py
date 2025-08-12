"""
Discord Bot: pookan-langchain (Slash Commands Version)
Tesla Stock Recommendation using LangChain Single Agent
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

# Initialize agent
agent = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

@bot.event
async def on_ready():
    """Bot startup event"""
    print(f'🤖 {bot.user} has connected to Discord!')
    print(f'📊 LangChain Stock Bot (Slash Commands) is ready!')
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
@bot.tree.command(name="analyze", description="Analyze any stock using LangChain single agent")
async def analyze_stock_slash(interaction: discord.Interaction, ticker: str, query: str = None):
    """Slash command handler for stock analysis"""
    await interaction.response.defer()
    
    try:
        # Send initial message
        embed = discord.Embed(
            title="📊 Stock Analysis in Progress",
            description=f"Analyzing **{ticker.upper()}**...",
            color=0x00ff00
        )
        embed.add_field(name="Status", value="🔄 Fetching data and analyzing...", inline=False)
        message = await interaction.followup.send(embed=embed)
        
        # Prepare the analysis query
        analysis_query = f"Analyze {ticker.upper()} stock"
        if query:
            analysis_query += f" with focus on: {query}"
        
        # Get market data first using the MarketDataTool
        market_tool = MarketDataTool()
        market_data_result = market_tool.run(ticker)
        
        # Get recommendation from agent
        result = agent.run(analysis_query)
        
        # Create detailed response embed
        embed = discord.Embed(
            title=f"📊 {ticker.upper()} Stock Analysis - LangChain Single Agent",
            description="Analysis completed using LangChain single agent system",
            color=0x00ff00
        )
        
        # Add market data - extract from market data result
        try:
            print(f"DEBUG - Market data result: {market_data_result}")
            # Look for market data in the market data result
            market_data_match = re.search(r'([A-Z]+) Market Data: Price=\$([^,]+), Change=([^,]+), Volume=([0-9,]+),', market_data_result)
            print(f"DEBUG - Market data match: {market_data_match}")
            if market_data_match:
                price_str = market_data_match.group(2)
                change_str = market_data_match.group(3)
                volume_str = market_data_match.group(4)
                print(f"DEBUG - Extracted: price={price_str}, change={change_str}, volume={volume_str}")
            else:
                price_str = None
                change_str = None
                volume_str = None
                print("DEBUG - No market data match found")
            
            if price_str and change_str and volume_str:
                embed.add_field(
                    name="💰 Current Price", 
                    value=f"${price_str}", 
                    inline=True
                )
                embed.add_field(
                    name="📈 Change", 
                    value=change_str, 
                    inline=True
                )
                embed.add_field(
                    name="📊 Volume", 
                    value=volume_str, 
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
        
        # Determine recommendation type and confidence
        result_lower = result.lower()
        if "buy" in result_lower:
            recommendation_type = "BUY"
            confidence = "High" if "strong" in result_lower or "recommend" in result_lower else "Medium"
        elif "sell" in result_lower:
            recommendation_type = "SELL"
            confidence = "High" if "strong" in result_lower else "Medium"
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
        
        # Add reasoning (truncated result)
        result_short = result[:1024] if len(result) > 1024 else result
        embed.add_field(
            name="💭 Reasoning", 
            value=result_short, 
            inline=False
        )
        
        # Add workflow status and framework info
        embed.add_field(
            name="🔄 Workflow Status", 
            value="Completed", 
            inline=True
        )
        embed.add_field(
            name="⚡ Framework", 
            value="LangChain Single Agent", 
            inline=True
        )
        embed.add_field(
            name="⏰ Completed", 
            value=datetime.now().strftime("%H:%M:%S"), 
            inline=True
        )
        
        embed.set_footer(text=f"pookan-langchain • {provider.upper()} • Real-time market data")
        await message.edit(embed=embed)
        
    except Exception as e:
        error_embed = discord.Embed(
            title="❌ Error",
            description=f"Failed to analyze {ticker.upper()}: {str(e)}",
            color=0xff0000
        )
        await interaction.followup.send(embed=error_embed)

@bot.tree.command(name="help", description="Show comprehensive help information")
async def help_command_slash(interaction: discord.Interaction):
    """Slash command handler for help information"""
    embed = discord.Embed(
        title="🤖 LangChain Stock Analysis Bot - Help Guide",
        description="**AI-powered stock analysis using LangChain single agent and real-time market data**\n\nThis bot provides comprehensive stock analysis with AI recommendations using the latest market data.",
        color=0x0099ff
    )
    
    # Basic commands
    embed.add_field(
        name="📊 **Basic Commands**",
        value="""
`/analyze <ticker>` - Analyze any stock (e.g., `/analyze MSFT`)
`/analyze <ticker> <query>` - Specific analysis with custom query
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
    await interaction.response.send_message(embed=embed)

@bot.tree.command(name="status", description="Show bot status and configuration")
async def status_command_slash(interaction: discord.Interaction):
    """Slash command handler for status"""
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
        value="/analyze, /help, /status",
        inline=True
    )
    
    embed.set_footer(text=f"pookan-langchain • {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    await interaction.response.send_message(embed=embed)

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
    print(f"💡 Use /analyze <ticker> to analyze a stock")
    print(f"💡 Use /help to see comprehensive help")
    print(f"💡 Use /status to see bot configuration")
    
    try:
        bot.run(token)
    except Exception as e:
        print(f"❌ Failed to start bot: {e}")

if __name__ == "__main__":
    main()
