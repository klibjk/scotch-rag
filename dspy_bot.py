import discord
from discord.ext import commands
import dspy
import yfinance as yf
import json
from typing import Dict, Any
from dataclasses import dataclass
import os
import warnings
from datetime import datetime
from dotenv import load_dotenv

# Suppress Pydantic warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# Load environment variables
load_dotenv()

# Configure DSPy with fallback mechanism: Anthropic first, OpenAI second
def configure_dspy():
    """Configure DSPy with fallback mechanism"""
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if anthropic_key:
        try:
            print("🔵 Using Anthropic Claude Sonnet as primary LLM")
            lm = dspy.LM("anthropic/claude-sonnet-4-0", api_key=anthropic_key)
            dspy.configure(lm=lm)
            return "anthropic"
        except Exception as e:
            print(f"⚠️ Anthropic configuration failed: {e}")
            if openai_key:
                try:
                    print("🟢 Falling back to OpenAI GPT-4")
                    lm = dspy.LM("openai/gpt-4", api_key=openai_key)
                    dspy.configure(lm=lm)
                    return "openai"
                except Exception as e2:
                    print(f"❌ OpenAI fallback also failed: {e2}")
                    raise Exception("Both Anthropic and OpenAI configurations failed")
            else:
                raise Exception("Anthropic failed and no OpenAI API key available")
    elif openai_key:
        try:
            print("🟢 Using OpenAI GPT-4 as primary LLM")
            lm = dspy.LM("openai/gpt-4", api_key=openai_key)
            dspy.configure(lm=lm)
            return "openai"
        except Exception as e:
            print(f"❌ OpenAI configuration failed: {e}")
            raise Exception("OpenAI configuration failed")
    else:
        raise Exception("No API keys found. Please set either ANTHROPIC_API_KEY or OPENAI_API_KEY")

# Configure DSPy
try:
    provider = configure_dspy()
    print(f"✅ DSPy configured successfully with {provider.upper()}")
except Exception as e:
    print(f"❌ DSPy configuration failed: {e}")
    print("💡 Please check your API keys and try again")
    exit(1)

# Discord bot setup
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents, help_command=None)

@dataclass
class StockData:
    """Data structure for stock information"""
    price: float
    change: str
    volume: str
    rsi: float
    sma_5: float
    sma_20: float

class AnalyzeStockModule(dspy.Module):
    """Module 1: Extract interpretable financial insights from raw stock metrics"""
    
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict("stock_data -> analysis_summary")
    
    def forward(self, stock_data: str) -> str:
        """Analyze stock data and return qualitative summary"""
        return self.predictor(stock_data=stock_data).analysis_summary

class DecisionModule(dspy.Module):
    """Module 2: Use analysis to decide recommendation"""
    
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict("analysis_summary -> recommendation, reasoning")
    
    def forward(self, analysis_summary: str) -> Dict[str, str]:
        """Make recommendation decision based on analysis"""
        result = self.predictor(analysis_summary=analysis_summary)
        return {
            "recommendation": result.recommendation,
            "reasoning": result.reasoning
        }

class ExplainModule(dspy.Module):
    """Module 3: Convert structured decision into natural language explanation"""
    
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict("recommendation, reasoning -> final_explanation")
    
    def forward(self, recommendation: str, reasoning: str) -> str:
        """Generate natural language explanation"""
        return self.predictor(recommendation=recommendation, reasoning=reasoning).final_explanation

class StockRecommenderAgent(dspy.Module):
    """Single-Agent Stock Recommendation System using DSPy"""
    
    def __init__(self):
        super().__init__()
        self.analyze_module = AnalyzeStockModule()
        self.decision_module = DecisionModule()
        self.explain_module = ExplainModule()
    
    def forward(self, stock_data: str) -> Dict[str, Any]:
        """
        Main pipeline: Analyze -> Decide -> Explain
        
        Args:
            stock_data: Stringified stock data (e.g., "price=255.3, change=+1.2%, volume=20M, RSI=63, 5dma=250, 20dma=245")
        
        Returns:
            Dictionary with recommendation, reasoning, and explanation
        """
        # Step 1: Analyze stock data
        analysis_summary = self.analyze_module(stock_data)
        
        # Step 2: Make decision
        decision_result = self.decision_module(analysis_summary)
        
        # Step 3: Generate explanation
        final_explanation = self.explain_module(
            recommendation=decision_result["recommendation"],
            reasoning=decision_result["reasoning"]
        )
        
        return {
            "recommendation": decision_result["recommendation"],
            "reasoning": decision_result["reasoning"],
            "explanation": final_explanation
        }

def get_stock_data(ticker: str) -> str:
    """Fetch stock data from Yahoo Finance with real calculations"""
    try:
        stock = yf.Ticker(ticker.upper())
        
        # Get historical data for calculations
        hist_data = stock.history(period="30d")
        
        if hist_data.empty:
            raise Exception("No historical data available")
        
        # Get current price and basic info
        current_price = hist_data['Close'].iloc[-1]
        previous_close = hist_data['Close'].iloc[-2]
        
        # Calculate change
        change_pct = ((current_price - previous_close) / previous_close) * 100
        change_str = f"{change_pct:+.2f}%"
        
        # Get volume
        volume = hist_data['Volume'].iloc[-1]
        volume_str = f"{volume/1000000:.1f}M" if volume > 1000000 else f"{volume/1000:.1f}K"
        
        # Calculate real RSI (14-period)
        def calculate_rsi(prices, period=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1]
        
        rsi = calculate_rsi(hist_data['Close'])
        
        # Calculate real moving averages
        sma_5 = hist_data['Close'].rolling(window=5).mean().iloc[-1]
        sma_20 = hist_data['Close'].rolling(window=20).mean().iloc[-1]
        
        stock_data = f"price={current_price:.2f}, change={change_str}, volume={volume_str}, RSI={rsi:.1f}, 5dma={sma_5:.2f}, 20dma={sma_20:.2f}"
        
        return stock_data
        
    except Exception as e:
        raise Exception(f"Error fetching stock data for {ticker}: {str(e)}")

# Initialize the agent
agent = StockRecommenderAgent()

@bot.event
async def on_ready():
    """Called when the bot is ready"""
    print(f'🤖 {bot.user} has connected to Discord!')
    print(f'📊 DSPy Stock Analysis Bot is ready!')
    print(f'🤖 Provider: {provider.upper()}')
    print(f'💡 Use !analyze <ticker> to analyze a stock')
    print(f'💡 Use !stockhelp to see comprehensive help')
    print(f'💡 Use !ping to test connectivity')
    print(f'💡 Use !status to see bot configuration')

@bot.command(name='analyze')
async def analyze_stock(ctx, ticker: str):
    """Analyze a stock ticker and provide recommendation"""
    try:
        # Send initial message
        embed = discord.Embed(
            title="📊 Stock Analysis in Progress",
            description=f"Analyzing **{ticker.upper()}**...",
            color=0x00ff00
        )
        embed.add_field(name="Status", value="🔄 Fetching data and analyzing...", inline=False)
        message = await ctx.send(embed=embed)
        
        # Get stock data
        stock_data = get_stock_data(ticker)
        
        # Update message with data
        embed.description = f"Analyzing **{ticker.upper()}**\n\n**Stock Data:**\n`{stock_data}`"
        embed.set_field_at(0, name="Status", value="🤖 Generating AI recommendation...", inline=False)
        await message.edit(embed=embed)
        
        # Get recommendation
        result = agent(stock_data)
        
        # Create final embed
        embed = discord.Embed(
            title=f"📈 Stock Analysis: {ticker.upper()}",
            description=f"**Stock Data:**\n`{stock_data}`",
            color=0x0099ff
        )
        
        # Add recommendation
        recommendation_color = 0x00ff00 if "buy" in result['recommendation'].lower() or "recommend" in result['recommendation'].lower() else 0xff9900
        embed.color = recommendation_color
        
        # Truncate fields to fit Discord's 1024 character limit
        recommendation = result['recommendation'][:1024] if len(result['recommendation']) > 1024 else result['recommendation']
        reasoning = result['reasoning'][:1024] if len(result['reasoning']) > 1024 else result['reasoning']
        explanation = result['explanation'][:1024] if len(result['explanation']) > 1024 else result['explanation']
        
        embed.add_field(
            name="🎯 Recommendation", 
            value=recommendation, 
            inline=False
        )
        
        embed.add_field(
            name="🧠 Reasoning", 
            value=reasoning, 
            inline=False
        )
        
        embed.add_field(
            name="📝 Explanation", 
            value=explanation, 
            inline=False
        )
        
        embed.add_field(
            name="🤖 AI Provider", 
            value=provider.upper(), 
            inline=True
        )
        
        embed.add_field(
            name="⚡ Framework", 
            value="DSPy AI", 
            inline=True
        )
        
        # Add footer
        embed.set_footer(text=f"Real-time data from Yahoo Finance • Powered by {provider.upper()}")
        
        await message.edit(embed=embed)
        
    except Exception as e:
        error_embed = discord.Embed(
            title="❌ Error",
            description=f"Failed to analyze {ticker.upper()}: {str(e)}",
            color=0xff0000
        )
        await ctx.send(embed=error_embed)

@bot.command(name='stockhelp')
async def help_command(ctx):
    """Show comprehensive help information"""
    embed = discord.Embed(
        title="🤖 DSPy Stock Analysis Bot - Help Guide",
        description="**AI-powered stock analysis using DSPy framework and real-time market data**\n\nThis bot provides instant stock analysis with AI recommendations using the latest market data.",
        color=0x0099ff
    )
    
    # Basic commands
    embed.add_field(
        name="📊 **Basic Commands**",
        value="""
`!analyze <ticker>` - Analyze any stock (e.g., `!analyze MSFT`)
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
• `!analyze TSLA` - Analyze Tesla stock  
• `!analyze GOOGL` - Analyze Google stock
• `!analyze NVDA` - Analyze NVIDIA stock
        """,
        inline=False
    )
    
    # Features
    embed.add_field(
        name="🎯 **What You Get**",
        value="""
• **Real-time stock data** from Yahoo Finance
• **AI-powered analysis** using DSPy framework
• **Buy/Sell/Hold recommendations**
• **Detailed reasoning** for each recommendation
• **Technical indicators** (RSI, Moving Averages)
• **Market data** (Price, Volume, Change)
        """,
        inline=False
    )
    
    # Framework info
    embed.add_field(
        name="⚡ **DSPy Framework**",
        value="""
• **Modular AI pipelines** for reliable analysis
• **Fast response times** compared to other frameworks
• **Clean, concise recommendations**
• **Real-time data integration**
• **Error handling** for invalid tickers
        """,
        inline=False
    )
    
    # Tips
    embed.add_field(
        name="💭 **Pro Tips**",
        value="""
• Use any valid stock ticker (e.g., MSFT, AAPL, TSLA)
• Analysis includes current price, volume, and technical indicators
• Recommendations are based on real-time market data
• Invalid tickers will show helpful error messages
• This bot is optimized for speed and reliability
        """,
        inline=False
    )
    
    embed.set_footer(text=f"DSPy Bot • {provider.upper()} • Real-time market data")
    await ctx.send(embed=embed)

@bot.command(name='ping')
async def ping(ctx):
    """Check bot latency"""
    embed = discord.Embed(
        title="🏓 Pong!",
        description=f"Bot latency: {round(bot.latency * 1000)}ms",
        color=0x00ff00
    )
    embed.add_field(name="🤖 Bot", value="DSPy Stock Analysis Bot", inline=True)
    embed.add_field(name="⚡ Framework", value="DSPy AI", inline=True)
    embed.add_field(name="🤖 Provider", value=provider.upper(), inline=True)
    embed.add_field(name="💡 Quick Start", value="Try `!analyze MSFT` to test!", inline=True)
    embed.set_footer(text=f"Real-time data from Yahoo Finance • Powered by {provider.upper()}")
    await ctx.send(embed=embed)

@bot.command(name='welcome')
async def welcome(ctx):
    """Show welcome message and quick start guide"""
    embed = discord.Embed(
        title="🤖 Welcome to DSPy Stock Analysis Bot!",
        description="**AI-powered stock analysis using DSPy framework**\n\nGet instant stock analysis with real-time market data and AI recommendations.",
        color=0x0099ff
    )
    
    embed.add_field(
        name="🚀 **Quick Start**",
        value="""
Try these commands to get started:
• `!analyze MSFT` - Analyze Microsoft stock
• `!analyze AAPL` - Analyze Apple stock
• `!stockhelp` - See all commands and features
• `!status` - Check bot configuration
        """,
        inline=False
    )
    
    embed.add_field(
        name="⚡ **DSPy Framework**",
        value="Fast, modular AI pipelines for reliable stock analysis",
        inline=True
    )
    
    embed.add_field(
        name="📊 **Real-time Data**",
        value="Live market data from Yahoo Finance",
        inline=True
    )
    
    embed.add_field(
        name="🎯 **What You Get**",
        value="• Stock recommendations\n• Technical analysis\n• Market data\n• AI reasoning",
        inline=True
    )
    
    embed.set_footer(text=f"DSPy Bot • {provider.upper()} • Ready for analysis!")
    await ctx.send(embed=embed)

@bot.command(name='status')
async def status_command(ctx):
    """Show bot status and provider information"""
    embed = discord.Embed(
        title="🤖 DSPy Stock Bot Status",
        description="Bot is running and ready for stock analysis",
        color=0x00ff00
    )
    
    embed.add_field(
        name="✅ Status",
        value="Online and Ready",
        inline=True
    )
    
    embed.add_field(
        name="🤖 Provider",
        value=provider.upper(),
        inline=True
    )
    
    embed.add_field(
        name="⚡ Framework",
        value="DSPy AI",
        inline=True
    )
    
    embed.add_field(
        name="🔄 Fallback",
        value="Anthropic → OpenAI",
        inline=True
    )
    
    embed.add_field(
        name="📊 Capability",
        value="Stock Analysis",
        inline=True
    )
    
    embed.add_field(
        name="💡 Commands",
        value="!analyze, !stockhelp, !ping, !status",
        inline=True
    )
    
    embed.set_footer(text=f"DSPy AI • {provider.upper()} • {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    await ctx.send(embed=embed)

def main():
    """Run the Discord bot"""
    token = os.getenv("DSPY_DISCORD_TOKEN")
    if not token:
        print("❌ Error: DSPY_DISCORD_TOKEN not found in environment variables")
        print("💡 Make sure DSPY_DISCORD_TOKEN is set in environment variables")
        print("💡 Current environment variables:")
        print(f"   - DSPY_DISCORD_TOKEN: {'Set' if os.getenv('DSPY_DISCORD_TOKEN') else 'Not set'}")
        print(f"   - ANTHROPIC_API_KEY: {'Set' if os.getenv('ANTHROPIC_API_KEY') else 'Not set'}")
        print(f"   - OPENAI_API_KEY: {'Set' if os.getenv('OPENAI_API_KEY') else 'Not set'}")
        return
    
    print("🚀 Starting Discord Stock Recommendation Bot...")
    print("📊 Bot will be ready to analyze stocks!")
    print(f"✅ Environment variables loaded successfully")
    print(f"🤖 Provider: {provider.upper()}")
    bot.run(token)

if __name__ == "__main__":
    main() 