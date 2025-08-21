# --- DSPy Discord Bot with Excel/DB Query Integration ---
# (This file is a copy of dspy_bot.py with /query command added)

import os
import warnings
from dotenv import load_dotenv
import traceback

# Set environment variable to suppress huggingface/tokenizers warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import discord
from discord.ext import commands
import dspy
import yfinance as yf
import json
from typing import Dict, Any
from dataclasses import dataclass
from datetime import datetime

# Use local modules from final_dspy (commented out to avoid DuckDB dependency)
from router import Router
from semantic_index import SemanticIndex
from planner import Planner, rewrite_plan_for_count
from executor import Executor

CARDS_PATH = os.path.join('data', 'cards.json')
MANIFEST_PATH = os.path.join('data', 'manifest.json')

# Suppress Pydantic warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# Load environment variables
load_dotenv()

# --- GLOBAL DSPy LLM CONFIGURATION (for async Discord bots) ---
def configure_dspy():
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if openai_key:
        try:
            print("🟢 Using OpenAI GPT-4 as primary LLM")
            lm = dspy.LM("openai/gpt-4", api_key=openai_key)
            dspy.configure(lm=lm)
            return "openai"
        except Exception as e:
            print(f"❌ OpenAI configuration failed: {e}")

    if anthropic_key:
        try:
            print("🔵 Using Anthropic Claude Sonnet as primary LLM")
            lm = dspy.LM("anthropic/claude-sonnet-4-0", api_key=anthropic_key)
            dspy.configure(lm=lm)
            return "anthropic"
        except Exception as e:
            print(f"⚠️ Anthropic configuration failed: {e}")
            
    raise Exception("No API keys found. Please set either ANTHROPIC_API_KEY or OPENAI_API_KEY")

try:
    provider = configure_dspy()
    print(f"✅ DSPy configured successfully with {provider.upper()}")
except Exception as e:
    print(f"❌ DSPy configuration failed: {e}")
    print("💡 Please check your API keys and try again")
    exit(1)

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
        return self.predictor(stock_data=stock_data).analysis_summary

class DecisionModule(dspy.Module):
    """Module 2: Use analysis to decide recommendation"""
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict("analysis_summary -> recommendation, reasoning")
    def forward(self, analysis_summary: str) -> Dict[str, str]:
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
        return self.predictor(recommendation=recommendation, reasoning=reasoning).final_explanation

class StockRecommenderAgent(dspy.Module):
    """Single-Agent Stock Recommendation System using DSPy"""
    def __init__(self):
        super().__init__()
        self.analyze_module = AnalyzeStockModule()
        self.decision_module = DecisionModule()
        self.explain_module = ExplainModule()
    def forward(self, stock_data: str) -> Dict[str, Any]:
        analysis_summary = self.analyze_module(stock_data)
        decision_result = self.decision_module(analysis_summary)
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
    try:
        stock = yf.Ticker(ticker.upper())
        hist_data = stock.history(period="30d")
        if hist_data.empty:
            raise Exception("No historical data available")
        current_price = hist_data['Close'].iloc[-1]
        previous_close = hist_data['Close'].iloc[-2]
        change_pct = ((current_price - previous_close) / previous_close) * 100
        change_str = f"{change_pct:+.2f}%"
        volume = hist_data['Volume'].iloc[-1]
        volume_str = f"{volume:,}"
        def calculate_rsi(prices, period=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1]
        rsi = calculate_rsi(hist_data['Close'])
        sma_5 = hist_data['Close'].rolling(window=5).mean().iloc[-1]
        sma_20 = hist_data['Close'].rolling(window=20).mean().iloc[-1]
        stock_data = f"price={current_price:.2f}, change={change_str}, volume={volume_str}, volume_raw={volume}, RSI={rsi:.1f}, 5dma={sma_5:.2f}, 20dma={sma_20:.2f}"
        return stock_data
    except Exception as e:
        raise Exception(f"Error fetching stock data for {ticker}: {str(e)}")

# --- Excel/DB Query Logic (from agent.py) ---
def load_manifest():
    with open(MANIFEST_PATH, 'r') as f:
        manifest = json.load(f)
    tableid_to_parquet = {entry['table_id']: entry['parquet_path'] for entry in manifest}
    tableid_to_info = {entry['table_id']: entry for entry in manifest}
    return tableid_to_parquet, tableid_to_info

# Helper to map c_... columns to real columns for a list of table_ids
def get_column_map(cards, table_ids):
    if isinstance(table_ids, str):
        table_ids = [table_ids]
    col_map = {}
    for card in cards:
        if card['type'] == 'ColumnCard' and card['ids']['table_id'] in table_ids:
            c_id = card['ids']['column_id']
            real_col = c_id[2:] if c_id.startswith('c_') else c_id
            col_map[c_id] = real_col
    return col_map

# Recursively map all card column IDs in the plan to real column names
def map_plan_columns(plan, col_map):
    if isinstance(plan, dict):
        keys_to_update = []
        for k, v in plan.items():
            # Map keys if they match a card column ID
            new_k = col_map.get(k, k)
            if new_k != k:
                keys_to_update.append((k, new_k))
            # Map values as before, but also replace c_... in SQL expressions
            if isinstance(v, str):
                # Replace all c_<col> with <col> in the string
                new_v = v
                for c_id, real_col in col_map.items():
                    new_v = new_v.replace(c_id, real_col)
                plan[k] = new_v
            else:
                map_plan_columns(v, col_map)
        # Actually update the keys after iteration to avoid dict size change during iteration
        for old_k, new_k in keys_to_update:
            plan[new_k] = plan.pop(old_k)
    elif isinstance(plan, list):
        for i, v in enumerate(plan):
            if isinstance(v, str):
                new_v = v
                for c_id, real_col in col_map.items():
                    new_v = new_v.replace(c_id, real_col)
                plan[i] = new_v
            else:
                map_plan_columns(v, col_map)
    return plan

def validate_plan_columns(plan, valid_columns):
    errors = []
    def is_sql_expr(val):
        # Allow expressions like COUNT(*) AS count, SUM(...), etc.
        return isinstance(val, str) and (('(' in val and 'AS' in val) or val.strip().upper().startswith('COUNT(') or val.strip().upper().startswith('SUM(') or val.strip().upper().startswith('AVG(') or val.strip().upper().startswith('MIN(') or val.strip().upper().startswith('MAX('))
    def check_col(val):
        if isinstance(val, str) and not is_sql_expr(val) and val not in valid_columns:
            errors.append(val)
    def walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == 'columns' and isinstance(v, list):
                    for col in v:
                        check_col(col)
                elif k == 'by':
                    check_col(v)
                else:
                    walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)
    walk(plan)
    return errors

def synthesize_with_llm(question, table_str, provenance, explanations):
    prompt = f"""
You are a data analyst assistant. Given the following:

User Question: {question}

Table:
{table_str}

Provenance:
{json.dumps(provenance, indent=2)}

Column Explanations:
{explanations}

Provide a concise, accurate answer to the user's question, using only the data and explanations above. Do not fabricate numbers. If the answer is not in the data, say so.
"""
    try:
        import dspy
        response = dspy.Predict(prompt)
        return response.completion if hasattr(response, 'completion') else str(response)
    except Exception:
        pass
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        message = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=256,
            temperature=0.2,
            system="You are a data analyst assistant.",
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text if hasattr(message, 'content') else str(message)
    except Exception:
        pass
    return "[LLM synthesis unavailable: DSPy Anthropic and anthropic SDK not available or not configured]"

# Initialize the agent
agent = StockRecommenderAgent()

# Discord Bot Configuration
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents, help_command=None)

@bot.event
async def on_ready():
    print(f'🤖 {bot.user} has connected to Discord!')
    print(f'📊 DSPy Stock Bot (Slash Commands) is ready!')
    print(f'🤖 Provider: {provider.upper()}')
    print(f'💬 Use /analyze <ticker> to get stock recommendations')
    print(f'💬 Use /query <question> to query Excel/DB')
    try:
        print("🔄 Registering slash commands...")
        await bot.tree.sync()
        print("✅ Slash commands registered successfully!")
    except Exception as e:
        print(f"❌ Failed to register slash commands: {e}")

# --- Stock Analysis Command (original) ---
@bot.tree.command(name="analyze", description="Analyze any stock using DSPy framework")
async def analyze_stock_slash(interaction: discord.Interaction, ticker: str):
    await interaction.response.defer()
    try:
        embed = discord.Embed(
            title="📊 Stock Analysis in Progress",
            description=f"Analyzing **{ticker.upper()}**...",
            color=0x00ff00
        )
        embed.add_field(name="Status", value="🔄 Fetching data and analyzing...", inline=False)
        message = await interaction.followup.send(embed=embed)
        stock_data = get_stock_data(ticker)
        embed.description = f"Analyzing **{ticker.upper()}**\n\n**Stock Data:**\n`{stock_data}`"
        embed.set_field_at(0, name="Status", value="🤖 Generating AI recommendation...", inline=False)
        result = agent(stock_data)
        embed = discord.Embed(
            title=f"📊 {ticker.upper()} Stock Analysis - DSPy",
            description="Analysis completed using DSPy modular AI framework",
            color=0x00ff00
        )
        try:
            price_str = stock_data.split('price=')[1].split(',')[0]
            change_str = stock_data.split('change=')[1].split(',')[0]
            volume_str = stock_data.split('volume_raw=')[1].split(',')[0]
            embed.add_field(name="💰 Current Price", value=f"${float(price_str):.2f}", inline=True)
            embed.add_field(name="📈 Change", value=change_str, inline=True)
            embed.add_field(name="📊 Volume", value=volume_str, inline=True)
        except:
            embed.add_field(name="💰 Current Price", value="N/A", inline=True)
            embed.add_field(name="📈 Change", value="N/A", inline=True)
            embed.add_field(name="📊 Volume", value="N/A", inline=True)
        recommendation = result['recommendation'][:1024] if len(result['recommendation']) > 1024 else result['recommendation']
        reasoning = result['reasoning'][:1024] if len(result['reasoning']) > 1024 else result['reasoning']
        explanation = result['explanation'][:1024] if len(result['explanation']) > 1024 else result['explanation']
        rec_lower = result['recommendation'].lower()
        if "buy" in rec_lower:
            recommendation_type = "BUY"
            confidence = "High" if "strong" in rec_lower or "recommend" in rec_lower else "Medium"
        elif "sell" in rec_lower:
            recommendation_type = "SELL"
            confidence = "High" if "strong" in rec_lower else "Medium"
        else:
            recommendation_type = "HOLD"
            confidence = "Medium"
        embed.add_field(name="🎯 Recommendation", value=recommendation_type, inline=True)
        embed.add_field(name="📊 Confidence", value=confidence, inline=True)
        embed.add_field(name="⚠️ Risk Level", value="Medium", inline=True)
        if recommendation:
            embed.add_field(name="📋 Recommendation Details", value=recommendation, inline=False)
        if reasoning:
            embed.add_field(name="💭 Reasoning", value=reasoning, inline=False)
        if explanation:
            embed.add_field(name="📖 Explanation", value=explanation, inline=False)
        embed.add_field(name="🔄 Workflow Status", value="Completed", inline=True)
        embed.add_field(name="⚡ Framework", value="DSPy AI", inline=True)
        embed.add_field(name="⏰ Completed", value=datetime.now().strftime("%H:%M:%S"), inline=True)
        embed.set_footer(text=f"pookan-dspy • {provider.upper()} • Real-time market data")
        await message.edit(embed=embed)
    except Exception as e:
        error_embed = discord.Embed(
            title="❌ Error",
            description=f"Failed to analyze {ticker.upper()}: {str(e)}",
            color=0xff0000
        )
        await interaction.followup.send(embed=error_embed)

# --- Excel/DB Query Command ---
@bot.tree.command(name="query", description="Query the Excel/DB using natural language (LLM-powered)")
async def query_db_slash(interaction: discord.Interaction, question: str):
    await interaction.response.defer()
    try:
        # 1. Route
        router = Router()
        mode = router.classify(question)
        # 2. Retrieve cards
        with open(CARDS_PATH, 'r') as f:
            cards = json.load(f)
        index = SemanticIndex(cards)
        relevant_cards = index.retrieve(question, k=8)
        # 3. Plan
        planner = Planner()
        plan = planner.build(question, relevant_cards, mode)
        if isinstance(plan, list):
            plan = plan[0]
        # --- Robust plan structure check with subquery/CTE detection ---
        def find_table_id(plan):
            # Try to find a dict with 'table_id' in targets
            if isinstance(plan, dict) and 'targets' in plan and isinstance(plan['targets'], list):
                for t in plan['targets']:
                    if isinstance(t, dict) and 'table_id' in t:
                        return t['table_id'], t
            # Search nested dicts
            def search_dict(d):
                if isinstance(d, dict):
                    # Detect subquery/CTE
                    if any(k in d for k in ['subquery', 'cte', 'with']):
                        print("[QUERY HANDLER][ERROR] Subquery/CTE detected in plan:")
                        print(json.dumps(plan, indent=2))
                        return 'SUBQUERY_OR_CTE', d
                    if 'table_id' in d:
                        return d['table_id'], d
                    for v in d.values():
                        result = search_dict(v)
                        if result:
                            return result
                elif isinstance(d, list):
                    for v in d:
                        result = search_dict(v)
                        if result:
                            return result
                return None
            return search_dict(plan)
        found = find_table_id(plan)
        if not found or not found[0]:
            print("[QUERY HANDLER][ERROR] No table_id found in LLM plan:")
            print(json.dumps(plan, indent=2))
            error_embed = discord.Embed(
                title="❌ Query Error",
                description="Sorry, I couldn't find a valid table to query for this question. Please try rephrasing or ask a simpler question. (No table_id found in LLM plan)",
                color=0xff0000
            )
            await interaction.followup.send(embed=error_embed)
            return
        table_id, target_dict = found
        if table_id == 'SUBQUERY_OR_CTE':
            error_embed = discord.Embed(
                title="❌ Query Error",
                description="Sorry, this question requires a subquery or CTE, which is not yet supported. Please try a simpler question or contact the developer.",
                color=0xff0000
            )
            await interaction.followup.send(embed=error_embed)
            return
        # Use the found target_dict as the main target
        plan['targets'] = [target_dict]
        # 4. Map table_id to Parquet path
        tableid_to_parquet = {}
        tableid_to_info = {}
        with open(MANIFEST_PATH, 'r') as f:
            manifest = json.load(f)
            for entry in manifest:
                tableid_to_parquet[entry['table_id']] = entry['parquet_path']
                tableid_to_info[entry['table_id']] = entry
        parquet_path = tableid_to_parquet.get(table_id)
        info = tableid_to_info.get(table_id)
        if not parquet_path:
            raise Exception(f"table_id {table_id} not found in manifest.")
        # 5. Collect all table_ids (main + JOINs)
        table_ids = [table_id]
        for op in plan.get('ops', []):
            if op.get('type') == 'JOIN':
                join_table = op['args']['table_id']
                table_ids.append(join_table)
        # 6. Map c_... columns to real columns (RECURSIVELY, for all tables)
        col_map = get_column_map(cards, table_ids)
        plan = map_plan_columns(plan, col_map)
        # 7. Validate columns
        import pandas as pd
        df_check = pd.read_parquet(parquet_path)
        valid_columns = set(df_check.columns)
        # If JOIN, add columns from joined table
        for op in plan.get('ops', []):
            if op.get('type') == 'JOIN':
                import glob
                join_table = op['args']['table_id']
                candidates = glob.glob(f"data/{join_table}.parquet") + glob.glob(f"final_dspy/data/{join_table}.parquet")
                if candidates:
                    join_df = pd.read_parquet(candidates[0])
                    valid_columns.update(join_df.columns)
        errors = validate_plan_columns(plan, valid_columns)
        # --- Fallback: If LLM invented 'count', auto-rewrite plan for aggregation ---
        if errors and 'count' in errors:
            plan = rewrite_plan_for_count(plan, valid_columns)
            errors = validate_plan_columns(plan, valid_columns | {'count'})
        if errors:
            raise Exception(f"The following columns do not exist in table '{table_id}': {', '.join(errors)}. Please rephrase your question or try a different one.")
        # 8. Execute
        executor = Executor(parquet_path, table_id)
        df, provenance = executor.run(plan)
        # 9. Prepare answer
        explanations = "\n".join([
            f"{card['ids']['column_id']}: {card['text']}" for card in relevant_cards if card['type'] == 'ColumnCard' and any(col in card['ids']['column_id'] for col in plan['targets'][0]['columns'])
        ])
        llm_answer = synthesize_with_llm(question, df.to_string(index=False) if df is not None else '', provenance, explanations)
        # 10. Build Discord embed
        embed = discord.Embed(
            title=f"📊 Query Result",
            description=f"**Question:** {question}\n\n**LLM Answer:**\n{llm_answer}",
            color=0x0099ff
        )
        if df is not None and not df.empty:
            try:
                table_str = df.head(10).to_markdown(index=False)
            except Exception:
                table_str = df.head(10).to_string(index=False)
            embed.add_field(name="Table (top 10 rows)", value=f"```\n{table_str}\n```", inline=False)
        embed.set_footer(text=f"DSPy Query • {provider.upper()} • {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        await interaction.followup.send(embed=embed)
    except Exception as e:
        error_embed = discord.Embed(
            title="❌ Query Error",
            description=f"Failed to answer query: {str(e)}",
            color=0xff0000
        )
        await interaction.followup.send(embed=error_embed)

# --- Smart Query Engine Class ---
class SmartQueryEngine:
    """Enhanced LlamaIndex query engine with better handling of complex queries"""
    
    def __init__(self):
        self.engine = None
        self.query_engine = None
        self.setup_engine()
    
    def setup_engine(self):
        """Setup LlamaIndex query engine"""
        try:
            from llama_index.core import SQLDatabase
            from llama_index.core.query_engine import NLSQLTableQueryEngine
            from sqlalchemy import create_engine
            import glob
            import pandas as pd
            import os
            
            # Setup SQLite database
            self.engine = create_engine("sqlite:///:memory:")
            
            # Load parquet files
            script_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(script_dir, "data")
            parquet_files = glob.glob(os.path.join(data_dir, "*.parquet"))
            
            for pf in parquet_files:
                table_name = os.path.splitext(os.path.basename(pf))[0]
                df = pd.read_parquet(pf)
                df.to_sql(table_name, self.engine, if_exists='replace', index=False)
            
            # Create LlamaIndex query engine
            sql_db = SQLDatabase(self.engine)
            self.query_engine = NLSQLTableQueryEngine(sql_database=sql_db)
            
        except Exception as e:
            print(f"❌ Smart Query Engine setup failed: {e}")
            self.query_engine = None
    
    def break_down_complex_query(self, question):
        """Break down complex queries into simpler parts"""
        complex_indicators = [
            "and what", "and why", "and how", "and when", "and where",
            "highest balance", "most recent", "recommend", "compare",
            "for each", "total value", "all filled", "group by", "sum of"
        ]
        
        question_lower = question.lower()
        
        # Check if it's a complex query
        is_complex = any(indicator in question_lower for indicator in complex_indicators)
        
        if not is_complex:
            return [question]
        
        # Break down complex queries
        if "and what" in question_lower:
            parts = question.split(" and what")
            return [parts[0], "What is " + parts[1].lstrip()]
        
        elif "and why" in question_lower:
            parts = question.split(" and why")
            return [parts[0], "Why " + parts[1].lstrip()]
        
        elif "highest balance" in question_lower and "most recent" in question_lower:
            return [
                "Which account has the highest balance?",
                "What is the most recent order?"
            ]
        
        elif "recommend" in question_lower:
            return [
                "What are the available securities?",
                "What are their risk levels and fees?"
            ]
        
        # Handle "for each" analytical queries
        elif "for each" in question_lower:
            if "total value" in question_lower and "filled orders" in question_lower:
                return [
                    "What are all the accounts?",
                    "What are all the filled orders?",
                    "What is the total value of orders for each account?"
                ]
            elif "account" in question_lower:
                return [
                    "What are all the accounts?",
                    "What information do you want for each account?"
                ]
        
        # Handle aggregation queries
        elif "total value" in question_lower:
            return [
                "What orders are there?",
                "What is the total value of all orders?"
            ]
        
        # Handle "all filled" queries
        elif "all filled" in question_lower:
            return [
                "What are the filled orders?",
                "What information do you want about filled orders?"
            ]
        
        return [question]
    
    def query(self, question):
        """Smart query with fallback strategies"""
        if not self.query_engine:
            return "❌ Query engine not available"
        
        try:
            # Check if it's a complex query that should be broken down proactively
            question_lower = question.lower()
            complex_patterns = [
                "for each", "total value", "all filled", "group by", "sum of",
                "and what", "and why", "highest balance", "most recent"
            ]
            
            should_break_down = any(pattern in question_lower for pattern in complex_patterns)
            
            if should_break_down:
                print("🔄 Detected complex query, breaking down proactively...")
                sub_questions = self.break_down_complex_query(question)
                
                if len(sub_questions) > 1:
                    answers = []
                    
                    for sub_q in sub_questions:
                        try:
                            sub_response = self.query_engine.query(sub_q)
                            answers.append(str(sub_response))
                        except Exception as e:
                            answers.append(f"Error: {e}")
                    
                    # Combine answers
                    combined_answer = f"Based on the breakdown:\n"
                    for i, (sub_q, ans) in enumerate(zip(sub_questions, answers), 1):
                        combined_answer += f"{i}. {sub_q}: {ans}\n"
                    
                    return combined_answer
            
            # Try direct query first
            response = self.query_engine.query(question)
            answer = str(response)
            
            # Check if the answer seems incomplete or has errors
            if any(error_indicator in answer.lower() for error_indicator in [
                "invalid", "error", "not found", "does not exist", "column", "table", "does not exist in the table", "column.*does not exist", "references a column", "does not exist in the table"
            ]):
                # Break down complex query
                sub_questions = self.break_down_complex_query(question)
                
                if len(sub_questions) > 1:
                    answers = []
                    
                    for sub_q in sub_questions:
                        try:
                            sub_response = self.query_engine.query(sub_q)
                            answers.append(str(sub_response))
                        except Exception as e:
                            answers.append(f"Error: {e}")
                    
                    # Combine answers
                    combined_answer = f"Based on the breakdown:\n"
                    for i, (sub_q, ans) in enumerate(zip(sub_questions, answers), 1):
                        combined_answer += f"{i}. {sub_q}: {ans}\n"
                    
                    return combined_answer
            
            return answer
            
        except Exception as e:
            # Try a simpler approach
            try:
                simplified_question = self.simplify_question(question)
                response = self.query_engine.query(simplified_question)
                return f"Simplified answer: {response}"
            except Exception as e2:
                return f"❌ Query failed: {e2}"
    
    def simplify_question(self, question):
        """Simplify complex questions"""
        question_lower = question.lower()
        
        if "and what" in question_lower:
            return question.split(" and what")[0] + "?"
        
        if "and why" in question_lower:
            return question.split(" and why")[0] + "?"
        
        if "highest balance" in question_lower and "most recent" in question_lower:
            return "Which account has the highest balance?"
        
        return question

# Initialize Smart Query Engine (now with metadata enhancement)
smart_engine = SmartQueryEngine()



# --- LlamaIndex Query Engine Integration ---
@bot.tree.command(name="llama_query", description="Query your data using LlamaIndex's Query Engine (enhanced with Smart Query Engine)")
async def llama_query_slash(interaction: discord.Interaction, question: str):
    await interaction.response.defer()
    try:
        # Use the Smart Query Engine for enhanced LlamaIndex functionality
        response = smart_engine.query(question)
        
        embed = discord.Embed(
            title="🦙 LlamaIndex Query Result",
            description=f"**Question:** {question}\n\n**Answer:**\n{response}",
            color=0x4e8cff
        )
        await interaction.followup.send(embed=embed)
        
    except Exception as e:
        tb_str = traceback.format_exc()
        print("[LLAMA_QUERY][ERROR]", tb_str)
        error_embed = discord.Embed(
            title="❌ LlamaIndex Query Error",
            description=f"Failed to answer query: {str(e)}\n\n```{tb_str[-1500:]}```",
            color=0xff0000
        )
        await interaction.followup.send(embed=error_embed)

# --- Debug LlamaIndex Configuration ---
@bot.tree.command(name="debug_llama", description="Debug LlamaIndex configuration and API keys")
async def debug_llama_slash(interaction: discord.Interaction):
    await interaction.response.defer()
    try:
        debug_info = []
        
        # Check environment variables
        debug_info.append("🔍 **Environment Variables:**")
        llama_parse_key = os.getenv('LLAMAPARSE_API_KEY')
        anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        openai_key = os.getenv('OPENAI_API_KEY')
        
        debug_info.append(f"- LLAMAPARSE_API_KEY: {'✅ Set' if llama_parse_key else '❌ Not set'}")
        debug_info.append(f"- ANTHROPIC_API_KEY: {'✅ Set' if anthropic_key else '❌ Not set'}")
        debug_info.append(f"- OPENAI_API_KEY: {'✅ Set' if openai_key else '❌ Not set'}")
        
        # Check LlamaIndex installation
        debug_info.append("\n🔍 **LlamaIndex Installation:**")
        try:
            import llama_index
            debug_info.append(f"- LlamaIndex version: {llama_index.__version__}")
        except ImportError:
            debug_info.append("- ❌ LlamaIndex not installed")
        except Exception as e:
            debug_info.append(f"- ⚠️ LlamaIndex import error: {e}")
        
        # Check data directory
        debug_info.append("\n🔍 **Data Directory:**")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, "data")
        debug_info.append(f"- Data directory: {data_dir}")
        debug_info.append(f"- Directory exists: {'✅ Yes' if os.path.exists(data_dir) else '❌ No'}")
        
        if os.path.exists(data_dir):
            files = os.listdir(data_dir)
            parquet_files = [f for f in files if f.endswith('.parquet')]
            debug_info.append(f"- Total files: {len(files)}")
            debug_info.append(f"- Parquet files: {len(parquet_files)}")
            debug_info.append(f"- Parquet files: {parquet_files}")
        
        # Test database connection
        debug_info.append("\n🔍 **Database Test:**")
        try:
            import duckdb
            duckdb_con = duckdb.connect()
            debug_info.append("- ✅ DuckDB connection successful")
            duckdb_con.close()
        except Exception as e:
            debug_info.append(f"- ❌ DuckDB connection failed: {e}")
        
        embed = discord.Embed(
            title="🔧 LlamaIndex Debug Information",
            description="\n".join(debug_info),
            color=0x00ff00
        )
        await interaction.followup.send(embed=embed)
        
    except Exception as e:
        error_embed = discord.Embed(
            title="❌ Debug Error",
            description=f"Failed to get debug info: {str(e)}",
            color=0xff0000
        )
        await interaction.followup.send(embed=embed)

# --- Other commands (help, ping, welcome, status) ---
# (Copy unchanged from dspy_bot.py, omitted here for brevity, but will be included in the actual file)

# ... (rest of dspy_bot.py commands: help, ping, welcome, status, main) ...
@bot.tree.command(name="help", description="Show comprehensive help information")
async def help_command_slash(interaction: discord.Interaction):
    """Slash command handler for help information"""
    embed = discord.Embed(
        title="🤖 DSPy Stock Analysis Bot - Help Guide",
        description="**AI-powered stock analysis using DSPy framework and real-time market data**\n\nThis bot provides instant stock analysis with AI recommendations using the latest market data.",
        color=0x0099ff
    )
    # Basic commands
    embed.add_field(
        name="📊 **Basic Commands**",
        value="""
`/analyze <ticker>` - Analyze any stock (e.g., `/analyze MSFT`)
`/ping` - Test bot connectivity
`/status` - Show bot status and configuration
`/help` - Show this help message
        """,
        inline=False
    )
    # Examples
    embed.add_field(
        name="💡 **Usage Examples**",
        value="""
• `/analyze AAPL` - Analyze Apple stock
• `/analyze TSLA` - Analyze Tesla stock  
• `/analyze GOOGL` - Analyze Google stock
• `/analyze NVDA` - Analyze NVIDIA stock
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
    await interaction.response.send_message(embed=embed)

@bot.tree.command(name="ping", description="Test bot connectivity")
async def ping_slash(interaction: discord.Interaction):
    """Slash command handler for ping"""
    embed = discord.Embed(
        title="🏓 Pong!",
        description=f"Bot latency: {round(bot.latency * 1000)}ms",
        color=0x00ff00
    )
    embed.add_field(name="🤖 Bot", value="DSPy Stock Analysis Bot", inline=True)
    embed.add_field(name="⚡ Framework", value="DSPy AI", inline=True)
    embed.add_field(name="🤖 Provider", value=provider.upper(), inline=True)
    embed.add_field(name="💡 Quick Start", value="Try `/analyze MSFT` to test!", inline=True)
    embed.set_footer(text=f"Real-time data from Yahoo Finance • Powered by {provider.upper()}")
    await interaction.response.send_message(embed=embed)

@bot.tree.command(name="welcome", description="Show welcome message and quick start guide")
async def welcome_slash(interaction: discord.Interaction):
    """Slash command handler for welcome"""
    embed = discord.Embed(
        title="🤖 Welcome to DSPy Stock Analysis Bot!",
        description="**AI-powered stock analysis using DSPy framework**\n\nGet instant stock analysis with real-time market data and AI recommendations.",
        color=0x0099ff
    )
    embed.add_field(
        name="🚀 **Quick Start**",
        value="""
Try these commands to get started:
• `/analyze MSFT` - Analyze Microsoft stock
• `/analyze AAPL` - Analyze Apple stock
• `/help` - See all commands and features
• `/status` - Check bot configuration
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
    await interaction.response.send_message(embed=embed)

@bot.tree.command(name="status", description="Show bot status and configuration")
async def status_command_slash(interaction: discord.Interaction):
    """Slash command handler for status"""
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
        value="/analyze, /help, /ping, /welcome, /status",
        inline=True
    )
    embed.set_footer(text=f"DSPy AI • {provider.upper()} • {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    await interaction.response.send_message(embed=embed)

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

    print("🚀 Starting DSPy Discord Stock Recommendation Bot...")
    print("📊 Bot will be ready to analyze stocks!")
    print(f"✅ Environment variables loaded successfully")
    print(f"🤖 Provider: {provider.upper()}")
    print(f"💡 Use /analyze <ticker> to analyze a stock")
    print(f"💡 Use /help to see comprehensive help")
    print(f"💡 Use /ping to test connectivity")
    print(f"💡 Use /status to see bot configuration")

    try:
        bot.run(token)
    except Exception as e:
        print(f"❌ Failed to start bot: {e}")

if __name__ == "__main__":
    main()