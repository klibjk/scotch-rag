#!/usr/bin/env python3
"""
Smart Query Engine: Enhanced LlamaIndex with better error handling for complex queries
"""

import os
import sys
import glob
import json
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SmartQueryEngine:
    """Enhanced LlamaIndex query engine with better handling of complex queries"""
    
    def __init__(self):
        self.engine = None
        self.query_engine = None
        self.cards_data = None
        self.manifest_data = None
        self.setup_engine()
        self.load_metadata()
    
    def load_metadata(self):
        """Load metadata from cards.json and manifest.json for enhanced context"""
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(script_dir, "data")
            
            # Load cards.json
            cards_path = os.path.join(data_dir, "cards.json")
            if os.path.exists(cards_path):
                with open(cards_path, 'r') as f:
                    self.cards_data = json.load(f)
                print(f"✅ Loaded {len(self.cards_data)} metadata cards")
            
            # Load manifest.json
            manifest_path = os.path.join(data_dir, "manifest.json")
            if os.path.exists(manifest_path):
                with open(manifest_path, 'r') as f:
                    self.manifest_data = json.load(f)
                print(f"✅ Loaded manifest with {len(self.manifest_data)} table mappings")
                
        except Exception as e:
            print(f"⚠️ Metadata loading failed: {e}")
            self.cards_data = None
            self.manifest_data = None
    
    def get_enhanced_schema_context(self):
        """Generate enhanced schema context using metadata"""
        if not self.cards_data:
            return ""
        
        context = "Enhanced Database Schema:\n\n"
        
        # Group cards by table
        table_cards = {}
        for card in self.cards_data:
            if card.get('type') == 'TableCard':
                table_id = card['ids']['table_id']
                table_cards[table_id] = {
                    'table_info': card,
                    'columns': []
                }
            elif card.get('type') == 'ColumnCard':
                table_id = card['ids']['table_id']
                if table_id not in table_cards:
                    table_cards[table_id] = {'columns': []}
                table_cards[table_id]['columns'].append(card)
        
        # Build enhanced context
        for table_id, info in table_cards.items():
            context += f"Table: {table_id}\n"
            if 'table_info' in info:
                context += f"Description: {info['table_info']['text']}\n"
            
            context += "Columns:\n"
            for col_card in info['columns']:
                col_id = col_card['ids']['column_id']
                context += f"  - {col_id}: {col_card['text']}\n"
                if 'facets' in col_card and 'examples' in col_card['facets']:
                    examples = col_card['facets']['examples']
                    context += f"    Examples: {', '.join(examples[:3])}\n"
            context += "\n"
        
        return context
    
    def debug_metadata_usage(self, question):
        """Debug method to show how metadata could enhance queries"""
        if not self.cards_data:
            return "No metadata available"
        
        print("\n🔍 Metadata Analysis for Query:")
        print(f"Question: {question}")
        
        # Find relevant columns mentioned in the question
        question_lower = question.lower()
        relevant_cards = []
        
        for card in self.cards_data:
            if card.get('type') == 'ColumnCard':
                col_id = card['ids']['column_id']
                if any(word in question_lower for word in col_id.lower().split('_')):
                    relevant_cards.append(card)
        
        if relevant_cards:
            print(f"\n📋 Found {len(relevant_cards)} relevant columns:")
            for card in relevant_cards[:3]:  # Show first 3
                col_id = card['ids']['column_id']
                description = card['text']
                print(f"  - {col_id}: {description}")
        
        return f"Metadata could enhance understanding of {len(relevant_cards)} columns"
    
    def create_enhanced_prompt(self, question):
        """Create enhanced prompt using metadata from cards.json"""
        if not self.cards_data:
            return question
        
        # Get column mapping (metadata uses c_ prefix, actual columns don't)
        col_mapping = {}
        for card in self.cards_data:
            if card.get('type') == 'ColumnCard':
                metadata_col = card['ids']['column_id']  # e.g., c_order_id
                actual_col = metadata_col.replace('c_', '')  # e.g., order_id
                col_mapping[metadata_col] = actual_col
                col_mapping[actual_col] = metadata_col
        
        # Find relevant metadata for the question
        question_lower = question.lower()
        relevant_info = []
        
        for card in self.cards_data:
            if card.get('type') == 'ColumnCard':
                metadata_col = card['ids']['column_id']
                actual_col = col_mapping.get(metadata_col, metadata_col)
                
                # Check if column is relevant to question
                if (any(word in question_lower for word in actual_col.lower().split('_')) or
                    any(word in question_lower for word in metadata_col.lower().split('_'))):
                    relevant_info.append({
                        'column': actual_col,
                        'description': card['text'],
                        'examples': card.get('facets', {}).get('examples', [])
                    })
        
        if not relevant_info:
            return question
        
        # Build enhanced prompt
        context = "Database Schema Context:\n"
        for info in relevant_info[:5]:  # Limit to 5 most relevant
            context += f"- {info['column']}: {info['description']}\n"
            if info['examples']:
                context += f"  Examples: {', '.join(info['examples'][:3])}\n"
        
        enhanced_prompt = f"""
{context}

Question: {question}

Please use the schema context above to understand the database structure and provide an accurate answer.
"""
        return enhanced_prompt
    
    def get_metadata_summary(self, question):
        """Get summary of relevant metadata for a question"""
        if not self.cards_data:
            return "No metadata available"
        
        # Get column mapping
        col_mapping = {}
        for card in self.cards_data:
            if card.get('type') == 'ColumnCard':
                metadata_col = card['ids']['column_id']  # e.g., c_order_id
                actual_col = metadata_col.replace('c_', '')  # e.g., order_id
                col_mapping[metadata_col] = actual_col
                col_mapping[actual_col] = metadata_col
        
        question_lower = question.lower()
        relevant_cards = []
        
        for card in self.cards_data:
            if card.get('type') == 'ColumnCard':
                metadata_col = card['ids']['column_id']
                actual_col = col_mapping.get(metadata_col, metadata_col)
                
                if any(word in question_lower for word in actual_col.lower().split('_')):
                    relevant_cards.append({
                        'column': actual_col,
                        'description': card['text'],
                        'examples': card.get('facets', {}).get('examples', [])
                    })
        
        if relevant_cards:
            summary = f"Found {len(relevant_cards)} relevant columns:\n"
            for card in relevant_cards[:3]:  # Show first 3
                summary += f"- {card['column']}: {card['description'][:100]}...\n"
            return summary
        else:
            return "No specific metadata found for this question"

    def setup_engine(self):
        """Setup LlamaIndex query engine"""
        print("🔧 Setting up Smart Query Engine...")
        
        try:
            from llama_index.core import SQLDatabase
            from llama_index.core.query_engine import NLSQLTableQueryEngine
            
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
                print(f"✅ Loaded {table_name}")
            
            # Create LlamaIndex query engine
            sql_db = SQLDatabase(self.engine)
            self.query_engine = NLSQLTableQueryEngine(sql_database=sql_db)
            print("✅ Smart Query Engine ready")
            
        except Exception as e:
            print(f"❌ Setup failed: {e}")
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
    
    def analyze_query_complexity_with_llm(self, question):
        """Use LLM to analyze query complexity and suggest breakdown strategies"""
        try:
            # Create a prompt for the LLM to analyze the query
            analysis_prompt = f"""
Analyze this database query question and determine if it needs to be broken down into simpler parts.

Question: "{question}"

Available tables and their key columns:
- accounts_orders_orders: order_id, account_id, security, order_type, quantity, price, status, placed_time, executed_time
- accounts_orders_accounts: account_id, owner, account_type, balance
- securities_info_securities: security_id, name, type, current_price, risk_level
- securities_info_fees: security_id, fee_type, amount
- securities_info_holdings: etf_id, holding_security, weight_percent

Instructions:
1. If the query is simple (single table, basic filtering), respond with: "SIMPLE"
2. If the query is complex (multiple tables, joins, aggregations, multiple conditions), respond with: "COMPLEX"
3. If complex, suggest 2-3 specific, executable questions that use only the available columns.

Examples:
- "What is the status of order O1001?" → SIMPLE
- "Which account has the highest balance, and what is their most recent order?" → COMPLEX
  - "Which account has the highest balance?"
  - "What is the most recent order by placed_time?"
- "For each account, what is the total value of all filled orders?" → COMPLEX
  - "What are all the account_ids?"
  - "What are all the filled orders with their quantities and prices?"
  - "What is the total value (quantity * price) for each account?"

Your analysis:
"""

            # Use the query engine's LLM to analyze
            response = self.query_engine.query(analysis_prompt)
            analysis = str(response).strip()
            
            print(f"🔍 LLM Analysis: {analysis}")
            
            # Parse the response
            if "SIMPLE" in analysis.upper():
                return {"complexity": "simple", "breakdown": [question]}
            elif "COMPLEX" in analysis.upper():
                # Extract suggested questions from the response
                lines = analysis.split('\n')
                breakdown = []
                for line in lines:
                    line = line.strip()
                    if (line.startswith('-') or line.startswith('•') or 
                        line.startswith('1.') or line.startswith('2.') or line.startswith('3.')):
                        # Extract the question from the bullet point
                        question_part = line
                        # Remove bullet points and numbers
                        for prefix in ['- ', '• ', '1. ', '2. ', '3. ']:
                            if question_part.startswith(prefix):
                                question_part = question_part[len(prefix):]
                                break
                        
                        # Remove quotes if present
                        if question_part.startswith('"') and question_part.endswith('"'):
                            question_part = question_part[1:-1]
                        
                        if question_part and len(question_part) > 10:  # Ensure it's a real question
                            breakdown.append(question_part)
                
                if breakdown:
                    return {"complexity": "complex", "breakdown": breakdown}
                else:
                    # Fallback: use basic breakdown
                    return {"complexity": "complex", "breakdown": [question]}
            else:
                # If LLM analysis is unclear, treat as simple
                return {"complexity": "simple", "breakdown": [question]}
                
        except Exception as e:
            print(f"❌ LLM analysis failed: {e}")
            # Fallback to simple
            return {"complexity": "simple", "breakdown": [question]}
    
    def query(self, question):
        """Smart query with LLM-based complexity analysis"""
        if not self.query_engine:
            return "❌ Query engine not available"
        
        try:
            # Use LLM to analyze query complexity
            print("🔍 Analyzing query complexity with LLM...")
            analysis = self.analyze_query_complexity_with_llm(question)
            
            if analysis["complexity"] == "complex":
                print("🔄 LLM detected complex query, breaking down...")
                sub_questions = analysis["breakdown"]
                
                if len(sub_questions) > 1:
                    answers = []
                    
                    for i, sub_q in enumerate(sub_questions, 1):
                        print(f"  {i}. {sub_q}")
                        try:
                            # Use enhanced prompt for sub-questions too
                            enhanced_sub_prompt = self.create_enhanced_prompt(sub_q)
                            sub_response = self.query_engine.query(enhanced_sub_prompt)
                            sub_answer = str(sub_response)
                            
                            # Check if the sub-answer has errors
                            if any(error_indicator in sub_answer.lower() for error_indicator in [
                                "invalid", "error", "not found", "does not exist", "column", "table", "references a column"
                            ]):
                                # Try to simplify the sub-question
                                simplified_sub_q = self.simplify_sub_question(sub_q)
                                if simplified_sub_q != sub_q:
                                    print(f"    ⚠️ Trying simplified: {simplified_sub_q}")
                                    enhanced_simplified_prompt = self.create_enhanced_prompt(simplified_sub_q)
                                    sub_response = self.query_engine.query(enhanced_simplified_prompt)
                                    sub_answer = str(sub_response)
                            
                            answers.append(sub_answer)
                        except Exception as e:
                            answers.append(f"Error: {e}")
                    
                    # Combine answers
                    combined_answer = f"Based on LLM analysis, here's the breakdown:\n"
                    for i, (sub_q, ans) in enumerate(zip(sub_questions, answers), 1):
                        combined_answer += f"{i}. {sub_q}: {ans}\n"
                    
                    return combined_answer
            
            # Try direct query with enhanced prompt (either simple or complex that couldn't be broken down)
            print("🔄 Attempting direct query with enhanced context...")
            response = self.query_engine.query(enhanced_prompt)
            answer = str(response)
            
            # Check if the answer has errors and try breakdown as fallback
            if any(error_indicator in answer.lower() for error_indicator in [
                "invalid", "error", "not found", "does not exist", "column", "table", "references a column"
            ]):
                print("⚠️ Direct query had issues, trying LLM breakdown as fallback...")
                analysis = self.analyze_query_complexity_with_llm(question)
                sub_questions = analysis["breakdown"]
                
                if len(sub_questions) > 1:
                    answers = []
                    
                    for sub_q in sub_questions:
                        try:
                            # Use enhanced prompt for fallback sub-questions too
                            enhanced_sub_prompt = self.create_enhanced_prompt(sub_q)
                            sub_response = self.query_engine.query(enhanced_sub_prompt)
                            answers.append(str(sub_response))
                        except Exception as e:
                            answers.append(f"Error: {e}")
                    
                    # Combine answers
                    combined_answer = f"Fallback breakdown:\n"
                    for i, (sub_q, ans) in enumerate(zip(sub_questions, answers), 1):
                        combined_answer += f"{i}. {sub_q}: {ans}\n"
                    
                    return combined_answer
            
            return answer
            
        except Exception as e:
            print(f"❌ Query failed: {e}")
            
            # Try a simpler approach
            try:
                print("🔄 Trying simplified query...")
                simplified_question = self.simplify_question(question)
                enhanced_simplified_prompt = self.create_enhanced_prompt(simplified_question)
                response = self.query_engine.query(enhanced_simplified_prompt)
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
    
    def simplify_sub_question(self, question):
        """Simplify sub-questions to make them more executable"""
        question_lower = question.lower()
        
        # Simplify "most recent" queries
        if "most recent" in question_lower:
            if "order" in question_lower:
                return "What are all the orders ordered by placed_time?"
            elif "account" in question_lower:
                return "What are all the accounts?"
        
        # Simplify "total value" queries
        if "total value" in question_lower:
            if "quantity * price" in question_lower:
                return "What are all the orders with their quantities and prices?"
            else:
                return "What are all the orders?"
        
        # Simplify "filled orders" queries
        if "filled orders" in question_lower:
            return "What are all the orders with status 'filled'?"
        
        # Simplify "account_ids" queries
        if "account_ids" in question_lower or "account_id" in question_lower:
            return "What are all the accounts?"
        
        return question

def test_smart_engine():
    """Test the smart query engine"""
    print("🧪 Testing Smart Query Engine")
    print("=" * 50)
    
    engine = SmartQueryEngine()
    
    # Test questions of varying complexity
    test_questions = [
        # Simple questions
        # "What is the status of order O1001?",
        # "How many orders are there?",
        # "What is the highest balance?",
        
        # Complex questions
        # "Which account has the highest balance, and what is their most recent order?",
        # "Which security has the highest management fee, and what is its risk level?",
        # "Which ETF would you recommend for someone seeking stability, and why?",
        "Who owns the account with the highest balance, and what is their most recent order?",
        "For each account, what is the total value of all filled orders?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. Testing: {question}")
        print("-" * 40)
        
        try:
            answer = engine.query(question)
            print(f"✅ Answer: {answer}")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("=" * 50)

if __name__ == "__main__":
    test_smart_engine()
