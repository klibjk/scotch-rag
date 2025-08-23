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
    

    
    def analyze_query_complexity_with_llm(self, question):
        """Use LLM to analyze query complexity and suggest breakdown strategies in ONE API call"""
        try:
            # Single comprehensive prompt that does both analysis and breakdown
            analysis_prompt = f"""
You are a database query analyzer. Given a user question, determine if it needs to be broken down into simpler, executable sub-queries.

Question: "{question}"

Available database schema:
- accounts_orders_orders: order_id, account_id, security, order_type, quantity, price, status, placed_time, executed_time
- accounts_orders_accounts: account_id, owner, account_type, balance  
- securities_info_securities: security_id, name, type, current_price, risk_level
- securities_info_fees: security_id, fee_type, amount
- securities_info_holdings: etf_id, holding_security, weight_percent

Instructions:
1. Determine if this is a SIMPLE query (single table, basic operations) or COMPLEX query (multiple tables, joins, aggregations, multiple conditions)
2. If COMPLEX, break it down into 2-4 specific, executable sub-questions
3. Each sub-question should be self-contained and use only the available columns
4. Ensure sub-questions can be executed independently

Examples:
- "What is the status of order O1001?" → SIMPLE
- "Which account has the highest balance and what is their most recent order?" → COMPLEX
  Sub-questions:
  1. "Which account has the highest balance?"
  2. "What is the most recent order by placed_time?"
- "For each account, show the total value of filled orders and the owner name" → COMPLEX
  Sub-questions:
  1. "What are all the accounts with their owners?"
  2. "What are all the filled orders with quantities and prices?"
  3. "What is the total value (quantity * price) for each account?"

Your analysis for the given question:
"""

            # Use the query engine's LLM to analyze
            response = self.query_engine.query(analysis_prompt)
            analysis = str(response).strip()
            
            print(f"🔍 LLM Analysis: {analysis}")
            
            # Parse the response to extract complexity and sub-questions
            if "SIMPLE" in analysis.upper():
                return {"complexity": "simple", "breakdown": [question], "llm_analysis": analysis}
            elif "COMPLEX" in analysis.upper():
                # Extract sub-questions from the response
                sub_questions = []
                lines = analysis.split('\n')
                
                for line in lines:
                    line = line.strip()
                    # Look for numbered or bulleted sub-questions
                    if (line.startswith(('1.', '2.', '3.', '4.', '-', '•')) and 
                        '?' in line and len(line) > 20):
                        # Clean up the line
                        for prefix in ['1. ', '2. ', '3. ', '4. ', '- ', '• ']:
                            if line.startswith(prefix):
                                line = line[len(prefix):]
                                break
                        
                        # Remove quotes if present
                        if line.startswith('"') and line.endswith('"'):
                            line = line[1:-1]
                        
                        if line and len(line) > 10:
                            sub_questions.append(line)
                
                if sub_questions:
                    return {"complexity": "complex", "breakdown": sub_questions, "llm_analysis": analysis}
                else:
                    # Fallback: treat as simple if we can't parse sub-questions
                    return {"complexity": "simple", "breakdown": [question], "llm_analysis": analysis}
            else:
                # If analysis is unclear, treat as simple
                return {"complexity": "simple", "breakdown": [question], "llm_analysis": analysis}
                
        except Exception as e:
            print(f"❌ LLM analysis failed: {e}")
            # Fallback to simple
            return {"complexity": "simple", "breakdown": [question], "llm_analysis": f"Analysis failed: {e}"}
    
    def query(self, question):
        """Smart query with single LLM call for complexity analysis and breakdown"""
        if not self.query_engine:
            return {"answer": "❌ Query engine not available", "logs": []}
        
        logs = []
        try:
            # Single LLM call to analyze complexity and get breakdown
            log_entry = "🔍 Analyzing query complexity and breakdown with LLM..."
            print(log_entry)
            logs.append(log_entry)
            
            analysis = self.analyze_query_complexity_with_llm(question)
            
            # Add LLM analysis to logs
            if "llm_analysis" in analysis:
                log_entry = f"🤖 LLM Analysis: {analysis['llm_analysis'][:200]}..."
                print(log_entry)
                logs.append(log_entry)
            
            if analysis["complexity"] == "complex":
                log_entry = "🔄 LLM detected complex query, breaking down..."
                print(log_entry)
                logs.append(log_entry)
                
                sub_questions = analysis["breakdown"]
                
                if len(sub_questions) > 1:
                    answers = []
                    
                    for i, sub_q in enumerate(sub_questions, 1):
                        log_entry = f"  {i}. {sub_q}"
                        print(log_entry)
                        logs.append(log_entry)
                        
                        try:
                            # Use enhanced prompt for sub-questions
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
                                    log_entry = f"    ⚠️ Trying simplified: {simplified_sub_q}"
                                    print(log_entry)
                                    logs.append(log_entry)
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
                    
                    return {"answer": combined_answer, "logs": logs}
            
            # Try direct query with enhanced prompt (either simple or complex that couldn't be broken down)
            log_entry = "🔄 Attempting direct query with enhanced context..."
            print(log_entry)
            logs.append(log_entry)
            
            enhanced_prompt = self.create_enhanced_prompt(question)
            response = self.query_engine.query(enhanced_prompt)
            answer = str(response)
            
            # Check if the answer has errors and try breakdown as fallback
            if any(error_indicator in answer.lower() for error_indicator in [
                "invalid", "error", "not found", "does not exist", "column", "table", "references a column"
            ]):
                log_entry = "⚠️ Direct query had issues, trying LLM breakdown as fallback..."
                print(log_entry)
                logs.append(log_entry)
                
                # Re-analyze with more specific prompt for errors
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
                    
                    return {"answer": combined_answer, "logs": logs}
            
            return {"answer": answer, "logs": logs}
            
        except Exception as e:
            log_entry = f"❌ Query failed: {e}"
            print(log_entry)
            logs.append(log_entry)
            
            # Try a simpler approach
            try:
                log_entry = "🔄 Trying simplified query..."
                print(log_entry)
                logs.append(log_entry)
                
                simplified_question = self.simplify_question(question)
                enhanced_simplified_prompt = self.create_enhanced_prompt(simplified_question)
                response = self.query_engine.query(enhanced_simplified_prompt)
                return {"answer": f"Simplified answer: {response}", "logs": logs}
            except Exception as e2:
                return {"answer": f"❌ Query failed: {e2}", "logs": logs}
    
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
    """Test the smart query engine with single LLM call optimization"""
    print("🧪 Testing Smart Query Engine (Single LLM Call)")
    print("=" * 50)
    
    engine = SmartQueryEngine()
    
    # Test questions of varying complexity
    test_questions = [
        # Simple questions
        "What is the status of order O1001?",
        "How many orders are there?",
        "What is the highest balance?",
        
        # Complex questions
        "Which account has the highest balance, and what is their most recent order?",
        "Which security has the highest management fee, and what is its risk level?",
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
