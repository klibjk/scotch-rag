import sys
import json
import os
from router import Router
from semantic_index import SemanticIndex
from planner import Planner
from executor import Executor
from dotenv import load_dotenv

CARDS_PATH = 'data/cards.json'
MANIFEST_PATH = 'data/manifest.json'

# Helper to map table_id to Parquet path
def load_manifest():
    with open(MANIFEST_PATH, 'r') as f:
        manifest = json.load(f)
    tableid_to_parquet = {entry['table_id']: entry['parquet_path'] for entry in manifest}
    tableid_to_info = {entry['table_id']: entry for entry in manifest}
    return tableid_to_parquet, tableid_to_info

# Helper to map c_... columns to real columns for a given table_id
def get_column_map(cards, table_id):
    col_map = {}
    for card in cards:
        if card['type'] == 'ColumnCard' and card['ids']['table_id'] == table_id:
            c_id = card['ids']['column_id']
            real_col = c_id[2:] if c_id.startswith('c_') else c_id
            col_map[c_id] = real_col
    return col_map

def synthesize_with_llm(question, table_str, provenance, explanations):
    load_dotenv()
    api_key = os.getenv('ANTHROPIC_API_KEY')
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
    # Try DSPy Anthropic first
    try:
        import dspy
        if api_key:
            dspy.settings.configure(
                default_llm=dspy.Anthropic(model='claude-3-haiku-20240307', api_key=api_key)
            )
            response = dspy.Predict(prompt)
            return response.completion if hasattr(response, 'completion') else str(response)
    except Exception as e:
        pass
    # Try anthropic Python SDK
    try:
        import anthropic
        if api_key:
            client = anthropic.Anthropic(api_key=api_key)
            message = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=256,
                temperature=0.2,
                system="You are a data analyst assistant.",
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text if hasattr(message, 'content') else str(message)
    except Exception as e:
        pass
    return "[LLM synthesis unavailable: DSPy Anthropic and anthropic SDK not available or not configured]"

def main():
    if len(sys.argv) < 2:
        print("Usage: python src/agent.py 'Your question here'")
        return
    question = sys.argv[1]
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
    print("\n--- LLM-Generated Plan ---")
    print(json.dumps(plan, indent=2))
    # If plan is a list, use the first element
    if isinstance(plan, list):
        plan = plan[0]
    # 4. Map table_id to Parquet path
    tableid_to_parquet, tableid_to_info = load_manifest()
    table_id = plan['targets'][0]['table_id']
    parquet_path = tableid_to_parquet.get(table_id)
    info = tableid_to_info.get(table_id)
    if not parquet_path:
        print(f"Error: table_id {table_id} not found in manifest.")
        return
    print(f"\n--- Querying file: {info['file_id']} | sheet: {info['sheet_id']} | table: {table_id} ---")
    # 5. Map c_... columns to real columns
    col_map = get_column_map(cards, table_id)
    plan['targets'][0]['columns'] = [col_map.get(col, col) for col in plan['targets'][0]['columns']]
    # 6. Execute
    executor = Executor(parquet_path, table_id)
    df, provenance = executor.run(plan)
    # 7. Print answer table, shape, and columns (always print section)
    print("\n--- Answer Table ---")
    if df is not None and not df.empty:
        print(df)
        print(f"[Shape: {df.shape}, Columns: {list(df.columns)}]")
    else:
        print("[No rows returned. Shape: {} Columns: {}]".format(df.shape if df is not None else 'None', list(df.columns) if df is not None else 'None'))
    # 8. Provenance
    print("\n--- Provenance ---")
    print(json.dumps(provenance, indent=2))
    explanations = "\n".join([
        f"{card['ids']['column_id']}: {card['text']}" for card in relevant_cards if card['type'] == 'ColumnCard' and any(col in card['ids']['column_id'] for col in plan['targets'][0]['columns'])
    ])
    print("\n--- LLM Answer ---")
    llm_answer = synthesize_with_llm(question, df.to_string(index=False) if df is not None else '', provenance, explanations)
    print(llm_answer)

if __name__ == '__main__':
    main()
