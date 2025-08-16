import json
import os
import re
from dotenv import load_dotenv

def configure_dspy():
    load_dotenv()
    import dspy
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    if anthropic_key:
        try:
            print("🔵 Using Anthropic Claude as primary LLM for planning")
            lm = dspy.LM("anthropic/claude-3-haiku-20240307", api_key=anthropic_key)
            dspy.configure(lm=lm)
            return "anthropic"
        except Exception as e:
            print(f"Anthropic config failed: {e}")
    if openai_key:
        try:
            print("🟢 Using OpenAI GPT-4 as fallback LLM for planning")
            lm = dspy.LM("openai/gpt-4", api_key=openai_key)
            dspy.configure(lm=lm)
            return "openai"
        except Exception as e:
            print(f"OpenAI config failed: {e}")
    raise Exception("No valid LLM configuration found. Please set your API keys.")

def strict_llm_prompt(question, cards, example_plan=None):
    prompt = f"""
You are a Query Planner for a data analytics agent.
Given:
- User question: {question}
- Available tables/columns (cards): {json.dumps(cards, indent=2)}
- Example QueryPlan format: {json.dumps(example_plan, indent=2) if example_plan else ''}

Rules:
- Always output a valid JSON QueryPlan.
- Only use columns/tables that exist in the provided list.
- If the question is ambiguous, make a safe guess or ask for clarification.
- Never fabricate data or columns.

User question: {question}
"""
    return prompt

class Planner:
    def build(self, text: str, cards: list[dict], mode: str) -> dict:
        # Configure DSPy (Anthropic preferred, OpenAI fallback)
        try:
            provider = configure_dspy()
        except Exception as e:
            raise RuntimeError(f"No LLM available for planning. {e}")
        import dspy
        llm = dspy.settings.lm
        example_plan = {
            "mode": "ANALYTIC",
            "targets": [{"table_id": "t1", "columns": ["txn_id", "date", "description", "amount"]}],
            "ops": [
                {"type": "ORDER_BY", "args": {"by": "date", "dir": "DESC"}},
                {"type": "LIMIT", "args": {"n": 1}}
            ],
            "guards": {"read_only": True, "max_rows": 5000}
        }
        prompt = strict_llm_prompt(text, cards, example_plan)
        plan_json = llm(prompt)
        # If the LLM returns a list, try to extract JSON from any string element
        if isinstance(plan_json, list):
            # Try to find a dict in the list
            for item in plan_json:
                if isinstance(item, dict):
                    return item
            # Try to extract JSON from string elements
            for item in plan_json:
                if isinstance(item, str):
                    matches = list(re.finditer(r'\{[\s\S]*\}', item))
                    if matches:
                        json_str = max((m.group(0) for m in matches), key=len)
                        return json.loads(json_str)
                    try:
                        return json.loads(item)
                    except Exception:
                        continue
            # Fallback: return the first element if possible
            return plan_json[0]
        if isinstance(plan_json, dict):
            return plan_json
        # If it's a string, try to extract JSON
        matches = list(re.finditer(r'\{[\s\S]*\}', plan_json))
        if matches:
            json_str = max((m.group(0) for m in matches), key=len)
            return json.loads(json_str)
        # Fallback: try to parse the whole string
        return json.loads(plan_json)

def main():
    from semantic_index import SemanticIndex
    import json
    with open('data/cards.json', 'r') as f:
        cards = json.load(f)
    planner = Planner()
    plan = planner.build("I want latest transaction show me only 1", cards, "ANALYTIC")
    print(json.dumps(plan, indent=2))

if __name__ == '__main__':
    main()
