import json
import re

def strict_llm_prompt(question, cards, example_plan=None, join_example=None):
    prompt = f"""
You are a Query Planner for a data analytics agent.
Given:
- User question: {question}
- Available tables/columns (cards): {json.dumps(cards, indent=2)}
- Example QueryPlan format: {json.dumps(example_plan, indent=2) if example_plan else ''}
- Example Join QueryPlan: {json.dumps(join_example, indent=2) if join_example else ''}

Rules:
- ALWAYS output a valid JSON QueryPlan.
- ALWAYS output a plan with a 'table_id' in the 'targets' list.
- Only use columns/tables that exist in the provided list.
- If the question requires joining two tables, pick the most relevant table as the main target and use JOIN ops.
- If the question is ambiguous, make a safe guess or ask for clarification.
- Never fabricate data or columns.
- Never output a plan without a 'table_id'.
- If the question requires aggregation, use GROUP BY and COUNT/SUM/AVG as needed.
- If the question requires a subquery or CTE, try to flatten it into a single SELECT with JOINs if possible.

User question: {question}
"""
    return prompt

# Example aggregation plan for the LLM
example_agg_plan = {
    "mode": "ANALYTIC",
    "targets": [{
        "table_id": "accounts_orders_orders",
        "columns": ["account_id", "COUNT(*) AS count"]
    }],
    "ops": [
        {"type": "WHERE", "args": {"order_type": "buy", "security": "ETF-ABC"}},
        {"type": "GROUP_BY", "args": {"by": ["account_id"]}},
        {"type": "ORDER_BY", "args": {"by": "count", "dir": "DESC"}},
        {"type": "LIMIT", "args": {"n": 1}}
    ],
    "guards": {"read_only": True, "max_rows": 5000}
}

# Example join plan for the LLM
example_join_plan = {
    "mode": "ANALYTIC",
    "targets": [{
        "table_id": "securities_info_holdings",
        "columns": ["etf_id", "holding_security", "weight_percent"]
    }],
    "ops": [
        {"type": "JOIN", "args": {"table_id": "securities_info_securities", "on": "etf_id"}},
        {"type": "ORDER_BY", "args": {"by": "etf_id", "dir": "ASC"}}
    ],
    "guards": {"read_only": True, "max_rows": 5000}
}

class Planner:
    def build(self, text: str, cards: list[dict], mode: str) -> dict:
        import dspy
        llm = dspy.settings.lm
        prompt = strict_llm_prompt(text, cards, example_agg_plan, example_join_plan)
        plan_json = llm(prompt)
        plan = self._parse_plan(plan_json)
        # Fallback: if no table_id, re-prompt with a more direct instruction
        if not self._has_table_id(plan):
            print("[PLANNER][WARN] No table_id found in first plan, re-prompting with direct instruction.")
            prompt2 = strict_llm_prompt(
                text + " (IMPORTANT: Your plan MUST include a 'table_id' in the 'targets' list. If a join is needed, use JOIN ops and pick the most relevant table as the main target.)",
                cards, example_agg_plan, example_join_plan)
            plan_json2 = llm(prompt2)
            plan2 = self._parse_plan(plan_json2)
            if self._has_table_id(plan2):
                return plan2
            print("[PLANNER][ERROR] No table_id found after fallback prompt. Returning original plan.")
        return plan

    def _parse_plan(self, plan_json):
        # If the LLM returns a list, try to extract JSON from any string element
        if isinstance(plan_json, list):
            for item in plan_json:
                if isinstance(item, dict):
                    return item
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
            return plan_json[0]
        if isinstance(plan_json, dict):
            return plan_json
        matches = list(re.finditer(r'\{[\s\S]*\}', plan_json))
        if matches:
            json_str = max((m.group(0) for m in matches), key=len)
            return json.loads(json_str)
        return json.loads(plan_json)

    def _has_table_id(self, plan):
        # Recursively check for table_id in plan
        def search(d):
            if isinstance(d, dict):
                if 'table_id' in d:
                    return True
                for v in d.values():
                    if search(v):
                        return True
            elif isinstance(d, list):
                for v in d:
                    if search(v):
                        return True
            return False
        return search(plan)

# --- Helper: Auto-rewrite plan for invented columns like 'count' ---
def rewrite_plan_for_count(plan, valid_columns):
    def has_count(plan):
        if isinstance(plan, dict):
            for k, v in plan.items():
                if k == 'columns' and isinstance(v, list) and any(col == 'count' or 'COUNT(*)' in col for col in v):
                    return True
                if k == 'by' and (v == 'count' or v == 'COUNT(*)'):
                    return True
                if has_count(v):
                    return True
        elif isinstance(plan, list):
            for v in plan:
                if has_count(v):
                    return True
        return False
    if not has_count(plan):
        return plan
    group_col = None
    if 'targets' in plan and plan['targets']:
        for col in plan['targets'][0].get('columns', []):
            if col != 'count' and 'COUNT(' not in col and col in valid_columns:
                group_col = col
                break
    if not group_col:
        group_col = next(iter(valid_columns), None)
    new_columns = [group_col, 'COUNT(*) AS count'] if group_col else ['COUNT(*) AS count']
    plan['targets'][0]['columns'] = new_columns
    ops = plan.get('ops', [])
    ops = [op for op in ops if not (op.get('type') == 'GROUP_BY' and group_col in op.get('args', {}).get('by', []))]
    ops.insert(0, {"type": "GROUP_BY", "args": {"by": [group_col]}}) if group_col else None
    plan['ops'] = ops
    for op in plan.get('ops', []):
        if op.get('type') == 'ORDER_BY':
            op['args']['by'] = 'count'
    return plan
