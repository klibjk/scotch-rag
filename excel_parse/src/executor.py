import json
from store import StructuredStore

class Executor:
    def __init__(self, parquet_path, table_id):
        self.store = StructuredStore(parquet_path, table_id)
        self.table_id = table_id

    def run(self, plan: dict):
        table_id = plan['targets'][0]['table_id']
        columns = plan['targets'][0]['columns']
        query = f"SELECT {', '.join(columns)} FROM {table_id}"
        order_by = None
        limit = None
        for op in plan.get('ops', []):
            if op['type'] == 'ORDER_BY':
                order_by = op['args']
            if op['type'] == 'LIMIT':
                limit = op['args']['n']
        if order_by:
            query += f" ORDER BY {order_by['by']} {order_by['dir']}"
        if limit:
            query += f" LIMIT {limit}"
        df = self.store.sql(query)
        provenance = {
            "file_id": plan['targets'][0].get('file_id', None),
            "sheet": plan['targets'][0].get('sheet_id', None),
            "table_id": table_id,
            "row_index_span": f"{df.index.min()}-{df.index.max()}" if not df.empty else "-",
            "order_by": order_by['by'] if order_by else None,
            "executed_sql": query
        }
        return df, provenance

def main():
    import planner
    import json
    with open('data/cards.json', 'r') as f:
        cards = json.load(f)
    plan = planner.Planner().build("Show last 10 rows of Transactions and explain amount.", cards, "MIXED")
    executor = Executor('data/sample.parquet', plan['targets'][0]['table_id'])
    df, provenance = executor.run(plan)
    print(df)
    print(json.dumps(provenance, indent=2))

if __name__ == '__main__':
    main()
