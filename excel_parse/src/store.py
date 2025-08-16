import duckdb
import pandas as pd

class StructuredStore:
    def __init__(self, parquet_path, table_id):
        self.parquet_path = parquet_path
        self.table_id = table_id
        self.con = duckdb.connect()
        self.con.execute(f"CREATE OR REPLACE TABLE {self.table_id} AS SELECT * FROM read_parquet('{parquet_path}')")

    def top(self, table_id: str, n: int, order_by: str):
        query = f"SELECT * FROM {table_id} ORDER BY {order_by} ASC LIMIT {n}"
        return self.con.execute(query).df()

    def bottom(self, table_id: str, n: int, order_by: str):
        query = f"SELECT * FROM {table_id} ORDER BY {order_by} DESC LIMIT {n}"
        df = self.con.execute(query).df()
        return df.iloc[::-1]  # re-sort ASC for display

    def sql(self, query: str, params: dict | None = None):
        return self.con.execute(query, params or {}).df()

    def get_range(self, sheet_id: str, a1: str):
        # For demo: just return all rows (A1 notation not implemented)
        return self.con.execute(f"SELECT * FROM {self.table_id}").df()

def main():
    store = StructuredStore('data/sample.parquet', 't1')
    print("Top 3 rows:")
    print(store.top('t1', 3, 'row_index_original'))
    print("Bottom 3 rows:")
    print(store.bottom('t1', 3, 'row_index_original'))
    print("SQL query:")
    print(store.sql('SELECT txn_id, amount FROM t1 WHERE amount > 0'))

if __name__ == '__main__':
    main()
