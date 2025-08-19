import json
from store import StructuredStore
import re

class Executor:
    def __init__(self, parquet_path, table_id):
        self.store = StructuredStore(parquet_path, table_id)
        self.table_id = table_id

    def run(self, plan: dict):
        print("[SQL EXECUTOR][DEBUG] Full plan:")
        print(json.dumps(plan, indent=2))
        table_id = plan['targets'][0]['table_id']
        columns = plan['targets'][0]['columns']
        join_ops = []
        join_tables = []
        where_clauses = []
        group_by = None
        order_by = None
        limit = None
        having_clause = None
        # Parse ops for JOIN, WHERE, GROUP BY, ORDER BY, LIMIT, HAVING
        for op in plan.get('ops', []):
            print(f"[SQL EXECUTOR][DEBUG] op: {op}")
            if op['type'] == 'JOIN':
                join_ops.append(op)
            if op['type'] == 'WHERE':
                args = op['args']
                if isinstance(args, dict):
                    for k, v in args.items():
                        # Fix for IS NULL/IS NOT NULL
                        if isinstance(v, str) and v.strip().upper() == 'IS NOT NULL':
                            where_clauses.append(f"{k} IS NOT NULL")
                        elif isinstance(v, str) and v.strip().upper() == 'IS NULL':
                            where_clauses.append(f"{k} IS NULL")
                        else:
                            if isinstance(v, str):
                                where_clauses.append(f"{k} = '{v}'")
                            else:
                                where_clauses.append(f"{k} = {v}")
            if op['type'] == 'GROUP_BY':
                group_by = op['args']['by']
            if op['type'] == 'ORDER_BY':
                order_by = op['args']
            if op['type'] == 'LIMIT':
                limit = op['args']['n']
            if op['type'] == 'HAVING':
                # Support HAVING as a dict of conditions (e.g., {"etf_count": "> 1"})
                args = op['args']
                if isinstance(args, dict):
                    having_clauses = []
                    for k, v in args.items():
                        if isinstance(v, str) and v.strip().startswith(('>', '<', '=', '!=')):
                            having_clauses.append(f"{k} {v}")
                        else:
                            having_clauses.append(f"{k} = '{v}'")
                    having_clause = ' AND '.join(having_clauses)
        # Track column aliases in SELECT
        aliases = set()
        for col in columns:
            m = re.search(r'AS\s+([a-zA-Z0-9_]+)', col, re.IGNORECASE)
            if m:
                aliases.add(m.group(1))
        # Load all joined tables in DuckDB
        import glob
        import os
        for join_op in join_ops:
            join_table = join_op['args']['table_id']
            join_tables.append(join_table)
            candidates = glob.glob(f"data/{join_table}.parquet") + glob.glob(f"final_dspy/data/{join_table}.parquet")
            if not candidates:
                print(f"[SQL EXECUTOR][ERROR] Could not find parquet for join table {join_table}")
                raise Exception(f"Could not find parquet for join table {join_table}")
            join_parquet = candidates[0]
            self.store.con.execute(f"CREATE OR REPLACE TABLE {join_table} AS SELECT * FROM read_parquet('{join_parquet}')")
        # Build SQL
        query = f"SELECT {', '.join(columns)} FROM {table_id}"
        for join_op in join_ops:
            join_table = join_op['args']['table_id']
            join_on = join_op['args']['on']
            join_type = join_op['args'].get('join_type', 'INNER').upper()
            if '=' in join_on:
                join_condition = join_on
            else:
                join_condition = f"{table_id}.{join_on} = {join_table}.{join_on}"
            print(f"[SQL EXECUTOR][DEBUG] Adding {join_type} JOIN with {join_table} ON {join_condition}")
            query += f" {join_type} JOIN {join_table} ON {join_condition}"
        if where_clauses:
            query += f" WHERE {' AND '.join(where_clauses)}"
        if group_by:
            if isinstance(group_by, list):
                query += f" GROUP BY {', '.join(group_by)}"
            else:
                query += f" GROUP BY {group_by}"
        if having_clause:
            query += f" HAVING {having_clause}"
        if order_by:
            query += f" ORDER BY {order_by['by']} {order_by['dir']}"
        if limit:
            query += f" LIMIT {limit}"
        print(f"[SQL EXECUTOR] Executing SQL: {query}")

        # --- SQL Validation ---
        # 1. Basic structure
        if not (query.strip().upper().startswith('SELECT') and 'FROM' in query.upper()):
            print("[SQL EXECUTOR][ERROR] SQL missing SELECT or FROM.")
            raise Exception("Generated SQL is missing SELECT or FROM.")
        # 2. Check columns exist in main and joined tables (except for SQL expressions, aliases, and table.column)
        valid_columns = set(self.store.con.execute(f"PRAGMA table_info({table_id})").df()['name'])
        table_columns = {table_id: set(self.store.con.execute(f"PRAGMA table_info({table_id})").df()['name'])}
        for join_table in join_tables:
            join_cols = set(self.store.con.execute(f"PRAGMA table_info({join_table})").df()['name'])
            valid_columns.update(join_cols)
            table_columns[join_table] = join_cols
        valid_columns.update(aliases)  # Allow aliases in validation
        referenced_tables = set([table_id] + join_tables)
        def is_sql_expr(val):
            return isinstance(val, str) and (('(' in val and 'AS' in val) or val.strip().upper().startswith('COUNT(') or val.strip().upper().startswith('SUM(') or val.strip().upper().startswith('AVG(') or val.strip().upper().startswith('MIN(') or val.strip().upper().startswith('MAX('))
        def is_table_col(val):
            if isinstance(val, str) and '.' in val:
                t, c = val.split('.', 1)
                return t in referenced_tables and c in table_columns.get(t, set())
            return False
        # Check SELECT columns
        for col in columns:
            if not is_sql_expr(col) and col not in valid_columns and not is_table_col(col):
                print(f"[SQL EXECUTOR][ERROR] Column '{col}' does not exist in any referenced table.")
                raise Exception(f"Column '{col}' does not exist in any referenced table.")
        # Check WHERE columns
        for clause in where_clauses:
            m = re.match(r"([a-zA-Z0-9_\.]+) =", clause)
            if m:
                col = m.group(1)
                if col not in valid_columns and not is_table_col(col):
                    print(f"[SQL EXECUTOR][ERROR] WHERE column '{col}' does not exist in any referenced table.")
                    raise Exception(f"WHERE column '{col}' does not exist in any referenced table.")
        # Check GROUP BY columns
        if group_by:
            gb_cols = group_by if isinstance(group_by, list) else [group_by]
            for col in gb_cols:
                if col not in valid_columns and not is_table_col(col):
                    print(f"[SQL EXECUTOR][ERROR] GROUP BY column '{col}' does not exist in any referenced table.")
                    raise Exception(f"GROUP BY column '{col}' does not exist in any referenced table.")
        # Check HAVING columns
        if having_clause:
            for alias in aliases:
                if alias in having_clause:
                    break
            else:
                for token in re.findall(r'([a-zA-Z0-9_\.]+)', having_clause):
                    if token not in valid_columns and not is_table_col(token):
                        print(f"[SQL EXECUTOR][ERROR] HAVING column '{token}' does not exist in any referenced table or as an alias.")
                        raise Exception(f"HAVING column '{token}' does not exist in any referenced table or as an alias.")
        # Check ORDER BY column
        if order_by and order_by['by'] not in valid_columns and not is_table_col(order_by['by']) and order_by['by'] != 'count':
            print(f"[SQL EXECUTOR][ERROR] ORDER BY column '{order_by['by']}' does not exist in any referenced table.")
            raise Exception(f"ORDER BY column '{order_by['by']}' does not exist in any referenced table.")
        # 3. Try EXPLAIN
        try:
            self.store.con.execute(f"EXPLAIN {query}")
        except Exception as e:
            print(f"[SQL EXECUTOR][ERROR] SQL EXPLAIN failed: {e}")
            raise Exception(f"SQL validation failed: {e}")
        # --- End SQL Validation ---

        try:
            df = self.store.sql(query)
            print("[SQL EXECUTOR][DEBUG] Query result (first 5 rows):")
            print(df.head())
        except Exception as e:
            print(f"[SQL EXECUTOR][ERROR] SQL execution failed: {e}")
            raise
        provenance = {
            "file_id": plan['targets'][0].get('file_id', None),
            "sheet": plan['targets'][0].get('sheet_id', None),
            "table_id": table_id,
            "row_index_span": f"{df.index.min()}-{df.index.max()}" if not df.empty else "-",
            "order_by": order_by['by'] if order_by else None,
            "executed_sql": query
        }
        return df, provenance
