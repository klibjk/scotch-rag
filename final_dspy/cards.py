import pandas as pd
import json
import os

def smart_column_description(col, df, file_id, sheet_id, table_id):
    # Business logic mapping for common columns
    mapping = {
        'order_id': "Unique order identifier for tracking and referencing individual trade orders.",
        'account_id': "Account identifier. Links orders to accounts and used for balance tracking.",
        'security': "Security symbol or code (e.g., stock or ETF) involved in the order.",
        'order_type': "Type of order: 'buy' for purchase, 'sell' for sale. Determines trade direction.",
        'quantity': "Number of units (shares, lots, etc.) involved in the order.",
        'price': "Order price per unit of security. Used to calculate total transaction value.",
        'status': "Current status of the order (e.g., pending, filled, cancelled).",
        'placed_time': "Timestamp when the order was placed.",
        'executed_time': "Timestamp when the order was executed (if filled).",
        'owner': "Name of the account owner.",
        'account_type': "Type of account (e.g., individual, joint, retirement).",
        'balance': "Current account balance in THB.",
        'name': "Full name of the security (stock, ETF, etc.).",
        'type': "Type/category of the security (e.g., stock, ETF).",
        'current_price': "Latest available price for the security.",
        'risk_level': "Risk profile of the security (e.g., growth, stability).",
        'holding_security': "Constituent security held within an ETF.",
        'weight_percent': "Percentage weight of the holding in the ETF portfolio.",
        'fee_type': "Type of fee (e.g., management, commission) associated with the security.",
        'amount': "Net amount in THB. Negative values indicate outflows (e.g., purchases), positive values indicate inflows (e.g., sales or credits). Used for settlement and balance updates.",
        'description': "Textual description or notes about the transaction, security, or holding."
    }
    dtype = str(df[col].dtype)
    n_unique = df[col].nunique(dropna=True)
    examples = [str(x) for x in df[col].dropna().unique()[:3]]
    # Categorical values
    if dtype in ('object', 'string', 'category') and n_unique <= 5:
        values = ', '.join([str(x) for x in df[col].dropna().unique()])
        cat_info = f" Possible values: {values}."
    else:
        cat_info = ""
    # Numeric range
    if dtype.startswith('int') or dtype.startswith('float'):
        try:
            min_val = df[col].min()
            max_val = df[col].max()
            range_info = f" Typical range: {min_val} to {max_val}."
        except Exception:
            range_info = ""
    else:
        range_info = ""
    # Unit
    unit_info = ""
    if col == 'amount' and file_id in {'accounts_orders', 'securities_info'}:
        unit_info = " Values are in THB."
    # Business mapping
    if col in mapping:
        desc = mapping[col]
    else:
        desc = f"Column '{col}' in table '{table_id}' (sheet '{sheet_id}', file '{file_id}')."
    # Compose final description
    description = desc + unit_info + cat_info + range_info
    return description.strip()

def generate_cards_from_manifest(manifest_path='data/manifest.json'):
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    cards = []
    for entry in manifest:
        file_id = entry['file_id']
        sheet_id = entry['sheet_id']
        table_id = entry['table_id']
        parquet_path = entry['parquet_path']
        df = pd.read_parquet(parquet_path)
        # SheetCard
        sheet_card = {
            'type': 'SheetCard',
            'scope': 'sheet',
            'ids': {'file_id': file_id, 'sheet_id': sheet_id},
            'text': f"Sheet '{sheet_id}' in file '{file_id}' contains {len(df)} rows and {len(df.columns)-6} columns.",
            'facets': {'n_rows': len(df), 'n_columns': len(df.columns)-6}
        }
        # TableCard
        table_card = {
            'type': 'TableCard',
            'scope': 'table',
            'ids': {'file_id': file_id, 'sheet_id': sheet_id, 'table_id': table_id},
            'text': f"Table '{table_id}' from sheet '{sheet_id}' in file '{file_id}'.",
            'facets': {'n_rows': len(df), 'n_columns': len(df.columns)-6}
        }
        # ColumnCards
        skip_cols = {'row_index_original', 'file_id', 'sheet_id', 'table_id'}
        column_cards = []
        for col in df.columns:
            if col in skip_cols:
                continue
            card = {
                'type': 'ColumnCard',
                'scope': 'column',
                'ids': {'file_id': file_id, 'sheet_id': sheet_id, 'table_id': table_id, 'column_id': f'c_{col}'},
                'text': smart_column_description(col, df, file_id, sheet_id, table_id),
                'facets': {
                    'dtype': str(df[col].dtype),
                    'examples': [str(x) for x in df[col].dropna().unique()[:3]]
                }
            }
            if col == 'amount' and file_id in {'accounts_orders', 'securities_info'}:
                card['facets']['unit'] = 'THB'
            column_cards.append(card)
        cards.extend([sheet_card, table_card] + column_cards)
    return cards

def main():
    cards = generate_cards_from_manifest('data/manifest.json')
    with open('data/cards.json', 'w') as f:
        json.dump(cards, f, indent=2)
    print(f"Generated {len(cards)} cards for all files and sheets.")

if __name__ == '__main__':
    main()
