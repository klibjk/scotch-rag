import pandas as pd
import json
import os

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
                'text': f"Column '{col}' in table '{table_id}' (sheet '{sheet_id}', file '{file_id}').",
                'facets': {
                    'dtype': str(df[col].dtype),
                    'examples': [str(x) for x in df[col].dropna().unique()[:3]]
                }
            }
            if col == 'amount':
                card['facets']['unit'] = 'THB'
                card['text'] = f"Net amount in THB after discounts (if applicable) in column '{col}' of table '{table_id}'."
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
